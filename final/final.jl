# make sure we're in the right place
cd(@__DIR__)
using Pkg
pkg"activate .."

include("basics.jl")
include("helpers.jl")
include("simulation.jl")
include("gp.jl")
include("render.jl")


"""Observes the position of a particle with noise."""
@gen (static) function observe_position(position :: Point2d, noise :: Float64) :: Point2d
    x ~ Gen.normal(position[1], noise)
    y ~ Gen.normal(position[2], noise)
    return Point2d(x, y)
end

"""
Samples a particle, simulates its interaction with the grid, observes its position at every time step with noise,
and returns its TRUE positions and bounces. Note that observations are different from return value!
"""
@gen (static) function random_particle(observe_noise :: Float64,
    forces :: Grid{Vec2d},
    timesteps :: Int64,
    p :: ExtraParams) :: Tuple{Vector{Point2d}, Vector{Vec2{Bounce}}}

    initial_x ~ Gen.uniform(forces.xrange[1], forces.xrange[2])
    initial_y ~ Gen.uniform(forces.yrange[1], forces.yrange[2])

    initial_vel = Vec2d(0.0, 0.0)
    initial_mass = 1.0

    true_positions, true_bounces = run_particle(Point2d(initial_x, initial_y), initial_vel, initial_mass;
        forces=forces, timesteps=timesteps, p=p)

    noises = [observe_noise for t in 1:timesteps]
    observations ~ Gen.Map(observe_position)(true_positions, noises)

    return (true_positions, true_bounces)
end

"""Our full prior."""
@gen (static) function force_model(
                          width :: Float64,
                          res :: Int64,
                          n_particles :: Int64,
                          force_scale :: Float64,
                          timesteps :: Int64,
                          p :: ExtraParams) :: Vector{Tuple{Vector{Point2d}, Vector{Vec2{Bounce}}}}
    ## Build grid
    # Sample a length scale
    length_scale ~ gamma_bounded_below(1, width/10, width * 0.01)
    # Sample a global noise level
    noise ~ gamma_bounded_below(1, force_scale, 0.01)

    # Always use square grid
    bounds = (0.0, width)
    centers = grid_centers(bounds, bounds, res, res)

    # Sample values
    force_xs ~ grid_model(centers, length_scale, noise)
    force_ys ~ grid_model(centers, length_scale, noise)
    forces = grids_to_vec_grid(centers, force_xs, force_ys, bounds, bounds, res, res)

    ## Noise in velocity / position observations
    observe_noise = width / 500.0

    ## Sample a bunch of masses, compute their paths, then observe their
    ## positions
    mass_paths ~ Gen.Map(random_particle)(
        [observe_noise for i in 1:n_particles],
        [forces for i in 1:n_particles],
        [timesteps for i in 1:n_particles],
        [p for i in 1:n_particles])

    return mass_paths
end

Gen.@load_generated_functions

"""Pull the sampled force grid out of a trace of our prior."""
function read_grid(trace) :: Grid{Vec2d}
    args = Gen.get_args(trace)
    if args[1] isa Gen.Trace
        # this is a grid proposal; pull from its source trace
        width, res, n_particles, force_scale, timesteps, p = Gen.get_args(args[1])
    else
        # this is a normal trace
        width, res, n_particles, force_scale, timesteps, p = args
    end
    bounds = (0.0, width)
    centers = grid_centers(bounds, bounds, res, res)

    grids_to_vec_grid(centers, trace[:force_xs], trace[:force_ys], bounds, bounds, res, res)
end

"""Pull the *observed* (i.e. noisy) positions out of a trace of our prior."""
function read_observations(trace) :: PositionArray
    width, res, n_particles, force_scale, timesteps, p = Gen.get_args(trace)
    observations = Matrix{Point2d}(undef, timesteps, n_particles)

    for i in 1:n_particles
        for t in 1:timesteps
            observations[t, i] = trace[:mass_paths => i => :observations => t]
        end
    end

    observations
end

"""Pull the true positions out of a trace of our prior.
They are "true" in the sense that they aren't noisy; they follow deterministically
from the chosen force field and starting particles. Note, however, those might not
correspond to the force field in something you're observing!"""
function read_true_positions_bounces(trace) :: Tuple{PositionArray, Array{Vec2{Bounce}}}
    width, res, n_particles, force_scale, timesteps, p = Gen.get_args(trace)
    positions = Matrix{Point2d}(undef, timesteps, n_particles)
    bounces = Matrix{Vec2{Bounce}}(undef, timesteps, n_particles)

    for i in 1:n_particles
        pp, bb = trace[:mass_paths => i]
        positions[:, i] = pp
        bounces[:, i] = bb
    end

    positions, bounces
end

@testset "trace reading" begin
    res = 5
    n_particles = 3
    timesteps = 24

    trace = Gen.simulate(force_model, (1.0, res, n_particles, 0.1, timesteps, ExtraParams(1.0/24, 0.9)))
    pp, bb = read_true_positions_bounces(trace)
    @test size(pp, 1) == timesteps
    @test size(pp, 2) == n_particles
    @test size(pp) == size(bb)

    oo = read_observations(trace)
    @test size(oo) == size(bb)

    gg = read_grid(trace)
    @test size(gg.values) == (res, res)
end

"""Add observations (or non-noisy paths, they're the same type) to a choicemap."""
function add_observations!(constraints :: Gen.ChoiceMap, observations :: PositionArray)
    timesteps, n_particles = size(observations)
    for i in 1:n_particles
        for t in 1:timesteps
            constraints[:mass_paths => i => :observations => t => :x] = observations[t, i][1]
            constraints[:mass_paths => i => :observations => t => :y] = observations[t, i][2]
        end
    end
end

@gen (static) function simple_mcmc_proposal(trace)
    width, res, n_particles, force_scale, timesteps, p = Gen.get_args(trace)
    bounds = (0.0, width)
    centers = grid_centers(bounds, bounds, res, res)
    noise = trace[:noise]
    length_scale = trace[:length_scale]

    force_xs ~ grid_model(centers, length_scale, noise)
    force_ys ~ grid_model(centers, length_scale, noise)
end

Gen.@load_generated_functions

"""Compute (velocities, forces) using naive second differences and bounce information.
Returned arrays are indexed such that the velocities / forces were applied in the position on the corresponding timestep.
The last timestep of forces is empty.
"""
function second_differences(observations :: PositionArray, bounces :: Matrix{Vec2{Bounce}}, masses :: Vector{Float64};
    xrange :: Bounds, yrange :: Bounds, p :: ExtraParams) :: Tuple{Matrix{Vec2d}, Matrix{Vec2d}}

    # TODO: allow specifying initialization
    velocities = Matrix{Vec2d}(undef, size(observations))
    forces = Matrix{Vec2d}(undef, size(observations))

    timesteps, n_particles = size(observations)

    for i in 1:n_particles
        velocities[1, i] = Vec2d(0.0, 0.0)
        forces[1, i] = Vec2d(0.0, 0.0)
    end

    for i in 1:n_particles
        for t in 2:timesteps
            pos = observations[t-1, i]
            vel = velocities[t-1, i]
            new_pos = observations[t, i]
            bounce = bounces[t, i]

            nvx, ax = invert_step_dim(pos[1], vel[1], new_pos[1], bounce[1], xrange, p)
            nvy, ay = invert_step_dim(pos[2], vel[2], new_pos[2], bounce[2], yrange, p)

            velocities[t, i] = Vec2d(nvx, nvy)
            forces[t, i] = Vec2d(ax, ay) * masses[i]
        end
    end

    (velocities, forces)
end

"""Uses second differences to compute observed accelerations at every point on a grid. Returns a list of observed forces
at each location."""
function accumulate_grid(positions :: PositionArray, forces :: Matrix{Vec2d};
        xres :: Int64, yres :: Int64, xrange :: Bounds, yrange :: Bounds) :: Grid{RunningStats{Vec2d}}
    timesteps, n_particles = size(forces)

    force_grid = Grid(zeros(RunningStats{Vec2d}, xres, yres), xrange, yrange)

    for i in 1:n_particles
        for t in 2:timesteps
            pos = positions[t-1, i]
            F = forces[t, i]
            I = real_to_index(force_grid, pos[1], pos[2])
            force_grid.values[I] = update_stats(force_grid.values[I], F)
        end
    end

    force_grid
end

@testset "accumulate_grid" begin
    n_particles = 12
    masses = [1.0 for i in 1:n_particles]
    p = ExtraParams(1.0/24, 0.9)
    timesteps = 48
    res = 10
    width = 1.0
    bounds = (0.0, 1.0)
    trace = Gen.simulate(force_model, (width, res, n_particles, 0.1, timesteps, p))

    true_grid = read_grid(trace)
    positions, bounces = read_true_positions_bounces(trace)

    vels, forces = second_differences(positions, bounces, masses, xrange=bounds, yrange=bounds, p=p)
    for i in 1:n_particles
        for t in 2:timesteps
            if bounces[t, i] == NONE
                @assert isapprox(vels[t, i], (positions[t, i] - positions[t-1, i]))
                @assert isapprox(forces[t, i] / masses[i], vels[t, i] - vels[t-1, i])
            end
        end
    end

    forcegrid = accumulate_grid(positions, forces, xres=10, yres=10, xrange=(0.0, 1.0), yrange=(0.0, 1.0))

    @test sum([z.count for z in forcegrid.values]) == length(positions[2:end, :])

    has_nonzero = false
    for I in CartesianIndices(forcegrid.values)
        mean_, var_ = complete(forcegrid.values[I])
        if !isnan(mean_)
            has_nonzero = true
            # we're using the true values so this should be the case
            @assert isapprox(mean_, true_grid.values[I], atol=0.00001) "$mean_ $(true_grid.values[I])"
        end
    end
    @test has_nonzero == true
end

@gen function condition_wacky_and_sample(
    noise :: Float64,
    length_scale :: Float64,
    known_locs :: Vector{Point2d},
    known_means :: Vector{Float64},
    known_vars :: Vector{Float64},
    unknown_locs :: Vector{Point2d},
    inv_perm :: Vector{Int32}
    ) :: Vector{Float64}

    @assert !any(isnan.(known_locs))
    @assert !any(isnan.(known_means))
    @assert !any(isnan.(known_vars))

    # compute the means + covariance of the unknown region
    unknown_cond_means, unknown_cond_cov = compute_predictive(
        make_cov_vectorized(length_scale), noise, known_locs, known_means, unknown_locs
    )
#    @assert !any(isnan.(unknown_cond_means)) "$unknown_cond_means"
#    @assert !any(isnan.(unknown_cond_cov)) "$unknown_cond_cov"
#
    known_count = length(known_locs)
    unknown_count = length(unknown_locs)
    res = known_count + unknown_count

    # mean vector
    means = zeros(res)
    means[1:known_count] = known_means # observed
    means[known_count+1:end] = unknown_cond_means # inferred

    #@assert !any(isnan.(means))

    # covariance matrix
    cov = zeros(res, res)
    # use measured variance for the observed part of the matrix.
    # this isn't technically inferred, but hey, we're in an MCMC proposal, anything goes.
    cov[1:known_count, 1:known_count] = diagm((known_vars .* .1) .+ eps()) # ensure positive definite
    #cov[1:known_count, 1:known_count] = I(known_count) * noise * 0.1 # we're more confident about this
    # and use inferred covariance for the rest of the matrix.
    cov[known_count+1:end, known_count+1:end] .= unknown_cond_cov
    # ignore the matrix corners, the two chunks don't interact.

    # TODO undebug
    #means[known_count+1:end] .= 0.0
    #cov[:, :] .= I(res) .* eps()

#    @assert !any(isnan.(cov))
#    @assert issymmetric(cov)
#    @assert isposdef(cov) "$cov"
#
    means_orig = means[inv_perm]
    cov_orig = cov[inv_perm, inv_perm]

    @assert isposdef(cov_orig)

    vals ~ mvnormal(means_orig, cov_orig)

    # return the full matrix
    return vals
end

@gen function second_diff_proposal(trace, observations, bounces, smooth)
    width, res, _, force_scale, _, p = Gen.get_args(trace)
    # for visualization, allow mismatching size w/ trace
    timesteps, n_particles = size(observations)

    masses = [1.0 for i in 1:n_particles]

    # get inferred scale properties
    length_scale = trace[:length_scale]
    noise = trace[:noise]

    bounds = (0.0, width)

    times = range(0.0, length=timesteps, step=p.dt)

    _, forces = second_differences(observations, bounces, masses, xrange=bounds, yrange=bounds, p=p)

    forces[2:end, :] = smooth(times[2:end], forces[2:end, :])

    # figure out where we've seen particles accelerate
    F_grid = accumulate_grid(observations, forces,
        xres=res, yres=res, xrange=bounds, yrange=bounds)

    # get vectors of position / observed value
    Is = reshape(CartesianIndices(F_grid.values), :)
    locs = [index_to_center(F_grid, I[1], I[2]) for I in Is]
    F_means = [complete(F_grid.values[I])[1] for I in Is]
    F_vars = [complete(F_grid.values[I])[2] for I in Is]

    # whether we've seen anything for a location or not
    known_mask = (x -> !isnan(x)).(F_means)

    # make permutation to rearrange
    known_count, perm = split_permutation(known_mask)
    inv_perm = invert_permutation(perm)

    # permute locs, split into known and unknown chunks
    locs_p = locs[perm]
    known_locs_p = locs_p[1:known_count]
    unknown_locs_p = locs_p[known_count+1:end]

    # permute our measurements, extract known chunks
    known_F_means_p = F_means[perm[1:known_count]]
    known_F_vars_p = F_vars[perm[1:known_count]]

    @assert !any(isnan.(known_F_means_p)) "$F_means $(perm[1:known_count])"

    # force x component
    force_xs ~ condition_wacky_and_sample(
        noise,
        length_scale,
        known_locs_p,
        [F_mean[1] for F_mean in known_F_means_p],
        [F_var[1] for F_var in known_F_vars_p],
        unknown_locs_p,
        inv_perm
        )

    @assert !any(isnan.(force_xs))

    # force y component
    force_ys ~ condition_wacky_and_sample(
        noise,
        length_scale,
        known_locs_p,
        [F_mean[2] for F_mean in known_F_means_p],
        [F_var[2] for F_var in known_F_vars_p],
        unknown_locs_p,
        inv_perm
        )

    @assert !any(isnan.(force_ys))

end

"""Apply no smoothing."""
nosmooth(ts, Fs) = Fs

@testset "basic second differences proposal" begin
    n = 3
    trace = Gen.simulate(force_model, (1.0, 4, n, 0.1, 5, ExtraParams(1.0/24, 0.9)))

    pp, bb = read_true_positions_bounces(trace)

    zz = Gen.simulate(second_diff_proposal, (trace, pp, bb, nosmooth))

    @test size(zz[:force_xs]) == (4*4,)
    @test size(zz[:force_ys]) == (4*4,)
end

"""Run simple MCMC, with everything fixed except the force grid, and a custom proposal distribution."""
function simple_mcmc(constraints :: Gen.ChoiceMap, prop, prop_args, args :: Tuple; computation = 100, cb=nothing)
    trace, _ = Gen.generate(force_model, args, constraints)
    if cb != nothing
        cb(trace, 0)
    end

    weights = Float64[]

    for step in 1:computation
        (trace, w) = metropolis_hastings(trace, prop, prop_args) #, check=true, observations=constraints)
        #(trace, w) = metropolis_hastings(trace, select(:force_xs, :force_ys)) #, check=true, observations=constraints)
        if cb != nothing
            cb(trace, step)
        end
        push!(weights, w)
    end

    trace, weights
end

@testset "basic mcmc second diff" begin
    n_particles = 3
    width = 1.0
    res = 10
    cc = choicemap()
    cc[:length_scale] = 0.1
    cc[:noise] = 0.1
    for i in 1:n_particles
        cc[:mass_paths => i => :initial_x] = uniform(0.0, width)
        cc[:mass_paths => i => :initial_y] = uniform(0.0, width)
    end

    args = (width, res, n_particles, 0.1, 48, ExtraParams(1.0/24, 0.9))
    trace, _ = Gen.generate(force_model, args, cc)
    positions, bounces = read_true_positions_bounces(trace)
    add_observations!(cc, positions) # note: we use noiseless recordings here

    zz, _ = simple_mcmc(cc, simple_mcmc_proposal, (), args, computation=2)
    @test read_observations(zz) == positions
end

CUBIC = [
    t -> 1.0,
    t -> t,
    t -> t^2,
    t -> t^3
]
# times: [t], values: [t, n], components: [c] -> [t, n]
function least_squares(times :: AbstractVector{T}, values :: AbstractArray{V}, components=CUBIC) :: Array{V} where {T, V}
    A = zeros(V, length(times), length(components))
    for (i, c) in enumerate(components)
        A[:, i] = c.(times)
    end
    p = A \ values
    A * p
end
function vec_least_squares(times, values, components=CUBIC)
    xs = least_squares(times, [v[1] for v in values], components)
    ys = least_squares(times, [v[2] for v in values], components)
    [Vec2d(x, y) for (x,y) in zip(xs, ys)]
end

@testset "vec_least_squares" begin
    Random.seed!(0)

    times = 1:.1:2pi
    points = [Vec2d(sin(t) * 5, cos(t) * 5) for t in times]
    points += .1 * randn(Vec2d, size(points))

    psol = vec_least_squares(times, points)

    @test mean(magnitude.(psol .- points)) < 1.0

    multipoints = hcat(points, points, points) + randn(Vec2d, length(points), 3)
    psols = vec_least_squares(times, multipoints)

    @test size(psols) == (length(points), 3)
end

function hard_mcmc(constraints :: Gen.ChoiceMap, prop, prop_args, args :: Tuple; computation = 100, cb=nothing)
    trace, _ = Gen.generate(force_model, args, constraints)
    if cb != nothing
        cb(trace, 0)
    end

    for step in 1:computation
        (trace, _) = metropolis_hastings(trace, prop, prop_args) #, check=true, observations=constraints)
        (trace, _) = metropolis_hastings(trace, select(:length_scale)) #, check=true, observations=constraints)
        (trace, _) = metropolis_hastings(trace, select(:noise)) #, check=true, observations=constraints)

        n_particles = Gen.get_args(trace)[3]
        for i in 1:n_particles
            (trace, _) = metropolis_hastings(trace, select(
                :mass_paths => i => :initial_x,
                :mass_paths => i => :initial_y
            ))
        end
        if cb != nothing
            cb(trace, step)
        end
    end

    trace
end

##

# ~~ static rendering ~~

n = 12
mm = [1.0 for i in 1:n]
tt = Gen.simulate(force_model, (1.0, 10, n, 0.1, 48, ExtraParams(1.0/24, 0.9)))
gg = read_grid(tt)
pp, bb = read_true_positions_bounces(tt)
oo = read_observations(tt)

s = Scene(resolution=(1600, 1600), show_axis=false, show_grid=false)
draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)
draw_particle_paths!(s, mm, pp, mass_scale=0.01, colormod=c -> RGBA(c, .5))
draw_particle_paths!(s, mm, oo, mass_scale=0.01)
display(s)

##

# ~~ vector least squares ~~

times = 1:.1:2pi
points = [Vec2d(sin(t) * 5, cos(t) * 5) for t in times]
points += .1 * randn(Vec2d, size(points))
s = scatter(Point2d.(points), resolution=(1200, 1200))
psol = vec_least_squares(times, points)
lines!(s, Point2d.(psol))


##

# ~~ static, second differences of noise ~~

n = 12
mm = [1.0 for i in 1:n]
p = ExtraParams(1.0/24, 0.9)
tt = Gen.simulate(force_model, (1.0, 10, n, 0.1, 48, p))
gg = read_grid(tt)
pp, bb = read_true_positions_bounces(tt)
oo = read_observations(tt)

s = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)
#draw_particle_paths!(s, mm, pp, mass_scale=0.01, colormod=c -> RGBA(c, .5))
#draw_particle_paths!(s, mm, pp, mass_scale=0.01)
draw_particle_paths!(s, mm, oo, mass_scale=0.01)

timesteps = size(oo, 1)

true_vel, true_acc = second_differences(pp, bb, mm, xrange=(0.0, 1.0), yrange=(0.0, 1.0), p=p)
noisy_vel, noisy_acc = second_differences(oo, bb, mm, xrange=(0.0, 1.0), yrange=(0.0, 1.0), p=p)
times = range(0.0, length=timesteps, step=p.dt)
smoothed_acc = vec_least_squares(times, noisy_acc)

reduce_amt = 12

colors = map(i -> get(ColorSchemes.rainbow, (i-1)/n), 1:n)
for i in 1:n
    #arrows!(s, pp[1:reduce_amt:end-1, i], true_acc[1:reduce_amt:end-1, i] / 24.0, arrowcolor=:red, linecolor=:red, arrowsize=0.01)
    #arrows!(s, oo[1:reduce_amt:end-1, i], noisy_acc[1:reduce_amt:end-1, i] / 24.0, arrowcolor=:red, linecolor=:red, arrowsize=0.01)
    arrows!(s, oo[1:reduce_amt:end-1, i], smoothed_acc[1:reduce_amt:end-1, i] / 24.0, arrowcolor=:red, linecolor=:red, arrowsize=0.01)
    #arrows!(s, pp[1:end-1, i], noisy_vel[1:end-1, i] / 24.0, arrowcolor=colors[i], linecolor=colors[i], arrowsize=0.01)
    #arrows!(s, pp[1:end-1, i], noisy_vel[1:end-1, i] / 24.0, arrowcolor=colors[i], linecolor=colors[i], arrowsize=0.01)
end

display(s)


##

# ~~ dynamic rendering ~~

n = 12
mm = [1.0 for i in 1:n]
tt = Gen.simulate(force_model, (1.0, 10, n, 0.1, 48, ExtraParams(1.0/24, 0.9)))
gg = read_grid(tt)
pp, _ = read_true_positions_bounces(tt)
oo = read_observations(tt)

s = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
t = Makie.Observable(1)

draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)
# true positions, low alpha
animate_particles!(s, mm, pp, t, mass_scale=0.02, colormod=c -> RGBA(c, .5))
# false positions, high alpha
animate_particles!(s, mm, oo, t, mass_scale=0.02)

updateloop(s) do
    t[] = (t[] % size(pp, 1)) + 1
end

##

# ~~ mcmc w/ true paths, all other things fixed ~~

n = 3
w = 1.0
r = 10
cc = choicemap()
cc[:length_scale] = 0.1
cc[:noise] = 0.1
for i in 1:n
    cc[:mass_paths => i => :initial_x] = uniform(0.0, w)
    cc[:mass_paths => i => :initial_y] = uniform(0.0, w)
end

args = (w, r, n, 0.1, 48, ExtraParams(1.0/24, 0.9))

tt, _ = Gen.generate(force_model, args, cc)
gg = read_grid(tt)
pp, _ = read_true_positions_bounces(tt)
add_observations!(cc, pp) # note: we use noiseless recordings here

mm = [1.0 for i in 1:n]

s = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)

guess_grid = draw_grid!(s, gg, arrowcolor=:lightblue, linecolor=:lightblue)
draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)

draw_particle_paths!(s, mm, pp, mass_scale=0.01, colormod=c -> RGBA(c, .5))
ll, ss = draw_particle_paths!(s, mm, pp .* 0.0, mass_scale=0.01)

text!(s, "booting...", transparency=false, textsize=0.03, position=(0.0, w), align=(:left, :top))
status = s[end]

display(s)

simple_mcmc(cc, simple_mcmc_proposal, (), args, computation=1000, cb=debounce(s, function(trace, step)
    gg = read_grid(trace)
    guess_grid[:directions] = reshape(gg.values./24, :)

    oo, _ = read_true_positions_bounces(trace)
    for i in 1:size(oo, 2)
        ll[i][:positions] = oo[:, i]
    end
    ss[:positions] = oo[1, :]
    status[:text] = "mcmc step $step"
end))

##

# ~~ second differences proposal visualization ~~

r = 10
w = 1.0
p = ExtraParams(1.0/24, 0.9)
n = 25
mm = [1.0 for i in 1:n]
timesteps = 48
tt = Gen.simulate(force_model, (w, r, n, 0.1, timesteps, p))
gg = read_grid(tt)
pp, bb = read_true_positions_bounces(tt)
oo = read_observations(tt)

s = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
t = Makie.Observable(1)

draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)
obs_grid = draw_grid!(s, map_grid(x -> Vec2d(0.0), Vec2d, gg), arrowcolor=:red, linecolor=:red)
prop_grid = draw_grid!(s, map_grid(x -> Vec2d(0.0), Vec2d, gg), arrowcolor=:blue, linecolor=:blue)
#prop_grid[:visible] = false

# true positions, low alpha
animate_particles!(s, mm, pp, t, mass_scale=0.02, colormod=c -> RGBA(c, .5))
# false positions, high alpha
animate_particles!(s, mm, oo, t, mass_scale=0.02)

function vizmean(s :: RunningStats{Vec2d}) :: Vec2d
    mean_, var_ = complete(s)
    if isnan(mean_)
        Vec2d(0)
    else
        mean_
    end
end

second_diffs_aot = [begin
        _, forces = second_differences(oo[1:t_, :], bb[1:t_, :], mm;  xrange=(0.0, w), yrange=(0.0, w), p=p)
        times = range(0.0, length=t_, step=p.dt)
        forces[2:end, :] = vec_least_squares(times[2:end], forces[2:end, :])
        accumulate_grid(oo[1:t_, :], forces; xres=r, yres=r, xrange=(0.0, w), yrange=(0.0, w))
    end
    for t_ in 1:size(oo, 1)
]
diff_arrows_aot = [
    [vizmean(s) / 24.0 for s in reshape(second_diff.values, :)]
    for second_diff in second_diffs_aot
]

prop_grids_aot = [
    #read_grid(Gen.simulate(second_diff_proposal, (tt, oo[1:t_, :], bb[1:t_, :], nosmooth)))
    read_grid(Gen.simulate(second_diff_proposal, (tt, oo[1:t_, :], bb[1:t_, :], vec_least_squares)))
    for t_ in 1:size(oo, 1)
]
prop_arrows_aot = [
    [f / 24.0 for f in reshape(g.values, :)]
    for g in prop_grids_aot
]

updateloop(s) do
    t[] = (t[] % size(pp, 1)) + 1
    obs_grid[:directions][] = diff_arrows_aot[t[]]
    prop_grid[:directions][] = prop_arrows_aot[t[]]
end



##

# ~~ mcmc w/ noisy paths, all other things fixed ~~

n = 10
w = 1.0
r = 10
cc = choicemap()
cc[:length_scale] = 0.1
cc[:noise] = 0.1
for i in 1:n
    cc[:mass_paths => i => :initial_x] = uniform(0.0, w)
    cc[:mass_paths => i => :initial_y] = uniform(0.0, w)
end

args = (w, r, n, 0.1, 48, ExtraParams(1.0/24, 0.9))

tt, _ = Gen.generate(force_model, args, cc)
gg = read_grid(tt)
oo = read_observations(tt)
pp, bb = read_true_positions_bounces(tt)
add_observations!(cc, pp) # note: we use noiseless recordings here

mm = [1.0 for i in 1:n]

s = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)

guess_grid = draw_grid!(s, gg, arrowcolor=:lightblue, linecolor=:lightblue)
draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)

draw_particle_paths!(s, mm, pp, mass_scale=0.01, colormod=c -> RGBA(c, .5))
ll, ss = draw_particle_paths!(s, mm, pp .* 0.0, mass_scale=0.01)

text!(s, "booting...", transparency=false, textsize=0.03, position=(0.0, w), align=(:left, :top))
status = s[end]

display(s)

simple_mcmc(cc, second_diff_proposal, (oo, bb, vec_least_squares), args, computation=1000, cb=debounce(s, function(trace, step)
    gg = read_grid(trace)
    guess_grid[:directions] = reshape(gg.values./24, :)

    oo, _ = read_true_positions_bounces(trace)
    for i in 1:size(oo, 2)
        ll[i][:positions] = oo[:, i]
    end
    ss[:positions] = oo[1, :]
    status[:text] = "mcmc step $step"
end))

##

# ~~ mcmc: force + length_scale + noise

n = 20
w = 1.0
r = 10
cc = choicemap()
#for i in 1:n
#    cc[:mass_paths => i => :initial_x] = uniform(0.0, w)
#    cc[:mass_paths => i => :initial_y] = uniform(0.0, w)
#end
#
p = ExtraParams(1.0/24, 0.9)
args = (w, r, n, 0.1, 48, p)

tt, _ = Gen.generate(force_model, args, cc)
gg = read_grid(tt)
oo = read_observations(tt)
pp, bb = read_true_positions_bounces(tt)
add_observations!(cc, pp) # note: we use noiseless recordings here

mm = [1.0 for i in 1:n]

s = Scene(resolution=(1600, 1600), show_axis=false, show_grid=false)

guess_grid = draw_grid!(s, gg, arrowcolor=:lightblue, linecolor=:lightblue)
draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)

draw_particle_paths!(s, mm, pp, mass_scale=0.01, colormod=c -> RGBA(c, .5))
ll, ss = draw_particle_paths!(s, mm, pp .* 0.0, mass_scale=0.01)

text!(s, "booting...", transparency=false, textsize=0.03, position=(0.0, w), align=(:left, :top))
status = s[end]

display(s)

hard_mcmc(cc, second_diff_proposal, (oo, bb, vec_least_squares), args, computation=1000, cb=debounce(s, function(trace, step)
    gg = read_grid(trace)
    guess_grid[:directions] = reshape(gg.values./24, :)

    oo, _ = read_true_positions_bounces(trace)
    for i in 1:size(oo, 2)
        ll[i][:positions] = oo[:, i]
    end
    ss[:positions] = oo[1, :]
    status[:text] = "mcmc step $step"

    #sleep(0.1)
end))


##

n = 20
w = 1.0
r = 10
cc = choicemap()
#
p = ExtraParams(1.0/24, 0.9)
args = (w, r, n, 0.1, 48, p)

tt, _ = Gen.generate(force_model, args, cc)
oo = read_observations(tt)
gg = read_grid(tt)
pp, bb = read_true_positions_bounces(tt)
add_observations!(cc, pp) # note: we use noiseless recordings here

hard_mcmc(cc, second_diff_proposal, (oo, bb, vec_least_squares), args, computation=2)

using ProfileSVG
ProfileSVG.@profview hard_mcmc(cc, second_diff_proposal, (oo, bb, vec_least_squares), args, computation=100)

##

println(@code_native step_dim(1.0, 1.0, 1.0, (0.0, 1.0), p))

##

const __grid = gg
const __p = p

println(@code_native step_particle(Point2d(0.5, 0.5), Vec2d(0.5, 0.5), 1.0, __grid, __p))

##

println(@code_native sample_grid(gg, 1.0, 1.0))

##

println(@code_typed run_particle(Point2d(0.5, 0.5), Vec2d(0.5, 0.5), 1.0, forces=gg, timesteps=64, p=p))

##
println(@code_typed Array{Float64}(undef, 1))
##
