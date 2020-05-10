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
    width, res, n_particles, force_scale, timesteps, p = Gen.get_args(trace)
    bounds = (0.0, width)
    centers = grid_centers(bounds, bounds, res, res)

    grids_to_vec_grid(centers, trace[:force_xs], trace[:force_ys], bounds, bounds, res, res)
end

"""Pull the *observed* (i.e. noisy) positions out of a trace of our prior."""
function read_observations(trace) :: PositionArray
    width, res, n_particles, force_scale, timesteps, p = Gen.get_args(trace)
    observations = zeros(Point2d, timesteps, n_particles)

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
    positions = zeros(Point2d, timesteps, n_particles)
    bounces = zeros(Vec2{Bounce}, timesteps, n_particles)

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

"""Run simple MCMC, updating only the force grid, with Gen's built-in proposal
distribution for mvnormal."""
function simple_mcmc(constraints :: Gen.ChoiceMap, args :: Tuple; computation = 100, cb=nothing)
    trace, _ = Gen.generate(force_model, args, constraints)
    if cb != nothing
        cb(trace, 0)
    end

    weights = Float64[]

    for step in 1:computation
        (trace, w) = metropolis_hastings(trace, select(:force_xs, :force_ys), check=true, observations=constraints)
        if cb != nothing
            cb(trace, step)
        end
        push!(weights, w)
    end

    trace, weights
end

"""Uses second differences to compute observed accelerations at every point on a grid. Returns a list of observed forces
at each location."""
function second_differences(observations :: PositionArray, bounces :: Array{Vec2{Bounce}, 2}, masses :: Vector{Float64};
        xres :: Int64, yres :: Int64, xrange :: Bounds, yrange :: Bounds, p :: ExtraParams) :: Grid{RunningStats{Vec2d}}

    obs_forces = Grid(reshape([RunningStats{Vec2d}() for i in 1:xres*yres], xres, yres), xrange, yrange)

    timesteps, n_particles = size(observations)
    velocities = zeros(Vec2d, size(observations)) # TODO guess + link

    for i in 1:n_particles
        for t in 2:timesteps
            pos = observations[t-1, i]
            vel = velocities[t-1, i]
            new_pos = observations[t, i]
            bounce = bounces[t, i]

            nvx, ax = invert_step_dim(pos[1], vel[1], new_pos[1], bounce[1], xrange, p)
            nvy, ay = invert_step_dim(pos[2], vel[2], new_pos[2], bounce[2], yrange, p)

            velocities[t, i] = Vec2d(nvx, nvy)
            acc = Vec2d(ax, ay)
            F = acc * masses[i]

            I = real_to_index(obs_forces, pos[1], pos[2])
            obs_forces.values[I] = update_stats(obs_forces.values[I], F)
        end
    end

    obs_forces
end

@testset "second differences" begin
    n = 12
    mm = [1.0 for i in 1:n]
    p = ExtraParams(1.0/24, 0.9)
    tt = Gen.simulate(force_model, (1.0, 10, n, 0.1, 48, p))

    gg = read_grid(tt)
    pp, bb = read_true_positions_bounces(tt)

    ff = second_differences(pp, bb, mm, xres=10, yres=10, xrange=(0.0, 1.0), yrange=(0.0, 1.0), p=p)

    @test sum([z.count for z in ff.values]) == length(pp[2:end, :])

    has_nonzero = false
    for I in CartesianIndices(ff.values)
        mean_, var_ = complete(ff.values[I])
        if !isnan(mean_)
            has_nonzero = true
            # we're using the true values so this should be the case
            @assert isapprox(mean_, gg.values[I], atol=0.00001)
        end
    end
    @test has_nonzero == true
end

function condition_wacky(
    noise :: Float64,
    length_scale :: Float64,
    known_locs :: Vector{Point2d},
    known_means :: Vector{Float64},
    known_vars :: Vector{Float64},
    unknown_locs :: Vector{Point2d},
    ) :: Tuple{Vector{Float64}, Matrix{Float64}}

    @assert !any(isnan.(known_locs))
    @assert !any(isnan.(known_means))
    @assert !any(isnan.(known_vars))

    # compute the means + covariance of the unknown region
    unknown_cond_means, unknown_cond_cov = compute_predictive(
        make_cov_vectorized(length_scale), noise, known_locs, known_means, unknown_locs
    )
    @assert !any(isnan.(unknown_cond_means)) "$unknown_cond_means"
    @assert !any(isnan.(unknown_cond_cov)) "$unknown_cond_cov"

    known_count = length(known_locs)
    unknown_count = length(unknown_locs)
    res = known_count + unknown_count

    # mean vector
    means = zeros(res)
    means[1:known_count] = known_means # observed
    means[known_count+1:end] = unknown_cond_means # inferred

    @assert !any(isnan.(means))

    # covariance matrix
    cov = zeros(res, res)
    # use measured variance for the observed part of the matrix.
    # this isn't technically inferred, but hey, we're in an MCMC proposal, anything goes.
    cov[1:known_count, 1:known_count] = diagm(known_vars)
    # and use inferred covariance for the rest of the matrix.
    cov[known_count+1:end, known_count+1:end] = unknown_cond_cov
    # ignore the matrix corners, the two chunks don't interact.

    @assert !any(isnan.(cov))
    @assert issymmetric(cov)

    # return the full matrix
    (means, cov)
end

@gen function second_diff_proposal(trace, observations, bounces)
    width, res, n_particles, force_scale, timesteps, p = Gen.get_args(trace)
    masses = [1.0 for i in 1:n_particles]

    # get inferred scale properties
    length_scale = trace[:length_scale]
    noise = trace[:noise]

    # figure out where we've seen particles accelerate
    diffgrid = second_differences(observations, bounces, masses,
        xres=res, yres=res, xrange=(0.0, width), yrange=(0.0, width), p=p)

    # get vectors of position / observed value
    Is = reshape(CartesianIndices(diffgrid.values), :)
    locs = [index_to_center(diffgrid, I[1], I[2]) for I in Is]
    F_means = [complete(diffgrid.values[I])[1] for I in Is]
    F_vars = [complete(diffgrid.values[I])[2] for I in Is]

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
    Fx_means_p, Fx_cov_p = condition_wacky(
        noise,
        length_scale,
        known_locs_p,
        [F_mean[1] for F_mean in known_F_means_p],
        [F_var[1] for F_var in known_F_vars_p],
        unknown_locs_p
        )
    # un-permute to line up w/ original indices
    Fx_means = Fx_means_p[inv_perm]
    Fx_cov = Fx_cov_p[inv_perm, inv_perm]
    # sample!
    force_xs ~ mvnormal(Fx_means, Fx_cov)

    @assert !any(isnan.(force_xs))

    # force y component
    Fy_means_p, Fy_cov_p = condition_wacky(
        noise,
        length_scale,
        known_locs_p,
        [F_mean[2] for F_mean in known_F_means_p],
        [F_var[2] for F_var in known_F_vars_p],
        unknown_locs_p
        )
    # un-permute to line up w/ original indices
    Fy_means = Fy_means_p[inv_perm]
    Fy_cov = Fy_cov_p[inv_perm, inv_perm]
    # sample!
    force_ys ~ mvnormal(Fy_means, Fy_cov)

    @assert !any(isnan.(force_ys))

end


@testset "basic second differences proposal" begin
    n = 3
    tt = Gen.simulate(force_model, (1.0, 4, n, 0.1, 5, ExtraParams(1.0/24, 0.9)))

    pp, bb = read_true_positions_bounces(tt)

    zz = Gen.simulate(second_diff_proposal, (tt, pp, bb))

    @test size(zz[:force_xs]) == (4*4,)
    @test size(zz[:force_ys]) == (4*4,)
end



#"""Run simple MCMC, updating only the force grid, with our custom second-differences
#proposal distribution."""
#function simple_mcmc_second_diff(constraints :: Gen.ChoiceMap, args :: Tuple; computation = 100, cb=nothing)
#    trace, _ = Gen.generate(force_model, args, constraints)
#    if cb != nothing
#        cb(trace, 0)
#    end
#
#    weights = Float64[]
#
#    for step in 1:computation
#        (trace, w) = metropolis_hastings(trace, select(:force_xs, :force_ys), check=true, observations=constraints)
#        if cb != nothing
#            cb(trace, step)
#        end
#        push!(weights, w)
#    end
#
#    trace, weights
#end
#
##

# ~~ static rendering ~~

n = 12
mm = [1.0 for i in 1:n]
tt = Gen.simulate(force_model, (1.0, 10, n, 0.1, 48, ExtraParams(1.0/24, 0.9)))
gg = read_grid(tt)
pp, bb = read_true_positions_bounces(tt)

s = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)
draw_particle_paths!(s, mm, pp, mass_scale=0.01)
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
println("done.")

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

simple_mcmc(cc, args, computation=1000, cb=debounce(s, function(trace, step)
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

# ~~ observed forces, no inference ~~

r = 10
w = 1.0
p = ExtraParams(1.0/24, 0.9)
n = 50
mm = [1.0 for i in 1:n]
tt = Gen.simulate(force_model, (w, r, n, 0.1, 48 * 3, p))
gg = read_grid(tt)
pp, bb = read_true_positions_bounces(tt)
oo = read_observations(tt)

s = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
t = Makie.Observable(1)

draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)
obs_grid = draw_grid!(s, map_grid(x -> Vec2d(0.0), Vec2d, gg), arrowcolor=:red, linecolor=:red)

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
function vizvar(s :: RunningStats{Vec2d}) :: RGBA
    mean_, var_ = complete(s)
    if isnan(mean_)
        RGBA(0.0, 0.0, 0.0, 0.0)
    else
        RGBA(1.0, 0.0, 0.0, clip(.5 * magnitude(mean_) / sqrt(magnitude(var_)), (0.0, 1.0)))
    end
end

second_diffs_aot = [
    second_differences(oo[1:t_, :], bb[1:t_, :], mm; xres=r, yres=r, xrange=(0.0, w), yrange=(0.0, w), p=p)
    for t_ in 1:size(oo, 1)
]
diff_arrows_aot = [
    [vizmean(s) / 24.0 for s in reshape(second_diff.values, :)]
    for second_diff in second_diffs_aot
]

updateloop(s) do
    t[] = (t[] % size(pp, 1)) + 1
    obs_grid[:directions][] = diff_arrows_aot[t[]]
end

##



##
#Profile.@profile zzz()
#Profile.print(maxdepth=3, sortedby=:overhead)
#Profile.clear()
#
##
