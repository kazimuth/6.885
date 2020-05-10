using Pkg
pkg"activate ."

using Gen
using Makie
using AbstractPlotting
using GeometryBasics
using Colors
using ColorSchemes
using Statistics

include("simulation.jl")
include("gp.jl")

##

function grid_centers(xrange :: Bounds, yrange :: Bounds, xres :: Int64, yres :: Int64) :: Vector{Point2d}
    reshape(map_grid(identity, Point2d, xrange, yrange, xres, yres).values, :)
end

function grids_to_vec_grid(
        centers :: Vector{Point2d},
        xs :: Vector{Float64}, ys :: Vector{Float64},
        xrange :: Bounds, yrange :: Bounds,
        xres :: Int64, yres :: Int64,
        ) :: Grid{Vec2d}
    values = [Vec2d(xs[i], ys[i]) for i in 1:length(xs)]
    result = Grid(reshape(values, xres, yres), xrange, yrange)

    for i in 1:length(xs)
        I = real_to_index(result, centers[i][1], centers[i][2])
        @assert result.values[I][1] == xs[i]
        @assert result.values[I][2] == ys[i]
    end

    result
end


function run_particle(initial_pos :: Point2d, initial_vel :: Vec2d, mass :: Float64;
        forces :: Grid, timesteps :: Int64, p :: ExtraParams) :: Tuple{Vector{Point2d}, Vector{Vec2{Bounce}}}
    positions = [initial_pos]
    bounces = [Vec2(NONE, NONE)]

    pos = initial_pos
    vel = initial_vel

    for t in 2:timesteps
        pos, vel, bounce = step_particle(pos, vel, mass, forces=forces, p=p)
        push!(positions, pos)
        push!(bounces, bounce)
    end

    (positions, bounces)
end

@testset "run_particle" begin
    ps, bs = run_particle(Point2d(0.5, 0.5), Vec2d(0.0, 0.0), 1.0,
        forces=map_grid(x -> Vec2d(0.1, 0.1), Vec2d, (0., 1.), (0.1, 1.), 10, 10),
        timesteps=48,
        p=ExtraParams(1.0/24, 0.9))
    @test size(ps, 1) == 48
end


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

    true_positions, true_bounces = run_particle(Point2d(initial_x, initial_y), Vec2d(0.0, 0.0), 1.0;
        forces=forces, timesteps=timesteps, p=p)

    noises = [observe_noise for t in 1:timesteps]
    observations ~ Gen.Map(observe_position)(true_positions, noises)

    return (true_positions, true_bounces)
end

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

"""I could probably have named these data structures better..."""
function read_grid(trace) :: Grid{Vec2d}
    width, res, n_particles, force_scale, timesteps, p = Gen.get_args(trace)
    bounds = (0.0, width)
    centers = grid_centers(bounds, bounds, res, res)

    grids_to_vec_grid(centers, trace[:force_xs], trace[:force_ys], bounds, bounds, res, res)
end

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


function add_observations!(constraints :: Gen.ChoiceMap, observations :: PositionArray)
    timesteps, n_particles = size(observations)
    for i in 1:n_particles
        for t in 1:timesteps
            constraints[:mass_paths => i => :observations => t => :x] = observations[t, i][1]
            constraints[:mass_paths => i => :observations => t => :y] = observations[t, i][2]
        end
    end
end

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

"""Uses Welford's algorithm to compute the mean and variance of a sequence in
constant memory, in a numerically stable manner.

- mean accumulates the mean of the entire dataset
- M2 aggregates the squared distance from the mean
- count aggregates the number of samples seen so far

See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
"""
struct RunningStats{T}
    count :: Int64
    mean :: T
    M2 :: T
end
function RunningStats{T}() where T
    RunningStats(0, T(0), T(0))
end

"""Update the statistics."""
function update(s :: RunningStats{T}, v :: T) :: RunningStats{T} where {T}
    (count, mean, M2) = s.count, s.mean, s.M2
    count += 1
    # avoid errors w/ vectors, this is supposed to be elementwise anyway
    delta = v .- mean
    mean += delta ./ count
    delta2 = v .- mean
    M2 += delta .* delta2
    return RunningStats(count, mean, M2)
end

"""Compute the mean and variance."""
function complete(s :: RunningStats{T}) :: Tuple{T, T} where {T}
    (count, mean, M2) = s.count, s.mean, s.M2
    if count < 2
        (T(NaN), T(NaN))
    else
       (mean, M2 / count)
    end
end

@testset "running stats" begin
    s = RunningStats{Float64}()
    a,b = complete(s)
    @test isnan(a)
    @test isnan(b)

    s = RunningStats{Float64}()
    vs = 0.0:.039:12.3
    for v in vs
        s = update(s, v)
    end
    mean_, var_ = complete(s)
    @test isapprox(mean_, mean(vs), atol=0.1)
    @test isapprox(var_, var(vs), atol=0.1)
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
            obs_forces.values[I] = update(obs_forces.values[I], F)
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


function updateloop(f, s, goal_dt=1.0/24)
    display(s)
    pt = Base.time()
    while length(s.current_screens) > 0
        f()
        ct = Base.time()
        dt = ct - pt
        pt = ct
        sleep(max(0.0, goal_dt - dt))
    end
    println("done.")
end


function debounce(s, f, goal_dt=1.0/24)
    pt = Base.time()
    f_t_est = goal_dt
    return function(args...)
        if length(s.current_screens) == 0
            throw(InterruptException())
        end
        current = Base.time()
        if current + f_t_est >= pt + goal_dt
            start = Base.time()
            f(args...)
            yield() # needed to let Makie draw stuff
            dt = Base.time() - start
            f_t_est = .9*f_t_est + .1*dt
        end
    end
end

##

# ~~ static rendering ~~

n = 12
mm = [1.0 for i in 1:n]
tt = Gen.simulate(force_model, (1.0, 10, n, 0.1, 48, ExtraParams(1.0/24, 0.9)))
gg = read_grid(tt)
pp, bb = read_true_positions_bounces(tt)

s = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)
draw_mass_paths!(s, mm, pp, mass_scale=0.01)
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
animate_record!(s, mm, pp, t, mass_scale=0.02, colormod=c -> RGBA(c, .5))
# false positions, high alpha
animate_record!(s, mm, oo, t, mass_scale=0.02)

updateloop(s) do
    t[] = (t[] % size(pp, 1)) + 1
end
println("done.")

## ~~ mcmc w/ true paths, all other things fixed ~~

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

pp, _ = read_true_positions_bounces(tt)
add_observations!(cc, pp) # note: we use noiseless recordings here

mm = [1.0 for i in 1:n]

s = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)

guess_grid = draw_grid!(s, gg, arrowcolor=:lightblue, linecolor=:lightblue)
draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)

draw_mass_paths!(s, mm, pp, mass_scale=0.01, colormod=c -> RGBA(c, .5))
ll, ss = draw_mass_paths!(s, mm, pp .* 0.0, mass_scale=0.01)

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

# ~~ observed forces ~~
r = 10
w = 1.0
p = ExtraParams(1.0/24, 0.9)
n = 50
mm = [1.0 for i in 1:n]
tt = Gen.simulate(force_model, (w, r, n, 0.1, 48, p))
gg = read_grid(tt)
pp, bb = read_true_positions_bounces(tt)
oo = read_observations(tt)

s = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
t = Makie.Observable(1)

draw_grid!(s, gg, arrowcolor=:lightgray, linecolor=:lightgray)
obs_grid = draw_grid!(s, map_grid(x -> Vec2d(0.0), Vec2d, gg), arrowcolor=:red, linecolor=:red)

# true positions, low alpha
animate_record!(s, mm, pp, t, mass_scale=0.02, colormod=c -> RGBA(c, .5))
# false positions, high alpha
animate_record!(s, mm, oo, t, mass_scale=0.02)

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
