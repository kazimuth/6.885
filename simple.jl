using Pkg
pkg"activate ."

using Gen
using Makie
using AbstractPlotting
using GeometryBasics
using Colors
using ColorSchemes

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
    observe_noise = width / 100.0

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
        cb(trace)
    end

    weights = Float64[]

    for i in 1:computation
        (trace, w) = metropolis_hastings(trace, select(:force_xs, :force_ys), check=true, observations=constraints)
        if cb != nothing
            cb(trace)
        end
        push!(weights, w)
    end

    trace, weights
end


##

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

display(s)
while length(s.current_screens) > 0
    sleep(1.0/24)
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

display(s)

simple_mcmc(cc, args, computation=1000, cb=function(trace)
    gg = read_grid(trace)
    guess_grid[:directions] = reshape(gg.values./24, :)

    oo, _ = read_true_positions_bounces(trace)
    for i in 1:size(oo, 2)
        ll[i][:positions] = oo[:, i]
    end
    ss[:positions] = oo[1, :]
    sleep(0.001)
end)

##

ll[1][:positions]



##

typeof(ll[1])



##

function denoise_curves(record :: SimulationRecord) :: SimulationRecord
    # TODO
    record
end

function second_differences(record :: SimulationRecord)
    n, timesteps = size(record.snapshots)

    # since we know the exact euler-integral form of the input simulation,
    # we can invert it algebraically (assuming no noise)

    # known: all ps, v[1]
    # v_ = (v[i-1] + dt*F[p[i-1]]/mass) * friction
    # p_ = p[i-1] + v[i] * dt
    # p[i], v[i] = clipflip(p_, v_)

    # to solve:
    # p_, v_ = inv_clipflip(p[i], v[i])
    # v_ = (p[i] - p[i-1]) / dt
    # p_ = (v[i]/friction - v[i-1]) * (mass/dt)


    # instead of directly setting forces, we add to the observations at
    # their locations.

    dt = record.dt
    friction = record.friction
    snapshots = record.snapshots

    # note: difference from previous timestep;
    # first timestep is 0s
    vels = zeros(Vec2d, n, timesteps)
    observed_forces = Grid([[] for v in record.forces.values],
        record.forces.xrange,
        record.forces.yrange)
    for i in 2:timesteps
        for j in 1:n
            vels[j, i] = (snapshots[j, i].pos - snapshots[j, i-1].pos) / dt

            F = vels[j, i]/friction - vels[j, i-1]
        end
    end

    # again, first timestep is 0s
    accs = zeros(Vec2d, n, timesteps)
    for step in 1:timesteps-1
        for i in 1:n
            vels[i, step+1] = record.snapshots[i, step+1].pos -
                record.snapshots[i, step]
        end
    end

    # TODO convert to running?


end


real_trace = Gen.simulate(force_model, (10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9))
@time trace, _ = do_inference((10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9), real_trace.retval.snapshots, computation=10)

second_differences(trace)
##

StaticChoiceMap


##

real_trace = Gen.simulate(force_model, (10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9))
@time trace, probs = do_inference((10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9), real_trace.retval.snapshots, computation=10)



##

##

real_trace[:noise], trace[:noise]


##

plot(probs)

##

probs

##

args = (10, 1.0, 0.1, 3, 24, 1.0/24.0, 0.9)
real_trace = Gen.simulate(force_model, args)
scene = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false, limits=AbstractPlotting.Rect(0.0, 0.0, 1.0, 1.0))

draw_grid!(scene, real_trace.retval.force, arrowcolor=:lightgray)
draw_grid!(scene, real_trace.retval.force, arrowcolor=:lightblue)
t = Makie.Observable(1)
animate_record!(scene, real_trace.retval, t, mass_scale=0.02)
function update_t()
    current = floor(Int64, Base.time() * 24 % args[5]) + 1
    if current != t[]
        t[] = current
    end
end

display(scene)

#plot(probs)

##


trace = Gen.simulate(force_model, args)
@time trace, probs = do_inference(args, trace.retval.snapshots, computation=10000, length_scale=real_trace[:length_scale], noise=real_trace[:noise], cb=function(trace)
    r = trace_to_record(trace)

    directions = []
    for I in CartesianIndices(r.force.values)
        push!(directions, r.force.values[I] / 24)
    end

    scene[2][:directions] = directions

    update_t()
    sleep(0.001)
end)

##

cmap = make_choicemap(real_trace.retval.snapshots, length_scale=real_trace[:length_scale], noise=real_trace[:noise])
trace, _ = Gen.generate(force_model, args, cmap)

v = Gen.map_optimize(trace, select(:force_xs, :force_ys))

scene = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false, limits=AbstractPlotting.Rect(0.0, 0.0, 1.0, 1.0))
draw_grid!(scene, real_trace.retval.force, arrowcolor=:lightgray)
draw_grid!(scene, v.retval.force, arrowcolor=:lightblue)
display(scene)

##

lines

##
trace = Gen.simulate(force_model, (10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9))
@time trace, probs = do_inference((10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9), trace.retval.snapshots, computation=100)

##

# TODO: custom proposal for forces given known velocities
# TODO: try moving initial point sampling before grid construction? + using a combinator?
# TODO: sample dist using 2nd-differences +
#           + prediction of nonexistent points from orig. code!
