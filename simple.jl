using Gen
using Makie
using GeometryBasics
using ColorSchemes

include("simulation.jl")
include("gp.jl")

##
#
#points = reshape(map_grid(identity, Point2d, (0.0, 1.0), (0.0, 1.0), 5).values, :)
#trace = Gen.simulate(grid_model, (points, 0.5, 0.1))
#scatter(points, color=trace.retval)


@gen function force_model(res :: Int64,
                          width :: Float64,
                          force_scale :: Float64,
                          n :: Int64,
                          timesteps :: Int64,
                          dt :: Float64,
                          friction :: Float64)

    ## Build grid
    # Sample a length scale
    length_scale ~ gamma_bounded_below(1, width/10, width * 0.01)
    # Sample a global noise level
    noise ~ gamma_bounded_below(1, force_scale, 0.01)
    # Make an empty grid
    forces = Grid(zeros(Vec2d, res, res), (0.0, width), (0.0, width))
    # The indices of grid cells and their centroids
    Is = CartesianIndices(forces.values)
    centers = [index_to_center(forces, I[1], I[2]) for I in Is] :: Array{Point2d, 2}

    # Sample forces
    (force_xs ~ grid_model(centers, length_scale, noise)) :: Array{Float64, 2}
    (force_ys ~ grid_model(centers, length_scale, noise)) :: Array{Float64, 2}
    # fill in grid
    for I in Is
        forces.values[I] = Vec2d(force_xs[I], force_ys[I])
    end

    ## Noise in velocity / position observations
    noise_factor = width / 100.0

    ## Build starting masses
    masses = [
        Mass(
            # mass
            1.0,
            # position
            Point2d(
                {(:px, i, 1)} ~ uniform(0.0, width),
                {(:py, i, 1)} ~ uniform(0.0, width)
            ),
            # velocity
            Vec2d(0.0, 0.0)
        )
        for i in 1:n
    ]

    ## Run simulation
    record = simulate_deterministic(forces, masses, dt=dt, timesteps=timesteps, friction=friction)

    ## Compute observations (noisy)
    for step in 2:timesteps
        for i in 1:n
            {(:px, i, step)} ~ normal(record.snapshots[i, step].pos[1], noise_factor)
            {(:py, i, step)} ~ normal(record.snapshots[i, step].pos[2], noise_factor)
            {(:vx, i, step)} ~ normal(record.snapshots[i, step].vel[1], noise_factor)
            {(:vy, i, step)} ~ normal(record.snapshots[i, step].vel[2], noise_factor)
        end
    end

    ## Return record (not noisy!)
    record
end


"""I could probably have named these data structures better..."""
function trace_to_record(trace :: Gen.DynamicDSLTrace) :: SimulationRecord
    res = trace.args[1]
    width = trace.args[2]
    n = trace.args[4]
    timesteps = trace.args[5]
    dt = trace.args[6]

    force_xs = trace[:force_xs] :: Array{Float64, 2}
    force_ys = trace[:force_ys] :: Array{Float64, 2}

    forces = Grid(zeros(Vec2d, res, res), (0.0, width), (0.0, width))
    for i in 1:res
        for j in 1:res
            forces.values[i, j] = Vec2d(force_xs[i, j], force_ys[i, j])
        end
    end
    snapshots = zeros(Mass, n, timesteps)
    for i in 1:n
        for step in 1:timesteps
            snapshots[i, step] = Mass(
                1.0,
                Point2d(
                    trace[(:px, i, step)],
                    trace[(:py, i, step)]
                ),
                Vec2d(0.0, 0.0)
            )
        end
    end

    SimulationRecord(dt, forces, snapshots)
end

##

"""
trace = Gen.simulate(force_model, (10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9))

r = trace_to_record(trace)


##

scene = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)

draw_grid!(scene, r.force)
#draw_mass_paths!(scene, r, mass_scale=0.01)
draw_mass_paths!(scene, trace.retval, mass_scale=0.01)

display(scene)

##

scene = Scene(resolution=(1600, 1600), show_axis=false, show_grid=false)

t = Makie.Observable(1)
animate_record!(scene, trace.retval, t, mass_scale=0.02)
display(scene)
while true
    sleep(1.0/24)
    t[] = (t[] % size(r.snapshots, 2)) + 1
end


##

"""
"""
trace = Gen.simulate(force_model, (10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9))
r = trace_to_record(trace)

scene = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
draw_grid!(scene, r.force)
display(scene)


for i in 1:50
    trace = Gen.simulate(force_model, (10, 1.0, 0.1, 20, 48, 1.0/24.0))
    r = trace_to_record(trace)

    directions = []
    for I in CartesianIndices(r.force.values)
        push!(directions, r.force.values[I] / 24)
    end

    scene[end][:directions] = directions
    sleep(1/5)
end
"""
nothing


##

function make_choicemap(snapshots :: Array{Mass, 2})
    n_masses, timesteps = size(snapshots)
    #@assert args[4] == n_masses
    #@assert args[5] == timesteps

    cmap = Gen.choicemap()
    for step in 1:timesteps
        for i in 1:n_masses
            cmap[(:px, i, step)] = snapshots[i, step].pos[1]
            cmap[(:py, i, step)] = snapshots[i, step].pos[2]
        end
    end
    cmap
end

function do_inference(args :: Tuple, snapshots :: Array{Mass, 2}; computation = 100, cb=nothing)
    cmap = make_choicemap(snapshots)
    trace, _ = Gen.generate(force_model, args, cmap)

    weights = Float64[]

    for i in 1:computation
        (trace, _) = metropolis_hastings(trace, select(:noise), check=true)
        (trace, _) = metropolis_hastings(trace, select(:length_scale), check=true)
        (trace, _) = metropolis_hastings(trace, select(:force_xs), check=true)
        (trace, w) = metropolis_hastings(trace, select(:force_ys), check=true)
        if cb != nothing && i % 1 == 0
            cb(trace)
        end
        push!(weights, w)
    end

    trace, weights
end

##

real_trace = Gen.simulate(force_model, (10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9))
@time trace, probs = do_inference((10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9), real_trace.retval.snapshots, computation=10)



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
@time trace, probs = do_inference(args, trace.retval.snapshots, computation=10000, cb=function(trace)
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

lines

##
trace = Gen.simulate(force_model, (10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9))
@time trace, probs = do_inference((10, 1.0, 0.1, 20, 48, 1.0/24.0, 0.9), trace.retval.snapshots, computation=100)

##
