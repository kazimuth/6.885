using Gen
using Makie
using GeometryBasics
using ColorSchemes

include("simulation.jl")
include("gp.jl")

##

points = reshape(map_grid(identity, Point2d, (0.0, 1.0), (0.0, 1.0), 5).values, :)

trace = Gen.simulate(grid_model, (points, 0.5, 0.1))

scatter(points, color=[trace.retval])

##

@gen function force_model(res :: Int64,
                          width :: Float64,
                          velocity_scale :: Float64,
                          n :: Int64,
                          timesteps :: Int64,
                          dt :: Float64)

    ## Build grid
    # Sample a length scale
    length_scale ~ gamma_bounded_below(1, 1, 0.01)
    # Sample a global noise level
    noise ~ gamma_bounded_below(1, 1, 0.01)
    # Make an empty grid
    forces = Grid(zeros(Point2d, res, res), (0.0, width), (0.0, width))
    # The indices of grid cells and their centroids
    Is = CartesianIndices(forces.values)
    centers = [index_to_center(forces, I[1], I[2]) for I in Is]
    # Sample forces
    force_xs ~ grid_model(centers, length_scale, noise)
    force_ys ~ grid_model(centers, length_scale, noise)
    # fill in grid
    for (I, fx, fy) in zip(Is, force_xs, force_ys)
        force.values[I] = Vec2d(fx, fy)
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
            Vec2d(
                {(:vx, i, 1)} ~ normal(0.0, width / 10.0),
                {(:vx, i, 1)} ~ normal(0.0, width / 10.0)
            )
        )
        for i in 1:n
    ]

    ## Run simulation
    trace = simulate_deterministic(force, masses, dt, timesteps)

    ## Compute observations
    for j in 2:timesteps
        for i in 1:n
            {(:px, i, j)} ~ normal(snapshots[i, n].pos[1], noise_factor)
            {(:py, i, j)} ~ normal(snapshots[i, n].pos[2], noise_factor)

            {(:vx, i, j)} ~ normal(snapshots[i, n].vel[1], noise_factor)
            {(:vy, i, j)} ~ normal(snapshots[i, n].vel[2], noise_factor)
        end
    end

    nothing
end

##

trace = Gen.simulate(force_model, (5, 1.0, 0.1, 3, 48, 1.0/24.0))
