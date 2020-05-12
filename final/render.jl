using Makie, Colors, ColorSchemes, AbstractPlotting
include("basics.jl")

"""Render a grid to a maie scene. Returns the resulting arrows object."""
function draw_grid!(scene, grid :: Grid{Vec2d}; scale=1/24, arrowsize=0.02, kwargs...)
    points = Point2d[]
    directions = Vec2d[]

    for I in CartesianIndices(grid.values)
        push!(points, index_to_center(grid, I[1], I[2]))
        push!(directions, grid.values[I] * scale)
    end

    arrows!(scene, points, directions; arrowsize=arrowsize, kwargs...)

    scene[end]
end

"""Given a scene, particle masses, particle positions, and a Makie.Observable timestep,
render masses as lines, with circles marking start points."""
function draw_particle_paths!(scene, masses :: Vector{Float64}, positions :: PositionArray;
    scheme=ColorSchemes.rainbow, mass_scale=0.05, colormod=identity)
    timesteps, n_particles = size(positions)

    colors = map(i -> colormod(get(scheme, (i-1)/n_particles)), 1:n_particles)

    lines = []

    for i in 1:n_particles
        lines!(scene, positions[:, i], color=colors[i], linewidth=2.0)
        push!(lines, scene[end])
    end

    starts = positions[1, :]
    weights = masses .* mass_scale

    scatter!(scene, starts, color=colors, markersize=weights)

    scatter = scene[end]

    lines, scatter
end

"""Given a scene, particle masses, particle positions, and a Makie.Observable timestep,
render masses as circles. Modifying the timestep will update the resulting plot.
Returns a scatterplot."""
function animate_particles!(scene, masses :: Vector{Float64}, positions :: PositionArray, t;
    scheme=ColorSchemes.rainbow, mass_scale = 0.05, colormod=identity)
    timesteps, n_particles = size(positions)

    positions_ = lift(t -> positions[t, :], t; typ=Vector{Vec2d})
    weights = masses .* mass_scale
    colors = map(i -> colormod(get(scheme, (i-1)/n_particles)), 1:n_particles)
    scatter!(scene, positions_, markersize=weights, color=colors)

    scene[end]
end

"""Calls a function `f` every `goal_dt` seconds.
Stops updating when the scene `s` has its window closed."""
function updateloop(f, s, goal_dt=1.0/24)
    display(s)
    pt = Base.time()
    while length(s.current_screens) > 0
        f()
        yield()
        ct = Base.time()
        dt = ct - pt
        pt = ct
        sleep(max(0.0, goal_dt - dt))
    end
    println("done.")
end

"""Wrap a function `f` so that it tries to only execute every `goal_dt` seconds.
Interrupts execution when the scene `s` is closed."""
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
            pt = Base.time()
            dt = pt - start
            f_t_est = .9*f_t_est + .1*dt
        end
    end
end
