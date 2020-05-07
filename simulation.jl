using Test
using GeometryBasics
using Makie
using AbstractPlotting
using ColorSchemes

# design:
# a grid of N x M cells
# [x][y]
# x increases right, y increases up
# so, printing will give strange results (y will be flipped)

struct Grid{T}
    values :: Array{T, 2}
    left :: Float64
    right :: Float64
    bottom :: Float64
    top :: Float64

    width :: Float64
    height :: Float64
end

function Grid(values, xrange, yrange)
    left, right = xrange
    bottom, top = yrange

    @assert left < right
    @assert bottom < top

    Grid(values, Float64(left), Float64(right),
        Float64(bottom), Float64(top), Float64(right - left), Float64(top - bottom))
end
@inline function map_range(
    x :: Float64,
    premin :: Float64,
    presize :: Float64,
    postmin :: Float64,
    postsize :: Float64)

    unit = (x - premin) / (presize)
    #unit = mod(unit, 1.0)
    postmin + unit * postsize
end
@inline function clip(
    x :: Float64,
    min :: Float64,
    max :: Float64)
    if x < min
        min
    elseif x > max
        max
    else
        x
    end
end

@test isapprox(map_range(1.0, 0.0, 4.0, 0.0, 1.0), 0.25)
#@test isapprox(map_range(-3.0, 0.0, 4.0, 0.0, 1.0), 0.25)

@test isapprox(map_range(2.0, 1.0, 4.0, 0.0, 1.0), 0.25)
#@test isapprox(map_range(-2.0, 1.0, 4.0, 0.0, 1.0), 0.25)

@inline function real_to_index(grid :: Grid, x :: Float64, y :: Float64) :: CartesianIndex
    sx, sy = size(grid.values)
    mapped_x = map_range(x, grid.left, grid.width, 1.0, Float64(sx))
    mapped_y = map_range(y, grid.bottom, grid.height, 1.0, Float64(sy))

    i = floor(Int32, mapped_x)
    j = floor(Int32, mapped_y)

    CartesianIndex((i, j))
end
@inline function sample_grid(grid :: Grid, x :: Float64, y :: Float64)
    grid.values[real_to_index(grid, x, y)]
end

g = Grid(zeros(3,2), (0.0, 1.5), (0.0, 1.0))

g.values[1,1] = 11
g.values[1,2] = 12
g.values[2,1] = 21
g.values[2,2] = 22
g.values[3,1] = 31
g.values[3,2] = 32

@test sample_grid(g, 0.25, 0.25) == 11
@test sample_grid(g, 0.25, 0.75) == 12
@test sample_grid(g, 0.75, 0.25) == 21
@test sample_grid(g, 0.75, 0.75) == 22
@test sample_grid(g, 1.25, 0.25) == 31
@test sample_grid(g, 1.25, 0.75) == 32

#@test sample_grid(g, -0.25, 0.25) == 31

@test sample_grid(g, 0.49999, 0.49999) == 11
@test sample_grid(g, 0.50001, 0.50001) == 22
@test sample_grid(g, 0.00001, 0.00001) == 11


@inline function index_to_center(grid :: Grid, x :: Int, y :: Int) :: Point2
    sx, sy = size(grid.values)

    mapped_x = map_range(Float64(x) + 0.5, 1.0, Float64(sx), grid.left, grid.width)
    mapped_y = map_range(Float64(y) + 0.5, 1.0, Float64(sy), grid.bottom, grid.height)

    Point2((mapped_x, mapped_y))
end

@test index_to_center(g, 1, 1) == Point2(.25, .25)
@test index_to_center(g, 3, 1) == Point2(1.25, .25)

# helper to print a grid's values how they'd be rendered
function disp(grid :: Grid)
    grid.values'[end:-1:1, :]
end

@test disp(g) == [12 22 32;
                  11 21 31]

"""
    map_grid(f, T, xrange, yrange, res)

Create a grid of type T, with ranges (xrange, yrange) and resolution res,
by mapping f over the slots.

"""
function map_grid(f,
                  T :: Type{_T},
                  xrange :: Tuple{Float64, Float64},
                  yrange :: Tuple{Float64, Float64},
                  res :: Int) :: Grid{_T} where _T
    g = Grid(zeros(T, res, res), xrange, yrange)
    for I in CartesianIndices(g.values)
        v = index_to_center(g, I[1], I[2])
        g.values[I] = f(v)
    end
    g
end

g = map_grid(p -> p, GeometryBasics.Point2, (-1.0, 1.0), (-1.0, 1.0), 10)

for I in CartesianIndices(g.values)
    @test g.values[I] == index_to_center(g, I[1], I[2])
end

Vec2d = Vec2{Float64}
Point2d = Point2{Float64}

struct Mass
    mass :: Float64
    pos :: Point2d
    vel :: Vec2d
end

Base.zero(Mass) = Mass(0.0, Vec2d(0.0), Vec2d(0.0))

# simple euler integration, nearest-neighbor interpolation on grid

function step_mass(force :: Grid, mass :: Mass; dt :: Float64 = 1 / 24) :: Mass
    I = real_to_index(force, mass.pos[1], mass.pos[2])
    F = force.values[I]

    vel = mass.vel + F * (dt / mass.mass)
    pos_ = mass.pos + vel * dt
    pos = Point2(
        clip(pos_[1], force.left, force.right),
        clip(pos_[2], force.bottom, force.top),
    )

    Mass(mass.mass, pos, vel)
end

g = map_grid(p -> Vec2d(1.0, -2.0), Vec2d, (-1.0, 1.0), (-1.0, 1.0), 9)

m = Mass(1.0, Vec2d(0.0, 0.0), Vec2d(0.0, 0.0))
dt = 1/24

mp = step_mass(g, m, dt=dt)

@test isapprox(mp.vel[1], 1 * dt)
@test isapprox(mp.vel[2], -2 * dt)
@test isapprox(mp.pos[1], (1 * dt) * dt)
@test isapprox(mp.pos[2], (-2 * dt) * dt)

struct SimulationTrace
    dt :: Float64
    force :: Grid{Vec2d}
    # [index, time]
    snapshots :: Array{Mass, 2}
end

function simulate_deterministic(force :: Grid{Vec2d}, masses :: Vector{Mass}; dt=Float64(1/24), timesteps=48)
    snapshots = zeros(Mass, length(masses), timesteps)
    snapshots[:, 1] = masses

    for t in 2:timesteps
        for i in 1:length(masses)
            snapshots[i, t] = step_mass(force, snapshots[i, t-1], dt=dt)
        end
    end

    SimulationTrace(dt, force, snapshots)
end


function draw_grid!(scene, grid :: Grid{Vec2d}; scale=1/24, arrowsize=0.02, kwargs...)
    points = Point2d[]
    directions = Vec2d[]

    for I in CartesianIndices(grid.values)
        push!(points, index_to_center(grid, I[1], I[2]))
        push!(directions, grid.values[I] * scale)
    end

    arrows!(scene, points, directions; arrowsize=arrowsize, kwargs...)

    scene
end



function animate_trace(trace; mass_scale = 0.05)
    sl = slider(range(1, size(trace.snapshots, 2), step=1), 1)
    t = sl[end][:value]

    scene = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
    draw_grid!(scene, trace.force; linecolor=:gray)

    masses = lift(t -> [m.pos for m in trace.snapshots[:, t]], t)
    weights = [m.mass * mass_scale for m in trace.snapshots[:, 1]]
    scatter!(scene, masses, markersize=weights)

    hbox(scene, sl), t
end


function animate_trace(trace; mass_scale = 0.05)
    sl = slider(range(1, size(trace.snapshots, 2), step=1), 1)
    t = sl[end][:value]

    scene = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
    draw_grid!(scene, trace.force; linecolor=:gray)

    masses = lift(t -> [m.pos for m in trace.snapshots[:, t]], t)
    weights = [m.mass * mass_scale for m in trace.snapshots[:, 1]]
    scatter!(scene, masses, markersize=weights)

    hbox(scene, sl), t
end

function draw_mass_paths!(scene, trace; scheme=ColorSchemes.rainbow, mass_scale=0.05, override_color=())
    n_masses = size(trace.snapshots, 1)
    n_times = size(trace.snapshots, 2)

    colors = map(i -> get(scheme, (i-1)/n_masses), 1:n_masses)

    for i in 1:n_masses
        points = [trace.snapshots[i, j].pos for j in 1:n_times]
        lines!(scene, points, color=colors[i], linewidth=2.0)
    end

    starts = [m.pos for m in trace.snapshots[:, 1]]
    weights = [m.mass * mass_scale for m in trace.snapshots[:, 1]]
    scatter!(scene, starts, color=colors, markersize=weights)

    scene
end


"""
g = map_grid(p -> 3 * Vec2d(-p), Vec2d, (-1.0, 1.0), (-1.0, 1.0), 9)

masses = [
    Mass(1.0, Vec2(0.25, 0.25), Vec2(-.3, .3)),
    Mass(1.0, Vec2(-0.25, 0.25), Vec2(-.3, -.3)),
]

trace_ = simulate_deterministic(g, masses)

scene = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)

draw_grid!(scene, trace_.force)
draw_mass_paths!(scene, trace_, )

display(scene)
"""
