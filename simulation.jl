using Test
using GeometryBasics



# design:
# a grid of N x M cells
# [x][y]
# x increases right, y increases up
# so, printing will give strange results (y will be flipped)

struct Grid{T}
    values :: Array{T, 2}
    left :: Float32
    right :: Float32
    bottom :: Float32
    top :: Float32

    width :: Float32
    height :: Float32
end

function Grid(values, xrange, yrange)
    left, right = xrange
    bottom, top = yrange

    @assert left < right
    @assert bottom < top

    Grid(values, Float32(left), Float32(right),
        Float32(bottom), Float32(top), Float32(right - left), Float32(top - bottom))
end
@inline function map_wrapping(
    x :: Float32,
    premin :: Float32,
    presize :: Float32,
    postmin :: Float32,
    postsize :: Float32)

    unit = (x - premin) / (presize)
    unit = mod(unit, 1.0)
    postmin + unit * postsize
end

@test isapprox(map_wrapping(1.0f0, 0.0f0, 4.0f0, 0.0f0, 1.0f0), 0.25f0)
@test isapprox(map_wrapping(-3.0f0, 0.0f0, 4.0f0, 0.0f0, 1.0f0), 0.25f0)

@test isapprox(map_wrapping(2.0f0, 1.0f0, 4.0f0, 0.0f0, 1.0f0), 0.25f0)
@test isapprox(map_wrapping(-2.0f0, 1.0f0, 4.0f0, 0.0f0, 1.0f0), 0.25f0)

@inline function real_to_index(grid :: Grid, x :: Float32, y :: Float32) :: CartesianIndex
    sx, sy = size(grid.values)
    mapped_x = map_wrapping(x, grid.left, grid.width, 1.0f0, Float32(sx))
    mapped_y = map_wrapping(y, grid.bottom, grid.height, 1.0f0, Float32(sy))

    i = floor(Int32, mapped_x)
    j = floor(Int32, mapped_y)

    CartesianIndex((i, j))
end
@inline function sample_grid(grid :: Grid, x :: Float32, y :: Float32)
    grid.values[real_to_index(grid, x, y)]
end

g = Grid(zeros(3,2), (0.0f0, 1.5f0), (0.0f0, 1.0f0))

g.values[1,1] = 11
g.values[1,2] = 12
g.values[2,1] = 21
g.values[2,2] = 22
g.values[3,1] = 31
g.values[3,2] = 32

@test sample_grid(g, 0.25f0, 0.25f0) == 11
@test sample_grid(g, 0.25f0, 0.75f0) == 12
@test sample_grid(g, 0.75f0, 0.25f0) == 21
@test sample_grid(g, 0.75f0, 0.75f0) == 22
@test sample_grid(g, 1.25f0, 0.25f0) == 31
@test sample_grid(g, 1.25f0, 0.75f0) == 32

@test sample_grid(g, -0.25f0, 0.25f0) == 31

@test sample_grid(g, 0.49999f0, 0.49999f0) == 11
@test sample_grid(g, 0.50001f0, 0.50001f0) == 22
@test sample_grid(g, 0.00001f0, 0.00001f0) == 11


@inline function index_to_center(grid :: Grid, x :: Int, y :: Int) :: Point2f0
    sx, sy = size(grid.values)

    mapped_x = map_wrapping(Float32(x) + 0.5f0, 1.0f0, Float32(sx), grid.left, grid.width)
    mapped_y = map_wrapping(Float32(y) + 0.5f0, 1.0f0, Float32(sy), grid.bottom, grid.height)

    Point2f0((mapped_x, mapped_y))
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

create a grid of type T, with ranges (xrange, yrange) and resolution res.

"""
function map_grid(f,
                  T :: Type{_T},
                  xrange :: Tuple{Float32, Float32},
                  yrange :: Tuple{Float32, Float32},
                  res :: Int) :: Grid{_T} where _T
    g = Grid(zeros(T, res, res), xrange, yrange)
    for I in CartesianIndices(g.values)
        v = index_to_center(g, I[1], I[2])
        g.values[I] = f(v)
    end
    g
end

g = map_grid(p -> p, GeometryBasics.Point2f0, (-1.0f0, 1.0f0), (-1.0f0, 1.0f0), 10)

for I in CartesianIndices(g.values)
    @test g.values[I] == index_to_center(g, I[1], I[2])
end

##

struct Mass
    mass :: Float32
    pos :: Point2f0
    vel :: Vec2f0
end

Base.zero(Mass) = Mass(0f0, Vec2f0(0f0), Vec2f0(0f0))

# simple euler integration, nearest-neighbor interpolation on grid

function step_mass(force :: Grid, mass :: Mass; dt :: Float32 = 1f0 / 24f0) :: Mass
    I = real_to_index(force, mass.pos[1], mass.pos[2])
    F = force.values[I]

    vel = mass.vel + F * (dt / mass.mass)
    pos_ = mass.pos + vel * dt
    pos = Point2f0(
        map_wrapping(pos_[1], force.left, force.width, force.left, force.width),
        map_wrapping(pos_[2], force.bottom, force.height, force.bottom, force.height)
    )

    Mass(mass.mass, pos, vel)
end

g = map_grid(p -> Vec2f0(1.0, -2.0), Vec2f0, (-1.0f0, 1.0f0), (-1.0f0, 1.0f0), 9)

m = Mass(1.0, Vec2f0(0.0, 0.0), Vec2f0(0.0, 0.0))
dt = 1f0/24f0

mp = step_mass(g, m, dt=dt)

@test isapprox(mp.vel[1], 1f0 * dt)
@test isapprox(mp.vel[2], -2f0 * dt)
@test isapprox(mp.pos[1], (1f0 * dt) * dt)
@test isapprox(mp.pos[2], (-2f0 * dt) * dt)

struct SimulationTrace
    dt :: Float32
    force :: Grid{Vec2f0}
    # [index, time]
    snapshots :: Array{Mass, 2}
end

function simulate_deterministic(force :: Grid, masses; dt=Float32(1/24), timesteps=48)
    snapshots = zeros(Mass, length(masses), timesteps)
    snapshots[:, 1] = masses

    for t in 2:timesteps
        for i in 1:length(masses)
            snapshots[i, t] = step_mass(force, snapshots[i, t-1], dt=dt)
        end
    end

    SimulationTrace(dt, force, snapshots)
end

##
using Makie
using AbstractPlotting

##

function draw_grid!(scene, grid :: Grid{Vec2f0}; scale=1/24, arrowsize=0.02, kwargs...)
    points = Point2f0[]
    directions = Vec2f0[]

    for I in CartesianIndices(grid.values)
        push!(points, index_to_center(grid, I[1], I[2]))
        push!(directions, grid.values[I] * scale)
    end

    arrows!(scene, points, directions; arrowsize=arrowsize, kwargs...)

    scene
end

##


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


g = map_grid(p -> 3 * Vec2f0(-p), Vec2f0, (-1.0f0, 1.0f0), (-1.0f0, 1.0f0), 9)

masses = [
    Mass(1.0f0, Vec2f0(0.25f0, 0.25f0), Vec2f0(-.3f0, .3f0)),
    Mass(1.0f0, Vec2f0(-0.25f0, 0.25f0), Vec2f0(-.3f0, -.3f0)),
]

trace_ = simulate_deterministic(g, masses)


s, t = animate_trace(trace_)

display(s)

##

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


##

function draw_mass_paths!(scene, trace; scheme=ColorSchemes.rainbow, mass_scale=0.05)
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

#
#scene = Scene(resolution=(1200, 1200), show_axis=false, show_grid=false)
#
#draw_grid!(scene, trace_.force)
#draw_mass_paths!(scene, trace_, )
#
#display(scene)
