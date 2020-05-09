using Pkg
pkg"activate ."

using Test
using Makie
using AbstractPlotting
using GeometryBasics
using ColorSchemes

"""Inclusive bounds."""
Bounds = Tuple{Float64, Float64}

"""Locations."""
Vec2d = Vec2{Float64}
Point2d = Point2{Float64}


"""A grid of values.
Design:
a grid of N x M cells
[x][y]
x increases right, y increases up
So, printing will give strange results (y will be flipped)."""
struct Grid{T}
    values :: Array{T, 2}
    xrange :: Bounds
    yrange :: Bounds

    # redundant, just avoid recomputation
    width :: Float64
    height :: Float64
end

function Grid(values, xrange, yrange)
    left, right = xrange
    bottom, top = yrange

    @assert left < right
    @assert bottom < top

    Grid(values, xrange, yrange, Float64(right - left), Float64(top - bottom))
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

@test isapprox(map_range(1.0, 0.0, 4.0, 0.0, 1.0), 0.25)
@test isapprox(map_range(2.0, 1.0, 4.0, 0.0, 1.0), 0.25)

"""Clip a value to a range, and return whether it was clipped."""
@inline function clip(
    x :: Float64,
    range :: Bounds) :: Float64
    min, max = range
    if x < min
        min
    elseif x > max
        max
    else
        x
    end
end

@test clip(0.5, (-3.0, 10.0)) == 0.5
@test clip(-100.0, (-3.0, 10.0)) == -3.0
@test clip(100.0, (-3.0, 10.0)) == 10.0
@test clip(prevfloat(-3.0), (-3.0, 10.0)) == -3.0
@test clip(nextfloat(-3.0), (-3.0, 10.0)) == nextfloat(-3.0)
@test clip(nextfloat(10.0), (-3.0, 10.0)) == 10.0
@test clip(prevfloat(10.0), (-3.0, 10.0)) == prevfloat(10.0)

@inline function real_to_index(grid :: Grid, x :: Float64, y :: Float64) :: CartesianIndex
    sx, sy = size(grid.values)
    mapped_x = map_range(x, grid.xrange[1], grid.width, 1.0, Float64(sx))
    mapped_y = map_range(y, grid.yrange[1], grid.height, 1.0, Float64(sy))

    mapped_x = clip(mapped_x, (1.0, prevfloat(Float64(sx+1))))
    mapped_y = clip(mapped_y, (1.0, prevfloat(Float64(sy+1))))

    i = floor(Int32, mapped_x)
    j = floor(Int32, mapped_y)

    I = CartesianIndex((i, j))
    @assert I[1] >= 1 && I[1] <= size(grid.values, 1) &&
        I[2] >= 1 && I[2] <= size(grid.values, 2) "$x,$y ->\n   $mapped_x,$mapped_y ->\n   $i,$j"

    I
end
@inline function sample_grid(grid :: Grid, x :: Float64, y :: Float64)
    grid.values[real_to_index(grid, x, y)]
end

@inline function index_to_center(grid :: Grid, x :: Int, y :: Int) :: Point2
    sx, sy = size(grid.values)

    mapped_x = map_range(Float64(x) + 0.5, 1.0, Float64(sx), grid.xrange[1], grid.width)
    mapped_y = map_range(Float64(y) + 0.5, 1.0, Float64(sy), grid.yrange[1], grid.height)

    Point2((mapped_x, mapped_y))
end

# helper to print a grid's values how they'd be rendered
function disp(grid :: Grid)
    grid.values'[end:-1:1, :]
end


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

@testset "grids" begin
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

    @test sample_grid(g, 0.49999, 0.49999) == 11
    @test sample_grid(g, 0.50001, 0.50001) == 22
    @test sample_grid(g, 0.00001, 0.00001) == 11
    @test index_to_center(g, 1, 1) == Point2(.25, .25)
    @test index_to_center(g, 3, 1) == Point2(1.25, .25)

    @test disp(g) == [12 22 32;
                      11 21 31]

    g = map_grid(identity, Point2d, (0.0, 1.0), (0.0, 1.0), 13)
    @test all([g.values[I] == index_to_center(g, I[1], I[2]) for I in CartesianIndices(g.values)])
end

##

struct Mass
    mass :: Float64
    pos :: Point2d
    vel :: Vec2d
end

Base.zero(Mass) = Mass(0.0, Point2d(0.0), Vec2d(0.0))

struct ExtraParams
    dt :: Float64
    friction :: Float64
end

@enum Bounce begin
    NONE = 0
    LOW = 1
    HIGH = 2
end

"""Step a single dimension of a single particle.
Particles collide perfectly elastically with the boundary.
Returns (new_pos, new_vel, bounced_low, bounced_high).

Given pos, vel, bounds, new_pos, bounced_low, and bounced_high, you should be able to solve for acc and new_vel.
"""
@inline function step_dim(pos :: Float64, vel :: Float64, acc :: Float64, bounds :: Bounds, p :: ExtraParams) :: Tuple{Float64, Float64, Bounce}
    new_vel_oob = (vel + acc * p.dt) * p.friction
    new_pos_oob = pos + new_vel_oob * p.dt

    if new_pos_oob < bounds[1]
        new_pos = bounds[1] + (bounds[1] - new_pos_oob) # 2 b_1 - p_
        new_vel = -1. * new_vel_oob
        bounce = LOW
    elseif new_pos_oob > bounds[2]
        new_pos = bounds[2] - (new_pos_oob - bounds[2]) # 2 b_2 - p_
        new_vel = -1 * new_vel_oob
        bounce = HIGH
    else
        new_pos = new_pos_oob
        new_vel = new_vel_oob
        bounce = NONE
    end
    # multiple bounces aren't allowed
    @assert clip(new_pos, bounds) == new_pos "object moving too fast!"
    new_pos, new_vel, bounce
end

"""Solves for acc and new_vel given position, velocity, and whether the particle bounced."""
@inline function invert_step_dim(pos :: Float64, vel :: Float64, new_pos :: Float64,
    bounce :: Bounce, bounds :: Bounds, p :: ExtraParams) :: Tuple{Float64, Float64}
    if bounce == LOW
        new_pos_oob = -(new_pos - 2*bounds[1])
    elseif bounce == HIGH
        new_pos_oob = -(new_pos - 2*bounds[2])
    else
        new_pos_oob = new_pos
    end
    new_vel_oob = (new_pos_oob - pos) / p.dt

    if bounce == NONE
        new_vel = new_vel_oob
    else
        new_vel = -1.0 * new_vel_oob
    end

    acc = ((new_vel_oob / p.friction) - vel) / p.dt

    new_vel, acc
end

@testset "invert_step_dim_multi" begin
    bounds = (0.0, 1.0)
    p = ExtraParams(1.0/24, 0.9)
    bounces = [false, false, false]
    for pos in 0.0:0.2:1.0
        for vel in -10.0:1:10.0
            for acc in -1.0:0.2:1.0
                new_pos, new_vel, bounce = step_dim(pos, vel, acc, bounds, p)
                new_vel_, acc_ = invert_step_dim(pos, vel, new_pos, bounce, bounds, p)
                @assert isapprox(acc_, acc, atol=0.0001) &&
                    isapprox(new_vel_, new_vel, atol=0.0001) "$pos : $vel : $acc : $bounce, $new_vel_?=$new_vel, $acc_?=$acc"
                bounces[Int64(bounce) + 1] = true
            end
        end
    end
    @test all(bounces)
end

##
Int64(NONE) + 1
##

isapprox(1.3, 1.3, )

##

##


# simple euler integration, nearest-neighbor interpolation on grid

function step_mass(force :: Grid, mass :: Mass; dt :: Float64 = 1 / 24, friction :: Float64 = 0.9) :: Mass
    I = real_to_index(force, mass.pos[1], mass.pos[2])

    @assert I[1] >= 1 && I[1] <= size(force.values, 1) &&
        I[2] >= 1 && I[2] <= size(force.values, 2) "$(mass.pos) -> $I"
    F = force.values[I]

    vel_ = mass.vel + F * (dt / mass.mass)
    vel_ *= friction
    pos_ = mass.pos + vel * dt

    pos, vel = clipflip(pos_, vel_)


    Mass(mass.mass, Point2d(px, py), Vec2d(vx, vy))
end

g = map_grid(p -> Vec2d(1.0, -2.0), Vec2d, (-1.0, 1.0), (-1.0, 1.0), 9)

m = Mass(1.0, Vec2d(0.0, 0.0), Vec2d(0.0, 0.0))
dt = 1/24

mp = step_mass(g, m, dt=dt)

@test isapprox(mp.vel[1], 1 * dt * .9)
@test isapprox(mp.vel[2], -2 * dt * .9)
@test isapprox(mp.pos[1], (1 * dt) * dt * .9)
@test isapprox(mp.pos[2], (-2 * dt) * dt * .9)

struct SimulationRecord
    dt :: Float64
    friction :: Float64
    force :: Grid{Vec2d}
    initial_masses :: Vector{Mass}

    # [index, time]
    positions :: Array{Vec2d, 2}
end

##
function simulate_deterministic(force :: Grid{Vec2d}, masses :: Vector{Mass}; dt=Float64(1/24), timesteps=48, friction::Float64 = 0.9)
    current_masses = copy(masses)
    positions = zeros(Vec2d, length(masses), timesteps)
    snapshots[:, 1] = masses

    for t in 2:timesteps
        for i in 1:length(masses)
            snapshots[i, t] = step_mass(force, snapshots[i, t-1], dt=dt, friction=friction)
        end
    end

    SimulationRecord(dt, friction, force, snapshots)
end

##


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


function animate_record!(scene, record, t; scheme=ColorSchemes.rainbow, mass_scale = 0.05)
    n_masses, timesteps = size(record.snapshots)
    #sl = slider(range(1, timesteps, step=1), 1)
    #t = sl[end][:value]

    #draw_grid!(scene, record.force; linecolor=:gray)

    masses = lift(t -> [m.pos for m in record.snapshots[:, t]], t)
    weights = [m.mass * mass_scale for m in record.snapshots[:, 1]]
    colors = map(i -> get(scheme, (i-1)/n_masses), 1:n_masses)
    scatter!(scene, masses, markersize=weights, color=colors)

    #hbox(scene, sl), t
end

function draw_mass_paths!(scene, record; scheme=ColorSchemes.rainbow, mass_scale=0.05, override_color=())
    n_masses = size(record.snapshots, 1)
    n_times = size(record.snapshots, 2)

    colors = map(i -> get(scheme, (i-1)/n_masses), 1:n_masses)

    for i in 1:n_masses
        points = [record.snapshots[i, t].pos for t in 1:n_times]
        lines!(scene, points, color=colors[i], linewidth=2.0)
    end

    starts = [m.pos for m in record.snapshots[:, 1]]
    weights = [m.mass * mass_scale for m in record.snapshots[:, 1]]
    scatter!(scene, starts, color=colors, markersize=weights)

    scene
end
