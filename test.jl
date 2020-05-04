using Test
using GeometryBasics


##

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
##
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
##

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

##

@inline function index_to_center(grid :: Grid, x :: Int, y :: Int) :: Point2f0
    sx, sy = size(grid.values)

    mapped_x = map_wrapping(Float32(x) + 0.5f0, 1.0f0, Float32(sx), grid.left, grid.width)
    mapped_y = map_wrapping(Float32(y) + 0.5f0, 1.0f0, Float32(sy), grid.bottom, grid.height)

    Point2f0((mapped_x, mapped_y))
end

@test index_to_center(g, 1, 1) == Point2(.25, .25)
@test index_to_center(g, 3, 1) == Point2(1.25, .25)
##

# helper to print a grid's values how they'd be rendered
function disp(grid :: Grid)
    grid.values'[end:-1:1, :]
end

@test disp(g) == [12 22 32;
                  11 21 31]
##

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
using Makie
using AbstractPlotting




##

using Gen
##

#s1 = slider(LinRange(0.01, 1, 100), raw = true, camera = campixel!, start = 0.3)
s1 = slider(LinRange(0.01, 1, 100), raw = true, start = 0.3)
s2 = slider(LinRange(-2pi, 2pi, 100), raw = true)
data = lift(s2[end][:value]) do v
    map(LinRange(0, 2pi, 100)) do x
        4f0 .* Point2f0(sin(x) + (sin(x * v) .* 0.1), cos(x) + (cos(x * v) .* 0.1))
    end
end
p = scatter(data, markersize = s1[end][:value])

final = hbox(p, vbox(s1, s2), parent = Scene(resolution = (500, 500)))

display(final)

##



##

##

##

N = 10

cell_lefts = [float(x-1)/N for x=1:N]
cell_tops = cell_lefts

cell_centers = cell_lefts[]


##

N = 10

xs = Array(range(0.0, 1.0, length=10))
ys = Array(range(0.0, 1.0, length=10))

zs = [i for i=1:N, j=1:N]

itp = interpolate(zs, BSpline(Constant()))

##

# Do not execute beyond this point!

#RecordEvents(final, "output")

##
