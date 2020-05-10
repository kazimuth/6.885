"""Various helper mathematics."""

using Statistics, Test, Random

## ~~ running statistics ##

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

## ~~ permutations ~~

"""Create a permutation based on a boolean mask.
The vector will permute a vector so that values with the mask set are all before
values that don't.
Returns the length of the "before" segment.
"""
function split_permutation(mask :: Vector{Bool}) :: Tuple{Int32, Vector{Int32}}
    before = Int32[]
    after = Int32[]
    for i in 1:length(mask)
        if mask[i]
            push!(before, i)
        else
            push!(after, i)
        end
    end
    length(before), vcat(before, after)
end

"""Inverts a permutation vector."""
function invert_permutation(xs :: Vector{Int32}) :: Vector{Int32}
    slots = zeros(Int32, length(xs))
    for i in 1:length(xs)
        slots[xs[i]] = i
    end
    slots
end

@testset "permutations" begin
    @test [1 2; 3 4][[2, 1], :] == [3 4; 1 2]
    @test [1 2; 3 4][:, [2, 1]] == [2 1; 4 3]
    @test [1 2; 3 4][[2, 1], [2, 1]] == [4 3; 2 1]

    n, p = split_permutation([true, false, true, false])
    @test [1,2,3,4][p] == [1,3,2,4]
    @test n == 2

    n, p = split_permutation([false, false, false, true])
    @test [1,2,3,4][p] == [4,1,2,3]
    @test n == 1

    n, p = split_permutation([true, false, false, false])
    @test [1,2,3,4][p] == [1,2,3,4]
    @test n == 1

    n, p = split_permutation([true, true, true, true])
    @test [1,2,3,4][p] == [1,2,3,4]
    @test n == 4

    Random.seed!(0)
    for _ in 1:5
        p = shuffle(Int32.(1:6))
        pp = invert_permutation(p)
        @assert (1:6)[p][pp] == 1:6 "$p $pp"
    end
end

##
