################################
# collapsed chamfer likelihood #
################################

struct ProbabilisticChamfer <: Distribution{Matrix{Float64}} end
const probabilistic_chamfer = ProbabilisticChamfer()

@gen function corr_prior(p_outlier::Float64, m::Int, n::Int)
    corr_prior = fill((1-p_outlier)/m, m+1)
    corr_prior[1] = p_outlier
    @assert isapprox(sum(corr_prior), 1.)
    corr = Vector{Int}(undef, n)
    for i=1:n
        corr[i] = @trace(categorical(corr_prior), (:corr, i)) - 1 # 0=outlier
    end
    corr
end

function Gen.random(
        ::ProbabilisticChamfer, X::Matrix{Float64}, n::Int,
        p_outlier::Float64, noise::Float64, bounds::Geometry.Bounds)
    if size(X)[1] != 3
        error("Dimension mismatch")
    end
    m = size(X)[2]
    corr = corr_prior(p_outlier, m, n) # sample correspondences
    outliers = corr .== 0
    inliers = .!(outliers)
    num_inliers = sum(inliers)
    num_outliers = n - num_inliers
    Y = Matrix{Float64}(undef, 3, n)
    Y[:,inliers] = X[:,corr[inliers]] .+ (noise * randn(3, num_inliers)) # isotropic noise
    Y[1,outliers] .= bounds.xmin .+ ((bounds.xmax - bounds.xmin) * rand(num_outliers))
    Y[2,outliers] .= bounds.ymin .+ ((bounds.ymax - bounds.ymin) * rand(num_outliers))
    Y[3,outliers] .= bounds.zmin .+ ((bounds.zmax - bounds.zmin) * rand(num_outliers))
    Y
end

function logsumexp_over_cols(arr::Matrix{Float64})
    (nrows, ncols) = size(arr)
    max_arr = maximum(arr, dims=1) # maximum element of each column
    @assert size(max_arr) == (1,ncols)
    result = max_arr .+ log.(sum(exp.(arr .- max_arr), dims=1)) # result is a (1, nrows) matrix
    @assert size(result) == (1,ncols)
    result
end

function Gen.logpdf(
        ::ProbabilisticChamfer, Y::Matrix{Float64}, X::Matrix{Float64}, n::Int,
        p_outlier::Float64, noise::Float64, bounds::Geometry.Bounds)
    if size(X)[1] != 3 || size(Y)[1] != 3
        error("Dimension mismatch")
    end
    n = size(Y)[2]
    m = size(X)[2]

    D = Distances.pairwise(Distances.SqEuclidean(), X, Y, dims=2) # dx^2 + dy^2 + dz^2
    @assert size(D) == (m, n)

    # log_outlier = log ( p(corr_i = outlier) * P(Y[:.i] | outlier))
    log_outlier = (
        log(p_outlier)
        - log(bounds.xmax - bounds.xmin)
        - log(bounds.ymax - bounds.ymin)
        - log(bounds.zmax - bounds.zmin))

    # Linlier[i,j] = log (p(corr_i = j)  * p(Y[:,i] | X[:,j]))
    var = noise * noise
    Linliers = log(1 - p_outlier) - log(m) - 3*(0.5 * log(2*pi*var)) .- (D/(2*var))
    @assert size(Linliers) == (m, n)

    # Linlier_totals
    Linlier_totals = logsumexp_over_cols(Linliers) # result is (1,n) matrix
    @assert size(Linlier_totals) == (1,n)

    # log_denom[i] = log( p(Y[:,i] ), marginalizing over all possible sources including outlier
    max_arr = max.(log_outlier, Linlier_totals)
    log_denoms = max_arr .+ log.(exp.(log_outlier .- max_arr) .+ exp.(Linlier_totals .- max_arr))
    @assert size(log_denoms) == (1,n)

    sum(log_denoms)
end


function test_probabilistic_chamfer()
    p_outlier = 0.5
    bounds = Geometry.Bounds(-1, 1, -1, 1, -1, 1)
    noise = 0.5
    n = 1
    X = zeros((3, n))
    Y = zeros((3, n))
    Y[1] = -0.3
    Y[2] = 0.4
    Y[3] = 0.1
    expected = p_outlier * (1/2)^3 + (1-p_outlier) * (
        exp(logpdf(normal, Y[1], X[1], noise)) *
        exp(logpdf(normal, Y[2], X[2], noise)) *
        exp(logpdf(normal, Y[3], X[3], noise))
    )
    actual = exp(logpdf(probabilistic_chamfer, Y, X, n, p_outlier, noise, bounds))
    println("expected: $expected, actual: $actual")
    @assert isapprox(expected, actual)
end

#test_probabilistic_chamfer()
