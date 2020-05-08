using Gen
using GeometryBasics
using LinearAlgebra

##

Vec2d = Vec2{Float64}
Point2d = Point2{Float64}

##

@dist gamma_bounded_below(shape, scale, bound) = gamma(shape, scale) + bound

##

"""Compute covariance matrix by evaluating function on each pair of inputs."""
function compute_cov_matrix(f, noise :: Float64, xs :: Vector{Float64})
    n = length(xs)
    cov_matrix = Matrix{Float64}(undef, n, n)
    for i=1:n
        for j=1:n
            cov_matrix[i, j] = f(xs[i], xs[j])
        end
        cov_matrix[i, i] += noise
    end
    return cov_matrix
end

"""Compute covariance function by recursively computing covariance matrices."""
function compute_cov_matrix_vectorized(f_vec, noise :: Float64, xs::Vector{Point2d})
    n = length(xs)
    f_vec(xs) + Matrix(noise * LinearAlgebra.I, n, n)
end

function magnitude(p :: Point2d)
    sqrt(p[1] * p[1] + p[2] * p[2])
end

"""Sample a GP on a 2d grid."""
@gen function grid_model(xs::Array{Point2d, 2}, length_scale :: Float64, noise :: Float64) :: Array{Float64, 2}

    # define vectorized covariance function
    f_vec = function(xs::Vector{Point2d})
        xs_ = reshape(xs, :, 1)
        xs__ = reshape(xs, 1, :)
        diff = magnitude.(xs_ .- xs__)
        exp.(-0.5 .* diff .* diff ./ length_scale)
    end

    xs_ = reshape(xs, :)

    # Compute the covariance between every pair (xs[i], xs[j])
    cov_matrix = compute_cov_matrix_vectorized(f_vec, noise, xs_)

    # Sample from the GP using a multivariate normal distribution with
    # the kernel-derived covariance matrix.
    ys_ ~ mvnormal(zeros(length(xs)), cov_matrix)

    ys = reshape(ys_, size(xs)...)

    return ys
end;

##


##
