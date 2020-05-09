using Pkg
pkg"activate ."

using Gen
using GeometryBasics
using LinearAlgebra

Vec2d = Vec2{Float64}
Point2d = Point2{Float64}

@dist gamma_bounded_below(shape, scale, bound) = gamma(shape, scale) + bound

"""Compute covariance function by recursively computing covariance matrices."""
function compute_cov_matrix_vectorized(f_vec, noise :: Float64, xs::Vector{Point2d})
    n = length(xs)
    f_vec(xs) + Matrix(noise * LinearAlgebra.I, n, n)
end

function magnitude(p :: Point2d)
    sqrt(p[1] * p[1] + p[2] * p[2])
end

function cov_vectorized(length_scale :: Float64, xs :: Vector{Point2d})
    xs_ = reshape(xs, :, 1)
    xs__ = reshape(xs, 1, :)
    diff = magnitude.(xs_ .- xs__)
    exp.(-0.5 .* diff .* diff ./ length_scale)
end

"""Sample a GP on a 2d grid."""
@gen function grid_model(xs::Array{Point2d, 2}, length_scale :: Float64, noise :: Float64) :: Array{Float64, 2}
    xs_ = reshape(xs, :)

    # Compute the covariance between every pair (xs[i], xs[j])
    cov_matrix = compute_cov_matrix_vectorized(
        xs -> cov_vectorized(length_scale, xs),
        noise,
        xs_
    )

    # Sample from the GP using a multivariate normal distribution with
    # the kernel-derived covariance matrix.
    ys_ ~ mvnormal(zeros(length(xs)), cov_matrix)

    ys = reshape(ys_, size(xs)...)

    return ys
end;


"""
Computes the conditional mean and covariance of a Gaussian process with prior mean zero
and prior covariance function `f_vec`, conditioned on noisy observations
`Normal(f(xs), noise * I) = ys`, evaluated at the points `new_xs`.
"""
function compute_predictive(f_vec, noise::Float64,
                            xs::Vector{Point2d}, ys::Vector{Float64},
                            new_xs::Vector{Point2d})
    n_prev = length(xs)
    n_new = length(new_xs)
    means = zeros(n_prev + n_new)
    #cov_matrix = compute_cov_matrix(covariance_fn, noise, vcat(xs, new_xs))
    cov_matrix = compute_cov_matrix_vectorized(f_vec, noise, vcat(xs, new_xs))
    cov_matrix_11 = cov_matrix[1:n_prev, 1:n_prev]
    cov_matrix_22 = cov_matrix[n_prev+1:n_prev+n_new, n_prev+1:n_prev+n_new]
    cov_matrix_12 = cov_matrix[1:n_prev, n_prev+1:n_prev+n_new]
    cov_matrix_21 = cov_matrix[n_prev+1:n_prev+n_new, 1:n_prev]
    @assert cov_matrix_12 == cov_matrix_21'
    mu1 = means[1:n_prev]
    mu2 = means[n_prev+1:n_prev+n_new]
    conditional_mu = mu2 + cov_matrix_21 * (cov_matrix_11 \ (ys - mu1))
    conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 * (cov_matrix_11 \ cov_matrix_12)
    conditional_cov_matrix = 0.5 * conditional_cov_matrix + 0.5 * conditional_cov_matrix'
    (conditional_mu, conditional_cov_matrix)
end

"""
Predict output values for some new input values
"""
function predict_ys(f_vec, noise::Float64,
                    xs::Vector{Point2d}, ys::Vector{Float64},
                    new_xs::Vector{Point2d})
    (conditional_mu, conditional_cov_matrix) = compute_predictive(
        f_vec, noise, xs, ys, new_xs)
    mvnormal(conditional_mu, conditional_cov_matrix)
end

function grid_centers(xrange :: Bounds, yrange :: Bounds, xres :: Int64, yres :: Int64) :: Vector{Point2d}
    reshape(map_grid(identity, Point2d, xrange, yrange, xres, yres).values, :)
end
