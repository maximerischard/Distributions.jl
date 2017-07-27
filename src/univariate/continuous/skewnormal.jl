doc"""
    SkewNormal(μ,σ,α)

The *Skew Normal distribution* with mean `μ`, standard deviation `σ`, and skewness `λ` has probability density function

$f(x; \mu, \sigma, \lambda) = \frac{2}{\sigma}\phi\left(\frac{x-\mu}{\sigma}\Phi\left(\frac{\lambda[x-\mu]}{\sigma}\right)$

```julia
SkewNormal(lambda)      # Skew Normal distribution with zero mean, unit variance, and skewness parameter lambda
SkewNormal(mu, sig, lambda)   # Normal distribution with mean mu, variance sig^2, and skewness parameter lambda

params(d)         # Get the parameters, i.e. (mu, sig, lambda)
mean(d)           # Get the mean, i.e. mu
std(d)            # Get the standard deviation, i.e. sig
```

External links

* [Normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Normal_distribution)

"""
immutable SkewNormalDP{T <: Real} <: ContinuousUnivariateDistribution
    ξ::T
    ω::T
    α::T

    SkewNormalDP(ξ, ω, α) = (@check_args(SkewNormalDP, ω > zero(ω)); new(ξ, ω, α))
end

#### Outer constructors
SkewNormalDP{T<:Real}(ξ::T, ω::T, α::T) = SkewNormalDP{T}(ξ, ω, α)
SkewNormalDP{T<:Real}(α::T) = Normal(zero(α), one(α), α)

# #### Conversions
#=convert{T <: Real, S <: Real}(::Type{Normal{T}}, μ::S, σ::S) = Normal(T(μ), T(σ))=#
#=convert{T <: Real, S <: Real}(::Type{Normal{T}}, d::Normal{S}) = Normal(T(d.μ), T(d.σ))=#

@distr_support SkewNormalDP -Inf Inf

#### Parameters

params(d::SkewNormalDP) = (d.ξ, d.ω, d.α)


#### Statistics
delta(d::SkewNormalDP) = d.α/√(1+d.α^2)
mean_z(d::SkewNormalDP) = √(2/π) * delta(d)
std_z(d::SkewNormalDP) = 1 - 2/π * delta(d)^2
mean(d::SkewNormalDP) = d.ξ + d.ω * mean_z(d)

var(d::SkewNormalDP) = abs2(d.ω)*(1-mean_z(d)^2)
std(d::SkewNormalDP) = √var(d)
skewness(d::SkewNormalDP) = (4-π)/2 * mean_z(d)^3 / (1-mean_z(d)^2)^(3/2)

#### Evaluation
pdf(d::SkewNormalDP, x::Real) = 2/d.ω*normpdf((x-d.ξ)/d.ω)*normcdf(d.α*(x-d.ξ)/d.ω)
logpdf(d::SkewNormalDP, x::Real) = log(2)-log(d.ω)+normlogpdf((x-d.ξ)/d.ω)+normlogcdf(d.α*(x-d.ξ)/d.ω)

#=#### Sampling=#

function rand(d::SkewNormalDP) 
    u0 = randn()
    v = randn()
    δ = delta(d)
    u1 = δ * u0 + √(1-δ^2) * v
    return d.ξ + d.ω * sign(u0) * u1
end

##### Central Parametrization

immutable SkewNormalCP{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    γ1::T

    SkewNormalCP(μ, σ, γ1) = new(μ, σ, γ1)#(@check_args(SkewNormalCP, σ > zero(σ)); new(μ, σ, γ1))
end

#### Outer constructors
SkewNormalCP{T<:Real}(ξ::T, ω::T, α::T) = SkewNormalCP{T}(ξ, ω, α)
SkewNormalCP{T<:Real}(α::T) = SkewNormalCP{T}(zero(α), one(α), α)

#### Parameters
params(d::SkewNormalCP) = (d.μ, d.σ, d.γ1)

mean(d::SkewNormalCP) = d.μ
var(d::SkewNormalCP) = abs2(d.σ)
std(d::SkewNormalCP) = d.σ
skewness(d::SkewNormalCP) = d.γ1
function convert{T<:Real}(::Type{SkewNormalCP{T}}, d::SkewNormalDP{T})
    μ = mean(d)
    σ = std(d)
    γ1 = skewness(d)
    return SkewNormalCP{T}(μ, σ, γ1)
end 
function convert{T<:Real}(::Type{SkewNormalDP{T}}, d::SkewNormalCP{T})
#     ξ = d.μ - - d.σ/std_z(d)*mean_z(d)
#     ω = d.σ / std_z(d)
    c = cbrt(2.0*d.γ1 / (4.0-π))
    μ_z = c / √(1+c^2)
    α = √(π/2.) * c / √(1+(1.0-π/2.)*c^2.)
    δ = α / √(1. +α^2.)
    σ_z = 1.0-2.0/π * δ^2.
    ω = sqrt(d.σ^2.0 / (1.0-μ_z^2.0))
    ξ = d.μ - ω*μ_z
    return SkewNormalDP{T}(ξ, ω, α)
end
pdf{T<:Real}(d::SkewNormalCP{T}, x::Real) = pdf(SkewNormalDP{T}(d),x)
logpdf{T<:Real}(d::SkewNormalCP{T}, x::Real) = logpdf(SkewNormalDP{T}(d),x)
function rand{T<:Real}(d::SkewNormalCP{T}) 
    return rand(SkewNormalDP{T}(d))
end

