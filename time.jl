using QuadGK
using Dierckx
using Plots
using LinearAlgebra
using BenchmarkTools

# данные
const data = [
    Dict(
        :n_value => 0.1,
        :n => 0.05235987755982988,
        :f0 => -0.010896299227692,
        :f1 => 0.03533871802446059,
        :f2 => 0.7375798016904298,
        :f3 => 0.4803943311511456,
        :f4 => -0.17149042380406146,
        :f5 => 0.6040419153779827,
        :f6 => -0.17603764656165466,
        :k1 => 4.07155209680121,
        :k2 => 4.76077138142061
    ),
    Dict(
        :n_value => 0.2,
        :n => 0.10471975511965977,
        :f0 => -0.040955430040710734,
        :f1 => 0.14046818273894418,
        :f2 => 1.5705072406894023,
        :f3 => 0.290760706585611,
        :f4 => -0.29726760769015437,
        :f5 => 3.2993377898684724,
        :f6 => -5.249382884498001,
        :k1 => 3.093382794870304,
        :k2 => 5.115610312103863
    ),
    Dict(
        :n_value => 0.3,
        :n => 0.15707963267948966,
        :f0 => -0.07937662775464152,
        :f1 => 0.309341378413055,
        :f2 => 2.20864472248648,
        :f3 => -1.1853851064431131,
        :f4 => 1.2037907201856726,
        :f5 => 9.223225337000306,
        :f6 => -25.939365178223138,
        :k1 => 2.486506582624432,
        :k2 => 5.398200639846574
    ),
    Dict(
        :n_value => 0.4,
        :n => 0.20943951023931953,
        :f0 => -0.10922581443753625,
        :f1 => 0.5135761556026422,
        :f2 => 2.175732478808618,
        :f3 => -4.465794619477479,
        :f4 => 7.908170728448949,
        :f5 => 19.122638943377872,
        :f6 => -60.433398136695736,
        :k1 => 2.0338135055198534,
        :k2 => 5.6506553082306885
    ),
    Dict(
        :n_value => 0.5,
        :n => 0.2617993877991494,
        :f0 => -0.10939238541917934,
        :f1 => 0.6820299487786214,
        :f2 => 0.7826461830054611,
        :f3 => -9.50051023478937,
        :f4 => 25.963924602703283,
        :f5 => 37.594136439099046,
        :f6 => -6.262541451779483,
        :k1 => 1.6673892924723686,
        :k2 => 5.888677094718416
    ),
    Dict(
        :n_value => 0.6,
        :n => 0.3141592653589793,
        :f0 => -0.05724077661682139,
        :f1 => 0.6767397591554315,
        :f2 => -2.72443698126277,
        :f3 => -14.554544245304914,
        :f4 => 63.4380969951293,
        :f5 => 99.83816262213533,
        :f6 => 619.4154447014668,
        :k1 => 1.3575848883484734,
        :k2 => 6.120359242206799
    ),
    Dict(
        :n_value => 0.7,
        :n => 0.3665191429188092,
        :f0 => 0.05624866068181238,
        :f1 => 0.30848844513431417,
        :f2 => -8.266654410734068,
        :f3 => -14.357080355861644,
        :f4 => 128.07626765123518,
        :f5 => 368.5622356348904,
        :f6 => 3181.982947178201,
        :k1 => 1.0894096285297696,
        :k2 => 6.3506224973383585
    ),
    Dict(
        :n_value => 0.8,
        :n => 0.41887902047863906,
        :f0 => 0.18519599692546337,
        :f1 => -0.43488961582191904,
        :f2 => -13.204179656897638,
        :f3 => -1.4729083827571146,
        :f4 => 237.28168978059932,
        :f5 => 1291.1753409262808,
        :f6 => 9583.619392264429,
        :k1 => 0.8549302040496924,
        :k2 => 6.582679041325202
    ),
    Dict(
        :n_value => 0.9,
        :n => 0.47123889803846897,
        :f0 => 0.24602516674204053,
        :f1 => -1.155160916468779,
        :f2 => -14.273806747444455,
        :f3 => 25.71895221593576,
        :f4 => 471.4503752608885,
        :f5 => 3550.364370015452,
        :f6 => 20132.826900627897,
        :k1 => 0.650217090786169,
        :k2 => 6.818520944399062
    )
]

# Параметры вычислений
const k_max = 30.0
const r12_range = range(0.1, 10, length=500)

function f(x::Float64, params::Dict)
    if x ≤ 1.0
        terms = [params[Symbol("f$m")] / factorial(m) * (-x)^m for m in 0:6]
        return sum(terms)
    else
        f0 = params[:f0]
        f1 = params[:f1]
        k1 = params[:k1]
        k2 = params[:k2]
        exp_term = exp(-k1 * x)
        sin_term = (f1 + k1 * f0) * sin(k2 * x) / k2
        cos_term = f0 * cos(k2 * x)
        return exp_term * (sin_term + cos_term)
    end
end

function f_prime(z::Float64, params::Dict)
    if z ≤ 1.0
        terms = [m * params[Symbol("f$m")] / factorial(m) * (-1)^m * z^(m-1) for m in 1:6]
        return sum(terms)
    else
        f0 = params[:f0]
        f1 = params[:f1]
        k1 = params[:k1]
        k2 = params[:k2]
        exp_term = exp(-k1 * (z-1))
        term1 = -k1 * exp_term * ((f1 + k1 * f0) * sin(k2 * z) / k2 + f0 * cos(k2 * z))
        term2 = exp_term * ((f1 + k1 * f0) * cos(k2 * z) - f0 * k2 * sin(k2 * z))
        return term1 + term2
    end
end

function F_function(z::Float64, params::Dict)
    fz = f(z, params)
    fz_deriv = f_prime(z, params)
    denominator = 2π * params[:n] * (1 + fz)
    abs(denominator) < 1e-10 ? 0.0 : (fz * fz_deriv) / denominator
end

function compute_Fk(k::Float64, params::Dict)
    integrand(z) = F_function(z, params) * sin(k * z)
    integral, _ = quadgk(integrand, 1e-6, k_max, rtol=1e-8)
    2 * integral
end

function compute_fk(k::Float64, params::Dict)
    integrand(z) = f(z, params) * cos(k * z)
    integral, _ = quadgk(integrand, 1e-6, k_max, rtol=1e-8)
    2 * integral
end

function compute_Sk(k::Float64, params::Dict, f0::Float64, Fk_cache::Dict, fk_cache::Dict)
    if !haskey(Fk_cache, k)
        Fk_cache[k] = compute_Fk(k, params)
    end
    if !haskey(fk_cache, k)
        fk_cache[k] = compute_fk(k, params)
    end
    
    Fk = Fk_cache[k]
    fk = fk_cache[k]
    denominator = 1 + f0 + fk
    abs(denominator) < 1e-10 ? 0.0 : Fk / denominator
end

function main_computation()
    results = Dict{Float64, Vector{Float64}}()
    k_grid = range(0.01, k_max, length=500)
    
    for params in data
        n_val = params[:n_value]
        println("Processing n = $n_val")
        
        Fk_cache = Dict{Float64, Float64}()
        fk_cache = Dict{Float64, Float64}()
        f0 = f(0.0, params)
        
        Sk_values = [compute_Sk(k, params, f0, Fk_cache, fk_cache) for k in k_grid]
        Sk_spline = Spline1D(k_grid, Sk_values; k=3, bc="zero")
        
        Sr_values = zeros(length(r12_range))
        Threads.@threads for i in eachindex(r12_range)
            r12 = r12_range[i]
            if r12 ≈ 0.0
                Sr_values[i] = 0.0
            else
                integrand(k) = Sk_spline(k) * sin(k * r12) / r12
                integral, _ = quadgk(integrand, 0.01, k_max, rtol=1e-8)
                Sr_values[i] = integral / π
            end
        end
        
        results[n_val] = Sr_values
    end
    
    return results
end

results = @time main_computation()

plt = plot(
    size=(800, 600),
    xlabel="r12", 
    ylabel="S(r12)", 
    title="S(r12)", 
    legend=:topright,
    linewidth=2
)

for (n_val, Sr_values) in sort(collect(results))
    plot!(plt, r12_range, Sr_values, label="n = $n_val")
end

savefig(plt, "S_r12_corrected.png")
println("saved as S_r12_corrected.png")