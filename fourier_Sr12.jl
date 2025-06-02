using QuadGK
using NLsolve
using Dierckx
using Plots
using LinearAlgebra
using SpecialFunctions: factorial
using BenchmarkTools
using Printf

# f0-f6, k1, k2
function compute_parameters()
    n_values = 0.05:0.10:0.85

    # eta, alpha, beta, gamma
    eta_val(n) = π * n / 6
    alpha_val(eta) = (2 * eta + 1)^2 / (eta- 1)^4
    beta_val(eta) = -3 * eta * (2 + eta)^2 / (2 * (eta - 1)^4)
    gamma_val(eta) = eta * (2 * eta + 1)^2 / (2 * (eta - 1)^4)

    # psi
    psi_1(eta) = -(alpha_val(eta) + beta_val(eta) + gamma_val(eta))
    psi_2(eta) = -(alpha_val(eta) + 2 * beta_val(eta) + 4 * gamma_val(eta))
    psi_3(eta) = -(2 * beta_val(eta) + 12 * gamma_val(eta))
    psi_4(eta) = -24 * gamma_val(eta)
    psi_5(eta) = -24 * gamma_val(eta)
    psi_3_0(eta) = -2 * beta_val(eta)

    #alpha_new, beta_new, gamma_new, delta_new
    alpha_new(f0, f1, eta) = (1/6) * (f0 * psi_2(eta) - f1 * psi_1(eta)) - (1/2) * f0 * psi_1(eta)
    beta_new(f0, f1, eta) = f0 * psi_1(eta) - (1/2) * (f0 * psi_2(eta) - f1 * psi_1(eta))
    gamma_new(f0, f1, eta) = (psi_2(eta) - psi_1(eta)) * f0 - psi_1(eta) * f1
    delta_new(f0, f1, eta) = f1 * psi_1(eta) - f0 * psi_2(eta)

    # Система уравнений для k1 и k2
    function solve_k!(F, vars, eta)
        k1, k2 = vars
        integrand_k1(x) = -x * (alpha_val(eta) + beta_val(eta) * x + gamma_val(eta) * x^3) * sinh(k1 * x) * cos(k2 * x)
        integrand_k2(x) = -x * (alpha_val(eta) + beta_val(eta) * x + gamma_val(eta) * x^3) * cosh(k1 * x) * sin(k2 * x)
        integral_k1 = 24 * eta * quadgk(integrand_k1, 0, 1)[1]
        integral_k2 = 24 * eta * quadgk(integrand_k2, 0, 1)[1]
        F[1] = integral_k1 - k1
        F[2] = integral_k2 - k2
    end

    # Система уравнений для f0-f6
    function system_f!(F, vars, eta)
        f0, f1, f2, f3, f4, f5, f6 = vars
        f = [f0, f1, f2, f3, f4, f5, f6]
        
        F[1] = f0 - 12 * eta * sum([(-1)^m * f[m+1] * (
                psi_1(eta) / factorial(m + 2) -
                psi_2(eta) / factorial(m + 3) +
                psi_3(eta) / factorial(m + 4) -
                psi_4(eta) / factorial(m + 5) +
                psi_5(eta) / factorial(m + 6)
            ) for m in 0:6]) + 12 * eta * alpha_new(f0, f1, eta)
        
        F[2] = f1 + 12 * eta * sum([(-1)^m * f[m+1] * (
                psi_1(eta) / factorial(m + 1) -
                psi_2(eta) / factorial(m + 2) +
                psi_3(eta) / factorial(m + 3) -
                psi_4(eta) / factorial(m + 4) +
                psi_5(eta) / factorial(m + 5)
            ) for m in 0:6]) - 12 * eta * beta_new(f0, f1, eta)
        
        F[3] = f2 + 12 * eta * (psi_1(eta) + sum([(-1)^m * f[m+1] * (
                psi_2(eta) / factorial(m + 1) -
                psi_3(eta) / factorial(m + 2) +
                psi_4(eta) / factorial(m + 3) -
                psi_5(eta) / factorial(m + 4)
            ) for m in 0:6])) + 12 * eta * gamma_new(f0, f1, eta)
        
        F[4] = f3 + 12 * eta * (psi_2(eta) + sum([(-1)^m * f[m+1] * (
                psi_3(eta) / factorial(m + 1) -
                psi_4(eta) / factorial(m + 2) +
                psi_5(eta) / factorial(m + 3)
            ) for m in 0:6])) - 12 * eta * delta_new(f0, f1, eta)
        
        F[5] = f4 + 12 * eta * (2 * f0 * psi_3_0(eta) + psi_3(eta) + sum([(-1)^m * f[m+1] * (
                psi_4(eta) / factorial(m + 1) -
                psi_5(eta) / factorial(m + 2)
            ) for m in 0:6]))
        
        F[6] = f5 + 12 * eta * (2 * f1 * psi_3_0(eta) + psi_4(eta) + sum([(-1)^m * f[m+1] * (
                psi_5(eta) / factorial(m + 1)
            ) for m in 0:6]))
        
        F[7] = f6 + 12 * eta * (2 * f2 * psi_3_0(eta) + 2 * f0 * psi_5(eta) + psi_5(eta))
    end

    data = []
    for n in n_values
        eta = eta_val(n)

        k_solution = nlsolve((F, x) -> solve_k!(F, x, eta), [4.07, 4.76])
        k1, k2 = k_solution.zero[1], k_solution.zero[2]

        f_solution = nlsolve((F, x) -> system_f!(F, x, eta), ones(7))
        f0, f1, f2, f3, f4, f5, f6 = f_solution.zero
        
        push!(data, Dict(
            :n_value => n,
            :n => eta,
            :f0 => f0,
            :f1 => f1,
            :f2 => f2,
            :f3 => f3,
            :f4 => f4,
            :f5 => f5,
            :f6 => f6,
            :k1 => k1,
            :k2 => k2,
            :alpha => alpha_val(eta),
            :beta => beta_val(eta),
            :gamma => gamma_val(eta)
        ))
    end
    
    return data
end

# Параметры вычислений
const k_max = 300.0
const r12_range = range(0.1, 10, length=500)

function f(x::Float64, params::Dict)
    if x ≤ 1.0
        terms = [(-1)^m * params[Symbol("f$m")] / factorial(m) * (1-x)^m for m in 0:6]
        return sum(terms)
    else
        f0 = params[:f0]
        f1 = params[:f1]
        k1 = params[:k1]
        k2 = params[:k2]
        exp_term = exp(-k1 * (x - 1))
        sin_term = sin(k2 * (x -1)) / k2
        cos_term = cos(k2 * (x - 1))
        return exp_term * ((f1 + k1 * f0) * sin_term + f0 * cos_term)
    end
end

function f_prime(x::Float64, params::Dict)
    if x ≤ 1.0
        terms = [(-1)^m * params[Symbol("f$(m+1)")] / factorial(m) * (1-x)^m for m in 0:5]
        return sum(terms)
    else
        f0 = params[:f0]
        f1 = params[:f1]
        k1 = params[:k1]
        k2 = params[:k2]
        exp_term = exp(-k1 * (x - 1))
        sin_term = sin(k2 * (x -1)) / k2
        cos_term = cos(k2 * (x - 1))
        return exp_term * (-(f1 + k1 * f0) * sin_term + f1 * cos_term)
    end
end

function f_double_prime(x::Float64, params::Dict)
    if x ≤ 1.0
        terms = [(-1)^m * params[Symbol("f$(m+2)")] / factorial(m) * (1-x)^m for m in 0:4]
        return sum(terms)
    else
        f0 = params[:f0]
        f1 = params[:f1]
        f2 = params[:f2]
        k1 = params[:k1]
        k2 = params[:k2]
        exp_term = exp(-k1 * (x - 1))
        sin_term = sin(k2 * (x -1)) / k2
        cos_term = cos(k2 * (x - 1))
        return exp_term * ((f1 + k1 * f0) * sin_term + f2 * cos_term)
    end
end

function F_function(z::Float64, params::Dict)
    f_z = f(z, params) # Функция f(z)
    f_p = f_prime(z, params)  # Первая производная
    numerator = f_z * f_p
    denominator = 2π * params[:n] * (1 + f_z)
    abs(denominator) < 1e-10 ? 0.0 : numerator / denominator
end

function P_function(z::Float64, params::Dict)
    f_z = f(z, params)   # Функция f(z)
    f_p = f_prime(z, params)  # Первая производная
    f_pp = f_double_prime(z, params)  # Вторая производная
    numerator = f_p^2 + f_z * f_pp * (1 + f_z)
    denominator = 2 * π * params[:n] * (1 + f_z)^2
    abs(denominator) < 1e-10 ? 0.0 : numerator / denominator
end

function compute_Fk(k::Float64, params::Dict)
    integrand(z) = P_function(z, params) * sin(k * z)
    integral, _ = quadgk(integrand, 1e-6, k_max, rtol=1e-8)
    2 * integral
end

function compute_fk(k::Float64, params::Dict)
    integrand(z) = f_prime(z, params) * cos(k * z)
    integral, _ = quadgk(integrand, 1e-6, k_max, rtol=1e-8)
    2 * integral
end

function compute_Sk(k::Float64, params::Dict, Fk_cache::Dict, fk_cache::Dict)
    if !haskey(Fk_cache, k)
        Fk_cache[k] = compute_Fk(k, params)
    end
    if !haskey(fk_cache, k)
        fk_cache[k] = compute_fk(k, params)
    end
    f0 = f(0.0, params)
    Fk = Fk_cache[k]
    fk = fk_cache[k]
    denominator = 1 + f0 + fk
    abs(denominator) < 1e-10 ? 0.0 : Fk / denominator
end

function main_computation(data)
    results = Dict{Float64, Vector{Float64}}()
    k_grid = range(0.01, k_max, length=500)
    
    open("S_r12_results_julia.txt", "w") do file
        println(file, "n r12 S(r12)")
        
        for params in data
            n_val = params[:n_value]
            println("Processing n = $n_val")
            
            Fk_cache = Dict{Float64, Float64}()
            fk_cache = Dict{Float64, Float64}()
            f0 = f(0.0, params)
            
            Sk_values = [compute_Sk(k, params, Fk_cache, fk_cache) for k in k_grid]
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

                @printf(file, "%.2f %.4f %.8f\n", n_val, r12, Sr_values[i])
            end
            
            results[n_val] = Sr_values
        end
    end
    
    return results
end

println("Вычисление параметров...")
data = compute_parameters()
println("Вычисление S(r12)...")
results = @time main_computation(data)

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

savefig(plt, "S_r12_calculated_jul.png")
println("График сохранен как S_r12_calculated.png")