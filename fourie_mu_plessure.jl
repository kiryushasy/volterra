using QuadGK
using NLsolve
using Dierckx
using Plots
using LinearAlgebra
using SpecialFunctions: factorial
using BenchmarkTools

# f0-f6, k1, k2
function compute_parameters()
    n_values = 0.05:0.05:0.85

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

    # расчет параметров
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

function compute_μ1(data)
    μ1_values = Float64[]
    for params in data
        n = params[:n_value]
        α = params[:alpha]
        β = params[:beta]
        γ = params[:gamma]
        μ1 = 4π * n * (α/3 + β/4 + γ/6)
        push!(μ1_values, μ1)
    end
    return μ1_values
end

function compute_μ2(data)
    μ2_values = Float64[]
    for params in data
        n = params[:n_value]
        f0 = f(0.0, params)
        
        numerator_integral, _ = quadgk(z -> z * P_function(z, params), 0, Inf)
        numerator = 2 * numerator_integral
        
        denominator_integral, _ = quadgk(z -> z * f_prime(z, params), 0, Inf)
        denominator = 1 + f0 + 2 * denominator_integral

        μ2 = abs(denominator) < 1e-10 ? 0.0 : numerator / denominator
        push!(μ2_values, μ2)
    end
    return μ2_values
end

function compute_pressure(data, μ_total, n_values)
    μ_spline = Spline1D(n_values, μ_total; k=3)
    
    pressure_values = Float64[]
    for (i, n) in enumerate(n_values)
        term1 = n * μ_total[i]
        term2 = quadgk(x -> μ_spline(x), 0.03, n, rtol=1e-8)[1]
        P = term1 - term2
        push!(pressure_values, P)
    end
    return pressure_values
end

println("Вычисление параметров...")
data = compute_parameters()

println("Вычисление химического потенциала...")
μ1 = compute_μ1(data)
μ2 = compute_μ2(data)
μ_total = μ1 .+ μ2

n_values = [params[:n_value] for params in data]

println("Вычисление давления...")
pressure_values = compute_pressure(data, μ_total, n_values)

# График суммарного химического потенциала
plt_mu = plot(
    n_values, μ_total,
    label="μ(n) = μ₁ + μ₂", 
    xlabel="n", 
    ylabel="μ",
    title="Суммарный химический потенциал",
    linewidth=2,
    legend=:topright,
    color=:blue,
    grid=true,
    framestyle=:box
)

savefig(plt_mu, "mu_plot.png")
println("График суммарного химического потенциала сохранен как mu_plot.png")

# График давления
plt_p = plot(
    n_values, pressure_values,
    label="P(n)", 
    xlabel="n", 
    ylabel="P",
    title="Давление P(n)",
    linewidth=2,
    legend=:topleft,
    color=:red,
    grid=true,
    framestyle=:box
)

savefig(plt_p, "p_plot.png")
println("График давления сохранен как p_plot.png")

println("n_values: ", n_values)
println("μ_total: ", μ_total)
