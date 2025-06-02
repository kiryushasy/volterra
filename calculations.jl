using QuadGK
using NLsolve
using SpecialFunctions: factorial
using Plots

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

# alpha_new, beta_new, gamma_new, delta_new
alpha_new(f0, f1, eta) = (1/6) * (f0 * psi_2(eta) - f1 * psi_1(eta)) - (1/2) * f0 * psi_1(eta)
beta_new(f0, f1, eta) = f0 * psi_1(eta) - (1/2) * (f0 * psi_2(eta) - f1 * psi_1(eta))
gamma_new(f0, f1, eta) = (psi_2(eta) - psi_1(eta)) * f0 - psi_1(eta) * f1
delta_new(f0, f1, eta) = f1 * psi_1(eta) - f0 * psi_2(eta)

# Система уравнений k1 и k2
function solve_k!(F, vars, eta)
    k1, k2 = vars
    integrand_k1(x) = -x * (alpha_val(eta) + beta_val(eta) * x + gamma_val(eta) * x^3) * sinh(k1 * x) * cos(k2 * x)
    integrand_k2(x) = -x * (alpha_val(eta) + beta_val(eta) * x + gamma_val(eta) * x^3) * cosh(k1 * x) * sin(k2 * x)
    integral_k1 = 24 * eta * quadgk(integrand_k1, 0, 1)[1]
    integral_k2 = 24 * eta * quadgk(integrand_k2, 0, 1)[1]
    F[1] = integral_k1 - k1
    F[2] = integral_k2 - k2
end

# Система уравнений f0-f6
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

# Основной расчет
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
        :k2 => k2
    ))
end

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


for item in data
    println(item)
end

z_values = range(0, 8, length=500)
plt = plot(title="Функция f(z) для разных n",
           xlabel="z",
           ylabel="f(z)",
           legend=:topright,
           grid=true)

for params in data
    n = params[:n_value]
    f_vals = [f(z, params) for z in z_values]
    plot!(plt, z_values, f_vals, label="n = $n", linewidth=2)
end

savefig(plt, "f3n.png")
println("График сохранен как f3n.png")