using Plots

n = 0.5
k1 = 1.66739
k2 = 5.88868
f0_1 = -0.10939238541917934
f1_1 = 0.6820299487786214
f2_1 = 0.7826461830054611
f3_1 = -9.50051023478937
f4_1 = 25.963924602703283
f5_1 = 37.594136439099046
f6_1 = -6.262541451779483

# f(x)
function f(x)
    if x <= 1
        result = 0
        derivatives = [f0_1, f1_1, f2_1, f3_1, f4_1, f5_1, f6_1]
        for m in 0:6
            result += ((-1)^m) * (derivatives[m+1] / factorial(m)) * (x^m)
        end
        return result
    else
        return exp(-k1 * x) * ((f1_1 + k1 * f0_1) * (sin(k2 * x) / k2) + f0_1 * cos(k2 * x))
    end
end

# f'(z)
function f_prime(z)
    if z <= 1
        result = 0
        derivatives = [f1_1, f2_1, f3_1, f4_1, f5_1, f6_1, 0]
        for m in 0:5
            result += ((-1)^m) * (derivatives[m+1] / factorial(m)) * ((1 - z)^m)
        end
        return result
    else
        return exp(-k1 * (z - 1)) * (((f1_1 + k1 * f0_1) * (sin(k2 * (z - 1))) / k2) + f1_1 * cos(k2 * (z - 1)))
    end
end

# G(z)
function G(z)
    return (f(z) * f_prime(z)) / (2 * π * n * (1 + f(z)))
end

function factorial(n)
    if n == 0
        return 1
    end
    result = 1
    for i in 1:n
        result *= i
    end
    return result
end

# метод трапеций
function trapezoidal(z, s_values)
    num_segments = 1000
    h = 50 / num_segments

    integral1 = 0
    for i in 0:num_segments-1
        x1 = i * h
        x2 = (i + 1) * h

        s1 = get_s_value(s_values, abs(z - x1))
        s2 = get_s_value(s_values, abs(z - x2))

        integral1 += ((z - x1) * f(x1) * s1 + (z - x2) * f(x2) * s2) * h / 2
    end

    integral2 = 0
    for i in 0:num_segments-1
        x1 = z + i * h
        x2 = z + (i + 1) * h

        s1 = get_s_value(s_values, x1)
        s2 = get_s_value(s_values, x2)

        integral2 += (x1 * s1 + x2 * s2) * h / 2
    end
    return G(z) + integral2 - integral1
end

function get_s_value(s_values, z)
    for item in s_values
        if abs(item["z"] - z) < 0.001
            return item["s"]
        end
    end
    return 0
end

function calculate_s()
    start_time = time()

    z_values = range(0, 10, length=500)
    s_values = []

    for z in z_values
        s = trapezoidal(z, s_values)
        push!(s_values, Dict("z" => z, "s" => s))
    end

    end_time = time()
    println("Время выполнения: ", end_time - start_time, " секунд")
    return s_values
end

# Построение графика
function plot_s_values(s_values)
    z_values = [item["z"] for item in s_values]
    s_values = [item["s"] for item in s_values]

    plt = plot(z_values, s_values, label="S(z)", xlabel="z", ylabel="S(z)", title="S(z) Function Approximation", legend=:topright, grid=true)
    display(plt)
end

s_values = calculate_s()
plot_s_values(s_values)