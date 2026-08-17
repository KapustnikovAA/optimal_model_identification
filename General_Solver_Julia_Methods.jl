module fast_polynomdif

using LinearAlgebra
"""
    Проводит дифференцирование со сглаживанием полиномом
    Аргуметны:
        \tP --- степень полинома
        \tm --- количество точек в окне
    Выход:
        \tdx --- прямоугольная матрица
            \tстолбцы --- n-а производная
            \tстроки --- k-й элемент массива
"""
function polynomdif(x::Vector{Float64}, dt::Float64, P::Int = 1, m::Int = 3)::Array{Float64}
    
    m2 = div(m - 1, 2)

    # Формируем безразмерное время:
    tn = Vector{Float64}(range(start = -m2, stop = m2, length = m))

    # Формируем базисные функции:
    Phi = ones(m, P + 1)
    for k in 2:P + 1
        Phi[:, k] .= Phi[:, k - 1] .* tn
    end

    # Матрица производных:
    dx = Array{Float64}(undef, length(x) - 2 * m2, P + 1)

    # По всем точкам:
    for n in 1:length(x) - 2 * m2
        dx[n, :] .= Phi \ x[n: n + m - 1]
    end

    # Масштабируем:
    for k in 2:P + 1
        dx[:, k] .*= factorial(k - 1) / (dt^(k - 1))
    end
    
    return dx

end

end