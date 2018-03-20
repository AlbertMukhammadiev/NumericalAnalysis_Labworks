% returns approximation of integral using Gauss quadrature
function [ S ] = GaussLegendreQuadrature( func, a, b, n )
    format long;
    if n > 5
        n = 5;
    end
    if n < 1
        n = 1;
    end

    S = 0;
    xs = GetLegendreRoots(n);
    for i = 1 : n
        dPnx = dLegendrePolynomial(xs(i), n);
        w = 2 / (1 - xs(i) ^ 2) / dPnx ^ 2;
        x = ((b - a) / 2 * xs(i) + (a + b) / 2);
        S_t = eval(w * eval(func));
        S = S + S_t;
    end
    S = (b - a) / 2 * S;
end