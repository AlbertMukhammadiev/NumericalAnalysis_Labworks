% returns first derivative of Legendre polynomial
function [ dPnx ] = dLegendrePolynomial( x, n )
    if n == 1
        dPnx = 1;
        return;
    end

    P = [1 x];
    for k = 2 : n
        P(k + 1) = ((2 * k - 1) * x * P(k) - (k - 1) * P(k - 1)) / k;
    end
    dPnx = n / (1 - x^2) * (P(n) - x * P(n + 1));
end