% returns approximation of integral using Mehler quadrature
function [ S ] = MehlerQuadrature( func, n )
    S = 0;
    for k = 1 : n
        % i-th root of n-th Chebyshev polynomial
        x = cos((2 * k - 1) * pi / 2 / n);
        S = S + eval(func);
    end
    
    S = S * pi / n;
end