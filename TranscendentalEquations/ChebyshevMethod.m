% approximates the root of the transcendental equation using the Chebyshev
%   method(expansion of the inverse function in the Taylor series)
% arguments:
%   func - string representation of the function
%   x0 - initial approximation
%   eps - the upper bound on the relative error, which affects the quality
%       of the approximation
function [ xs ] = ChebyshevMethod( func, x0, eps )
    f = symfun(sym(func), sym('x'));
    df = diff(func, sym('x'));
    d2f = diff(df, sym('x'));
    
    % derivatives of inverse function
    dF = symfun(1 / df, sym('x'));
    d2F = symfun(- d2f / df^3, sym('x'));
    
    condition = true;
    k = 1;
    xs(k) = sym(x0);
    firstNo = length(xs) + 1;
    while condition
        k = k + 1;
        dy = sym('y') - f(xs(k - 1));
        F = xs(k - 1) + dy * dF(xs(k - 1)) - dy ^ 2 * d2F(xs(k - 1)) / 2;
        F = symfun(F, sym('y'));
        xs(k) = vpa(F(0));
        condition = (abs(xs(k) - xs(k - 1)) > eps);
    end
    xs = xs(firstNo : length(xs));
end