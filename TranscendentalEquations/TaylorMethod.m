% approximates the root of the transcendental equation using the expansion
%   of the function in the Taylor series
% arguments:
%   func - string representation of the function
%   x0 - initial approximation
%   eps - the upper bound on the relative error, which affects the quality
%       of the approximation
function [ xs ] = TaylorMethod( func, x0, eps )
    f = symfun(sym(func), sym('x'));
    df = symfun(diff(func, sym('x')), sym('x'));
    d2f = symfun(diff(df, sym('x')), sym('x'));
    condition = true;
    k = 1;
    xs(k) = sym(x0);
    firstNo = length(xs) + 1;
    while condition
        T2 = f(xs(k)) + df(xs(k)) * (sym('x') - xs(k)) + ...
             + d2f(xs(k)) * (sym('x') - xs(k)) ^ 2 / 2;
        roots = solve(T2);
        k = k + 1;
        xs(k) = vpa(min(abs(roots)));
        condition = (abs(xs(k) - xs(k - 1)) > eps);
    end
    xs = xs(firstNo : length(xs));
end