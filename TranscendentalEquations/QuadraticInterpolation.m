% approximates the root of the transcendental equation using the Direct
%   Quadratic method of interpolation
% arguments:
%   func - string representation of the function
%   x1, x2, x3 - points for interpolation
%   eps - the upper bound on the relative error, which affects the quality
%       of the approximation
function [ xs ] = QuadraticInterpolation( func, x1, x2, x3, eps )
    xs = sym([x1 x2 x3]);
    firstNo = length(xs) + 1;
    k = 3;
    f = symfun(sym(func), sym('x'));
    condition = true;

    while condition
        ys = vpa(f(xs(k - 2 : k)));
        P2 = LagrangePolynomial(xs(k - 2 : k), ys);
        roots = solve(P2);
        xs(k + 1) = min(abs(roots));
        k = k + 1;
        condition = (abs(xs(k) - xs(k - 1)) > eps);
    end
    xs = xs(firstNo : length(xs));
end