% approximates the root of the transcendental equation using the Inverse
%   Quadratic method of interpolation
% arguments:
%   func - string representation of the function
%   x1, x2, x3 - points for interpolation
%   eps - the upper bound on the relative error, which affects the quality
%       of the approximation
function [ xs ] = InverseQuadraticInterpolation( func, x1, x2, x3, eps )
    xs = sym([x3 x2 x1]);
    firstNo = length(xs) + 1;
    k = 3;
    f = symfun(sym(func), sym('x'));
    condition = true;

    while condition
        ys = vpa(f(xs(k - 2 : k)));
        Q2 = LagrangePolynomial(ys, xs(k - 2 : k));
        funcQ = symfun(sym(Q2), sym('x'));
        xs(k + 1) = funcQ(0);
        k = k + 1;
        condition = (abs(xs(k) - xs(k - 1)) > eps);
    end
    xs = xs(firstNo : length(xs));
end