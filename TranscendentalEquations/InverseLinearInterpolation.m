% approximates the root of the transcendental equation using the Inverse
%   Linear method of interpolation
% arguments:
%   func - string representation of the function
%   x1, x2 - points for interpolation
%   eps - the upper bound on the relative error, which affects the quality
%       of the approximation
function [ xs ] = InverseLinearInterpolation( func, x1, x2, eps )
    xs = sym([x1 x2]);
    firstNo = length(xs) + 1;
    k = 2;
    f = symfun(sym(func), sym('x'));
    condition = true;

    while condition
        ys = vpa(f(xs(k - 1 : k)));
        Q1 = LagrangePolynomial(ys, xs(k - 1 : k));
        funcQ = symfun(sym(Q1), sym('x'));
        xs(k + 1) = funcQ(0);
        k = k + 1;
        condition = (abs(xs(k) - xs(k - 1)) > eps);
    end
    xs = xs(firstNo : length(xs));
end