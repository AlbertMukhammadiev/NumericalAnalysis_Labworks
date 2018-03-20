% approximates the root of the transcendental equation using the Secant
%   method
% arguments:
%   func - string representation of the function
%   x1, x2 - initial points
%   eps - the upper bound on the relative error, which affects the quality
%       of the approximation
function [ xs ] = SecantMethod( func, x1, x2, eps )
    f = symfun(sym(func), sym('x'));
    if (f(x1) * f(x2) > 0)
        display('points are chosen incorrectly');
        return;
    end
    
    condition = true;
    xs = sym([x1 x2]);
    firstNo = length(xs) + 1;
    k = 2;
    while condition
        xs(k + 1) = vpa(xs(k) - f(xs(k)) * (xs(k) - xs(k - 1))...
                    / (f(xs(k)) - f(xs(k - 1))));
        k = k + 1;
        condition = (abs(xs(k) - xs(k - 1)) > eps);
    end
    xs = xs(firstNo : length(xs));
end