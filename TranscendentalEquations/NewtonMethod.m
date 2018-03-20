% approximates the root of the transcendental equation using the Newton
%   method
% arguments:
%   func - string representation of the function
%   x0 - initial approximation
%   eps - the upper bound on the relative error, which affects the quality
%       of the approximation
function [ xs ] = NewtonMethod( func, x0, eps )
    f = symfun(sym(func), sym('x'));
    df = symfun(diff(func, sym('x')), sym('x'));
    
    k = 1;
    xs(k) = vpa(x0 - f(x0) / df(x0));
    firstNo = length(xs) + 1;
    condition = true;
    while condition
        k = k + 1;
        x = xs(k - 1);
        xs(k) = vpa(x - f(x) / df(x));
        condition = (abs(xs(k) - xs(k - 1)) > eps);
    end
    xs = xs(firstNo : length(xs));
end