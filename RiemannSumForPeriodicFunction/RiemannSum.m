% calculates an approximate value of the integral using Midpoint rule
% arguments:
%   func - function
%   a - left boarder
%   n - number of nodes
%   T - period
function [ S ] = RiemannSum( func, a, n, T )
    f = symfun(sym(func), sym('x'));
    b = a + T;
    dx = (b - a) / n;
    x = a;
    S = sym(0);
    for i = 0 : n - 1
        middle = (x + x + dx) / 2;
        S = S + eval(f(middle) * dx);
        x = x + dx;
    end
    S = vpa(S);
end