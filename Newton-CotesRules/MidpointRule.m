% computes a definite integral on the interval [a b]
% using Midpoint rule(n - number of nodes)
function [ S, err ] = MidpointRule( func, a, b, n )
    foo = symfun(sym(func), sym('x'));
    d2foo = diff(foo, 2);
    n = n - 1;
    dx = (b - a) / n;
    xs = a : dx : b;
    S = 0;
    err = 0;
    for k = 1 : n
        delta = xs(k + 1) - xs(k);
        t = (xs(k + 1) + xs(k)) / 2;
        S = S + eval(foo(t)) * delta;
        
        ksi = xs(k) + delta * 0.3;
        err = err + delta^3 * eval(d2foo(ksi)) / 24;
    end
end