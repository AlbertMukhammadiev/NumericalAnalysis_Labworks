% computes a definite integral on the interval [a b]
% using Trapezoid rule(n - number of nodes)
function [ S, err ] = TrapezoidRule( func, a, b, n )    
    foo = symfun(sym(func), sym('x'));
    d2foo = diff(foo, 2);
    n = n - 1;
    dx = (b - a) / n;
    xs = a : dx : b;
    ys = foo(xs);
    S = 0;
    err = 0;
    for k = 1 : n
        delta = xs(k + 1) - xs(k);
        S_trap = (ys(k + 1) + ys(k)) / 2 * delta;
        S = S + eval(S_trap);
        
        ksi = xs(k) + delta * 0.3;
        err = err - delta^3 * eval(d2foo(ksi)) / 12;
    end
end