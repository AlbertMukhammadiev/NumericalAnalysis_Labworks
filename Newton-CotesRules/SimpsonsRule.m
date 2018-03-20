% computes a definite integral on the interval [a b]
% using Simpson's rule(n - number of nodes)
function [ S, err ] = SimpsonsRule( func, a, b, n )
    foo = symfun(sym(func), sym('x'));
    d4foo = diff(foo, 4);
    n = n - 1;
    dx = (b - a) / n;
    xs = a : dx : b;
    ys = foo(xs);
    S = 0;
    err = 0;
    for k = 1 : n
        midX = (xs(k) + xs(k + 1)) / 2;
        midY = foo(midX);
        leftY = ys(k);
        rightY = ys(k + 1);
        delta = xs(k + 1) - xs(k);
        S_k = (leftY + 4 * midY + rightY) * delta / 6;
        S = S + eval(S_k);
        
        ksi = xs(k) + delta * 0.3;
        err = err - delta^5 * eval(d4foo(ksi)) / 2880;
    end
end