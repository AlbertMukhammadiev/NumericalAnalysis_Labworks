% returns roots of Legendre polynomial
function [ xs ] = GetLegendreRoots( n )
    P(1) = sym('1');
    P(2) = sym('x');
    for k = 2 : n
        P(k + 1) = ((2 * k - 1) * sym('x') * P(k) - (k - 1) * P(k - 1)) / k;
    end
    xs = vpa(solve(P(n + 1)), 15);
end