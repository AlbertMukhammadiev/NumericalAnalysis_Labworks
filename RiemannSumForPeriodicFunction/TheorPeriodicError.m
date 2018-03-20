% returns theoretical(periodic) error of middle Riemann sum
% arguments:
%   f - function
%   [a; b] - interval
%   n - number of nodes
%   T - period
%   accuracyDeg - degree of accuracy
function [ R ] = TheorPeriodicError( f, a, b, n, T, accuracyDeg )
    R = 0;
    for k = 1 : accuracyDeg
        a_n = vpa(2 / T * int(f * cos(2 * pi * k * n * sym('x') / T), sym('x'), a, b));
        b_n = vpa(2 / T * int(f * sin(2 * pi * k * n * sym('x') / T), sym('x'), a, b));
        R = R + a_n + b_n;
    end
    R = R * pi;
end