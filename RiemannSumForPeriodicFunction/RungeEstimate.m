% returns Runge estimate of the error for middle Riemann sum
% arguments:
%   f - function
%   [a; b] - interval
%   n - number of nodes
%   T - period
function [ R ] = RungeEstimate( f, a, n, T )
    R = vpa(abs(RiemannSum(f, a, n, T) - RiemannSum(f, a, 2 * n, T)));
end