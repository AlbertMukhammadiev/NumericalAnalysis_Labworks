% returns actual error of middle Riemann rule
% arguments:
%   f - function
%   [a; b] - interval
%   n - number of nodes
%   T - period
function [ R ] = ActualError( f, a, n, T )
    S = RiemannSum(f, a, n, T);
    I = int(f, sym('x'), a, T);
    R = vpa(abs(S - I));
end