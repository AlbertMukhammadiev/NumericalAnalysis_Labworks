% returns theoretical error of the middle Riemann sum
% arguments:
%   [a; b] - interval
%   n - number of nodes
%   max_d2f - max second derivative on [a; b]
function [ R ] = TheorError( a, b, n, max_d2f )
    R = vpa((b - a) * ((b - a) / n) ^ 2 * max_d2f / 24);
end