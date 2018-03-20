% returns coefficients of the Newton polinomial
%   in the order [a_n, a_n-1,..., a_1, a_0];
% arguments:
%   x - array of nodes
%   Y - matrix of divided differences
function [ coeff ] = GetNewtonPolinomial( x, Y )
    arr = 1;
    coeff = Y(1,1);
    for i = 2 : length(x)
        arr = [-x(i - 1) * arr 0] + [0 arr];
        coeff = [coeff 0] + arr * Y(1,i);
    end
    coeff = coeff(end:-1:1);
end