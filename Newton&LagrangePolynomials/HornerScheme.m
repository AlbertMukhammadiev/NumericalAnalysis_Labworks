% computing a polynomial using the Horner's method;
% the function assumes that the coefficients of the polynomial
%   are obtained in the order [a_n, a_n-1,..., a_1, a_0];
function [ Px ] = HornerScheme( x, array )
for i = 2 : length(array)
    array(i) = array(i) + array(i - 1) * x;
end
Px = array(length(array));
end