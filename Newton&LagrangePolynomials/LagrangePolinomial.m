% returns the value of the interpolation polynomial in the Lagrange form;
% arguments:
%   x - array of nodes
%   y - value in these nodes
%   x0 - the point at which the value of the polynomial is calculated
function [ y0 ] = LagrangePolinomial( x, y, x0)
if (length(x) ~= length(y))
    error('The size of arrays a and b should be the same');
end

n = length(y);
y0 = 0;
for k = 1 : n
    product = 1;
    for j = 1 : n
        if (k == j)
            continue;
        end
        product = product * (x0 - x(j)) / (x(k) - x(j));
    end
    
    y0 = y0 + y(k) * product;
end