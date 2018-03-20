% returns interpolation polynomial in the Lagrange form;
% arguments:
%   x - array of nodes
%   y - value in these nodes
%   x0 - the point at which the value of the polynomial is calculated
function [ P ] = LagrangePolynomial( x, y )
    if (length(x) ~= length(y))
        error('The size of arrays a and b should be the same');
    end

    n = length(y);
    P = sym('0');
    for k = 1 : n
        product = 1;
        for j = 1 : n
            if (k == j)
                continue;
            end
            product = product * (sym('x') - x(j)) / (x(k) - x(j));
        end

        P = P + y(k) * product;
    end
end