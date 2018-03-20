% Returns n-th derivative of polinomial by applying Horner's method n times
function [ result ] = HornerSchemeDerivative( x, array, n )
for k = 0 : n
    for i = 2 : length(array) - k
        array(i) = array(i) + array(i - 1) * x;
    end
end
result = array(length(array) - n) * factorial(n);
end
