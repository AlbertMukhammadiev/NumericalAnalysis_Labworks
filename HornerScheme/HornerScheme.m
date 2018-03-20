% Computing a polynomial using the Horner's method
function [ Px ] = HornerScheme( x, array )
for i = 2 : length(array)
    array(i) = array(i) + array(i - 1) * x;
end
Px = array(length(array));
end