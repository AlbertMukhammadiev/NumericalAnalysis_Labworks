function [ y ] = f( x )
for i = 1 : length(x)
    y(i) = (sin(5 * x(i))) ^ 5;
end
end

