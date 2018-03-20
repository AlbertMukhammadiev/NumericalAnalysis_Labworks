function [ y ] = f( x )
    for k = 1 : length(x)
        y(k) = 1 / sqrt(1 + x(k));
    end
end