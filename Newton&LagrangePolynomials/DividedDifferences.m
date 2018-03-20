% returns 2d array of divided differences
%   (f(x) in first column);
% arguments:
%   x - array of nodes, y - array of f(x)
%   m - max exp of divided differences
function [ arr ] = DividedDifferences( x, y, m )
arr(:, 1) = y;
if m > length(x)
    m = length(x);
end

for j = 1 : m
    for i = 1 : length(x) - j
        arr(i,j + 1) = (arr(i + 1, j) - arr(i,j)) / (x(i + j) - x(i));
    end
end
end