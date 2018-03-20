% returns 2d array of finite differences
% (f(x) in first column)
% arguments:
% [a,b] - definition area
% n - number of nodes
% m - max exp of finite differences
function [ array ] = FiniteDifferences( a, b, n, m )
dx = (b - a) / (n - 1);
x = a : dx : b;
array(:, 1) = f(x);
for j = 2 : m + 1
    for i = 1 : n - j + 1
        array(i,j) = array(i + 1, j - 1) - array(i, j - 1);
    end
end

% absolute error
relativeErr = 1 / 10^15;
exactValue = f(b);
absoluteErr = exactValue * relativeErr;
for i = 1 : n
    array(m + 2, i) = absoluteErr;
    absoluteErr = absoluteErr * 2;
end
end