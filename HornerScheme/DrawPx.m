% Draws polinomial with thease arguments:
% [a; b] - definition area
% n - the number of the derivative,
% array - polynomial coefficients
function [ ] = DrawPx( a, b, n, array )
numOfNodes = 51;
dx = (b - a) / (numOfNodes - 1);
x = a : dx : b;
for k = 1 : numOfNodes
    Px(k) = HornerSchemeDerivative(a + (k - 1) * dx, array, n);
end

PxPlot = plot(x, Px);
set(PxPlot, 'LineWidth', 1.5);
end