% draws function and interpolation polinomial
%   in the Newton form;
function [ ] = Interpolation_NewtonForm( nodesNum )
a = 0; b = 1;

% function
x = 0 : 0.001 : 1;
fx = f(x);

% interpolation
dx = (b - a) / (nodesNum - 1);
nodes = 0 : dx : 1;
Y = DividedDifferences(nodes, f(nodes), length(nodes));
coefficients = GetNewtonPolinomial(nodes, Y);
for i = 1 : length(x)
    px(i) = HornerScheme(x(i), coefficients);
end

h = figure;
title('Interpolation polynomial in the Newton form');
hold on;
grid on;
pxPlot = plot(x, px, 'DisplayName',...
                strrep('P_{n}(x)', 'n', int2str(nodesNum - 1)));
fxPlot = plot(x, fx, 'DisplayName', 'sin^5(5x)');
nodesPlot = plot(nodes, f(nodes), 'o', 'DisplayName',...
                'interpolation nodes');
set(pxPlot, 'LineWidth', 1.5);
set(fxPlot, 'LineWidth', 1.5);
set(nodesPlot, 'LineWidth', 1.5);
legend('show');
print(h, '-dpng', '-r300', 'graph2');
end