% draws function and interpolation polinomial
% in the Lagrange form
function [ ] = Interpolation_LagrangeForm( nodesNum )
a = 0; b = 1;

% function
x = 0 : 0.001 : 1;
fx = f(x);

% interpolation
dx = (b - a) / (nodesNum - 1);
nodes = 0 : dx : 1;
for i = 1 : length(x)
    px(i) = LagrangePolinomial(nodes, f(nodes), x(i));
end

h = figure;
title('Interpolation polynomial in the Lagrange form');
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
print(h, '-dpng', '-r300', 'graph1');
end