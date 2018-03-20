function [ ] = DisplayError( )
x0 = 0.5;
df = 1.058421499379;
format long;
for i = 1 : 6
    epsilon(i) = 1 / 10^( 7 - i)
    x = [x0 (x0 + epsilon(i)) (x0 - epsilon(i)) (x0 + epsilon(i) * 2) (x0 - epsilon(i) * 2)];
    y = f(x);
    div_difference = DividedDifferences(x, y, length(x));
    dfApprox = 24 * div_difference(1, length(x));
    error(i) = dfApprox - df;
end

fig = figure;
errorPlot = loglog(epsilon, error, '-s');
set(errorPlot, 'LineWidth', 1.5);
title('Dependence between epsilon and error');
xlabel('epslion');
ylabel('d^4f - 4! f(x_0,..,x_4)');
print(fig, '-dpng', '-r300', 'ErrorDivDiffs');
end

