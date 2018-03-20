% builds the polinomial of best approximation on [0; 1]
% for f(x) = 1 / sqrt(1 + x);
% draws function, chord between points 0 and 1,
% tangent line, that is parallel to the chord, required polinomial
% and alternance points;
% returns an array of values of the expression (f - P)
% at alternance points;
function [ altCheck ] = Approximate1(  )
k = (f(1) - f(0))/(1 - 0);
% The point at which the tangent line is parallel to the chord
x0 = 1 / (-2 * k)^(2 / 3) - 1;
y0 = f(x0);

numOfNodes = 101;
dx = 1 / (numOfNodes - 1);
x = 0 : dx : 1;
max = 0;

for i = 1 : numOfNodes
    % function
    fx(i) = f((i - 1) * dx);
    % tangent
    y_t(i) = (x(i) - x0) * k + y0;
    % chord
    y_c(i) = x(i) * k + f(0);
    % polinomial of best approximation of the first degree
    P1x(i) = x(i) * k + (fx(1) + y_t(1)) / 2;
    
    % searching of a max for search of alternance points
    norm = abs(fx(i) - P1x(i));
    if norm > max
        max = norm;
    end
end

% searching of alternance points
k = 1;
E = 1 / 100^3;
for i = 1 : numOfNodes
    if ((abs(P1x(i) - fx(i)) <= max + E) && (abs(P1x(i) - fx(i)) >= max - E))
        alt(k) = dx * (i - 1);
        altCheck(k) = fx(i) - P1x(i);
        k = k + 1;
    end
end

hold on;
fPlot = plot(x, fx, '-', 'DisplayName', '1 / sqrt(1 + x)');
tangentPlot = plot(x, y_t, '--', 'DisplayName', 'tangent');
chordPlot =  plot(x, y_c, '--', 'DisplayName', 'chord');
P1xPlot = plot(x, P1x, '-', 'DisplayName', 'P_1(x)');
altPlot = plot(alt, 1, 'o', 'DisplayName', 'alternance point');
hold off;

set(fPlot, 'LineWidth', 1.5);
set(tangentPlot, 'LineWidth', 1.5);
set(chordPlot, 'LineWidth', 1.5);
set(P1xPlot, 'LineWidth', 1.5);
set(altPlot, 'LineWidth', 2);
legend('show');
end