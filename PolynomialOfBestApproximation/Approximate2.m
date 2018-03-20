% builds the polinomial of best approximation on [-1; 1]
% for a polynomial of degree n
% draws Chebyshev polinomial of n-th degree, approximating function
% and alternance points;
% returns an array of values of the expression (Pnx - y)
% at alternance points;
% arguments: array - polynomial coefficients
function [ altCheck ] = Approximate2( array )
numOfNodes = 101;
% the boundaries of the interval [a,b]
a = -1;
b = 1;
dx = 2 / (numOfNodes - 1);
x = a : dx : b;
n = length(array) - 1;
max = 0;

for i = 1 : numOfNodes
    xn(i) = x(i) ^ n;
    
    % (n - 1)-th tail of the given polinomial
    tail(i) = 0;
    for k = 1 : n
        tail(i) = tail(i) + (x(i) ^ (k - 1)) * array(k);
    end
    
    % the given polinomial
    Pnx(i) = array(n + 1) * xn(i) + tail(i);
    
    % Chebyshev polynomial
    Tnx(i) = cos (n * acos(x(i)));
    % approximating function
    y(i) = (xn(i) - Tnx(i) / 2 ^ (n - 1)) * array(n + 1) + tail(i);
    
    % searching of a max for search of alternance points
    norm = abs(xn(i) - y(i));
    if norm > max
        max = norm;
    end
end

% alternance points
k = 1;
E = 1 / 10^4;
for i = 1 : numOfNodes
    if ((abs(xn(i) - y(i)) <= max + E) && (abs(xn(i) - y(i)) >= max - E))
        alt(k) = a + dx * (i - 1);
        altCheck(k) = Pnx(i) - y(i);
        k = k + 1;
    end
end

nStr = int2str(n);
PnStr = '';
for k = n + 1 : -1 : 2
    if (array(k)) ~= 0
        PnStr = [PnStr int2str(array(k)) 'x^' int2str(k - 1) '+ '];
    end
end;
PnStr = [PnStr int2str(array(1))];

hold on;

PnxPlot = plot(x, Pnx, 'DisplayName', PnStr);
xnPlot = plot(x, xn, 'DisplayName', strrep('f(x)=x^n', 'n', nStr));
TnPlot = plot(x, Tnx, 'DisplayName', strrep('T_n(x)', 'n', nStr));
yPlot =  plot(x, y, 'DisplayName', 'approximating function');
altPlot = plot(alt, 0, 'o', 'DisplayName', 'alternance point');
hold off;

set(PnxPlot, 'LineWidth', 1.5);
set(xnPlot, 'LineWidth', 1.5);
set(TnPlot, 'LineWidth', 1.5);
set(yPlot, 'LineWidth', 1.5);
set(altPlot, 'LineWidth', 2);
legend('show');
end