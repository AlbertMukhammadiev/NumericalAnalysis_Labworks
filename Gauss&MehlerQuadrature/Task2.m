% draws graph of errors for Mehler quadrature
function [ ] = Task2( func )
    format long;
    a = -1;
    b = 1;
    n = 10;
    xs = 1 : 1 : n;
    
    I = eval(int(func / sym('sqrt(1 - x^2)'), sym('x'), a, b));
    for k = 1 : n
        % search of max difference in [a, b]
%         df = diff(func, sym('x'), 2 * k);
%         x = a : 0.01 : b;
%         ys = eval(df);
%         maxDiff = max(abs(ys));
        
        df = diff(func, sym('x'), 2 * k);
        x = a / 2 + b / 2;
        maxDiff = abs(eval(df));

        theoreticalErrors(k) = pi * maxDiff / factorial(2 * k) / (2^(2 * k - 1));
        Sums(k) = MehlerQuadrature(func, k);
        actualErrors(k) = abs(I - Sums(k));
    end
    
    f = figure;
    title('Mehler Quadrature');
    hold on;
    grid on;
    actErrPlot = semilogy(xs, actualErrors, 'DisplayName', 'Actual errors');
    theorErrPlot = semilogy(xs, theoreticalErrors,...
                            'DisplayName', 'Theoretical errors');
    set(actErrPlot, 'LineWidth', 1.5);
    set(theorErrPlot, 'LineWidth', 1.5);
    legend('show');
    print(f, '-dpng', '-r300', 'Mehler');
end