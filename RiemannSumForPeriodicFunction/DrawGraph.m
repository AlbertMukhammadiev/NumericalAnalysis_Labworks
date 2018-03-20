% Draws graph of errors
function [ ] = DrawGraph( func, n )
    T = pi;
    a = 0;
    b = a + T;
    
    % search of max derivative^(2)
    d2f = simplify(diff(sym(func), sym('x'), 2));
    fplot(d2f, [a , b]);
    max_d2f = sym(max(max(ylim), abs(min(ylim))));
    
    for k = 1 : n
        periodic(k) = TheorPeriodicError(func, a, b, k, T, 4);
        theoretical(k) = TheorError(a, b, k, max_d2f);
        actual(k) = ActualError(func, a, k, T);
        Runge(k) = RungeEstimate(func, a, k, T);
    end
    
    f = figure;
    lineWidth = 1.5;
    xs = 1 : 1 : n;
    semilogy(xs, actual(1 : n),...
        'DisplayName','Actual errors',...
        'LineWidth', lineWidth);
    hold on;
    semilogy(xs, theoretical(1 : n),...
        'DisplayName', 'Theoretical errors',...
        'LineWidth', lineWidth);
    semilogy(xs, Runge(1 : n),...
        'DisplayName', 'Runge',...
        'LineWidth', lineWidth);
    semilogy(xs, periodic(1 : n),...
        'DisplayName', 'Theoretical(periodic) errors',...
        'LineWidth', lineWidth);
    legend('show');
    title('Riemann sum');
    grid on;
    hold off;
    print(f, '-dpng', '-r300', 'Errors');
end