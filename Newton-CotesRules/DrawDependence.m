% draws the dependence of errors on the number of nodes
function [ ] = DrawDependence( someRule, func )
    a = 0;
    b = 0.4;
    xs = 5 : 1 : 21;
    xsRunge = 5 : 2 : 21;
    foo = symfun(sym(func), sym('x'));
    IntF = eval(int(foo, a, b));
    
    theta = 3;
    if (someRule(1:12) == 'SimpsonsRule')
        theta = 15;
    end
    
    for k = 5 : 21
        [S, error] = feval(someRule, func, a, b, k);
        i = k - 4;
        actualErrors(i) = IntF - S;
        theoreticalErrors(i) = error;
        
        if (mod(k, 2) == 1)
            S2 = feval(someRule, func, a, b, ceil(k / 2));
            RungeEstimates(i - floor(i / 2)) = (S - S2) / theta;
        end
    end
    
    f = figure;
    title(someRule);
    hold on;
    grid on;
    actErrPlot = plot(xs, actualErrors, 'DisplayName',...
                      strrep('Actual errors(_)', '_', someRule));
    theorErrPlot = plot(xs, theoreticalErrors, 'DisplayName',...
                        strrep('Theoretical errors(_)', '_', someRule));
    RungePlot = plot(xsRunge, RungeEstimates, 'DisplayName',...
                     strrep('Runge estimate(_)', '_', someRule));
    set(actErrPlot, 'LineWidth', 1.5);
    set(theorErrPlot, 'LineWidth', 1.5);
    set(RungePlot, 'LineWidth', 1.5);
    legend('show');
    print(f, '-dpng', '-r300', someRule);
end