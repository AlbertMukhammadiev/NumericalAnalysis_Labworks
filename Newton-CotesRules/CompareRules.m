% draws a comparative table for rules for calculating the integral
function [ ] = CompareRules( func )
    a = 0;
    b = 0.4;
    n = 5;
    
    [S1, err1] = MidpointRule(func, a, b, n);
    [S2, err2] = TrapezoidRule(func, a, b, n);
    [S3, err3] = SimpsonsRule(func, a, b, n);
    
    foo = symfun(sym(func), sym('x'));
    exactV = eval(int(foo, a, b));
    exactValues = {exactV, exactV, exactV};
    approxs = {S1, S2, S3};
    actualErr = {exactV - S1, exactV - S2, exactV - S3};
    theoreticalErr = {err1, err2, err3};
    
    S12 = MidpointRule(func, a, b, ceil(n / 2));
    S22 = TrapezoidRule(func, a, b, ceil(n / 2));
    S32 = SimpsonsRule(func, a, b, ceil(n / 2));
    runge = {(S1 - S12) / 3, (S2 - S22) / 3, (S3 - S32) / 15};
    
    data = [exactValues; approxs; actualErr; theoreticalErr; runge];

    rowNames = {'exact value', 'approximation', 'actual error',...
                'theoretical error', 'Runge estimate'};
    columnNames = {'Midpoint rule', 'Trapezoid rule', 'Simpson''s rule'};
    f = figure;
    t = uitable( f, 'Data', data, 'ColumnName',columnNames,...
                'RowName',rowNames,'Position', [20 20 430 120]);
    print(f, '-dpng', '-r300', 'table');
end