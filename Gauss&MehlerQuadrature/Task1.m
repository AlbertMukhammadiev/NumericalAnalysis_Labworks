% draws table for Gauss Legendre quadrature
function [ ] = Task1( func )
    format long;
    a = 0;
    b = 0.4;
    n = 5;
    
    I = eval(int(func, sym('x'), a, b));
    for i = 1 : n
        % search of max difference in [a, b]
        df = diff(func, sym('x'), 2 * i);
        x = a : 0.01 : b;
        ys = eval(df);
        maxDiff = max(abs(ys));
        error = (b - a)^(2 * i + 1) * (factorial(i))^4 /...
                (2 * i + 1) / (factorial(2 * i))^3 * maxDiff;
        
        integrals(i) = I;
        Sums(i) = GaussLegendreQuadrature(func, a, b, i);
        actualErrors(i) = abs(integrals(i) - Sums(i));
        theoreticalErrors(i) = error;
    end
  
    integrals = num2cell(integrals);
    Sums = num2cell(Sums);
    actualErrors = num2cell(actualErrors);
    theoreticalErrors = num2cell(theoreticalErrors);
    data = [integrals; Sums; actualErrors; theoreticalErrors];
    
    rowNames = {'exact value', 'approximation', 'actual error',...
        'theoretical error'};
    columnNames = {'n = 1', 'n = 2', 'n = 3', 'n = 4', 'n = 5'};
    f = figure('Position', [500 500 570 100]);
    t = uitable( f, 'Data', data, 'ColumnName', columnNames,...
                  'RowName',rowNames,'Position', [0 0 565 95]);
    print(f, '-dpng', '-r300', 'table_GaussLegendre');
end