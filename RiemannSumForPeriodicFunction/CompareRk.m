% Shows difference betwen periodic error R_k
% arguments:
%   func - function
%   n - number of nodes
%   accuracyDeg - max degree of accuracy of the error
function [ ] = CompareRk(func, n, accuracyDeg)
    T = pi;
    a = 0;
    b = a + T;
    for k = 1 : n
        actual(k) = ActualError(func, a, k, T);
        for deg = 1 : accuracyDeg
            periodic(deg, k) = TheorPeriodicError(func, a, b, k, T, deg);
        end
    end
    
    f = figure;
    lineWidth = 1.5;
    xs = 1 : 1 : n;
    semilogy(xs, actual,...
        'DisplayName','actual error',...
        'LineWidth', lineWidth);
    hold on;
    semilogy(xs, periodic(1,:),...
        'DisplayName','R_1',...
        'LineWidth', lineWidth);
    for i = 2 : accuracyDeg
        semilogy(xs, periodic(i,:),...
            'DisplayName', strrep('R_#', '#', int2str(i)),...
            'LineWidth', lineWidth);
    end
    legend('show');
    xlabel('number of nodes');
    title('Comparison of R_k');
    grid on;
    hold off;
    print(f, '-dpng', '-r300', 'ComparisonOfRk');
end