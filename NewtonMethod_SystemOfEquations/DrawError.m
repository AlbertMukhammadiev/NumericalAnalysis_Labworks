% Draws functions and point of approximation
% Draws graph of actual error decreasing
function [ ] = DrawError( )
    eps = sym('1 / 10^10');
%    fs = ["x1^2 + x2^2 - 1", "x2 - sin(x1)"];
    fs = ["x3 - sin(x1)", "x3 - sin(x2)", "x1^2 + x2^2 + x3^2 - 1"];
    figure1 = figure;
    hold on;
    grid on;
    xs = sym('x', [length(fs) 1]);
    F = [];
    for i = 1 : length(fs)
        f = symfun(sym(fs(i)), xs);
        %ezplot(f);
        fimplicit3(f, [-1, 1]);
        F = [F; f];
    end
    
    sln = solve(F);
%    sln = vpa([sln.x1; sln.x2]);
    sln = vpa([sln.x1; sln.x2; sln.x3]);
    vec = sym(ones(length(fs), 1));
%    approximations = NewtonMethod2x2(F, sln - vec, eps);
    approximations = NewtonMethod3x3(F, sln - vec, eps);
    bestApprox = approximations(:, length(approximations));
%    plot(bestApprox, 'rh', 'LineWidth', 1.5);
    scatter3(bestApprox(1), bestApprox(2), bestApprox(3), 'rh', 'LineWidth', 3);
    title('Approximation');
    print(figure1, '-dpng', '-r300', 'Approximation');
    
    figure2 = figure;
    xs = 1 : 1 : length(approximations);
    for i = 1 : length(approximations)
        errors(i) = norm(approximations(:, i) - sln);
    end
    title('Graph Of Error');
    semilogy(xs, errors, 'LineWidth', 1.5);
    print(figure2, '-dpng', '-r300', 'GraphOfError');
end