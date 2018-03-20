% Draws graph of convergence of various methods
% arguments:
%   func - string representation of the function
%   n - quantity of steps of the algorithm
%   eps - the upper bound on the relative error, which affects the quality
%       of the approximation
function [ ] = CompareMethods( func, n, eps )
    root = solve(func);
    x1 = 0;
    x2 = 1;
    x3 = 1.5;
    xsNewton = NewtonMethod(func, x1, eps);
    xsSecant = SecantMethod(func, x1, x2, eps);
    xsQuadratic = QuadraticInterpolation(func, x1, x2, x3, eps);
    xsInvLinear = InverseLinearInterpolation(func, x1, x2, eps);
    xsInvQuadratic = InverseQuadraticInterpolation(func, x1, x2, x3, eps);
    xsChebyshev = ChebyshevMethod(func, x1, eps);
    xsTaylor = TaylorMethod(func, x1, eps);

    xs = 1 : 1 : n;
    root = Extend(root, n);
    xsNewton = abs(root - Extend(xsNewton, n));
    xsSecant = abs(root - Extend(xsSecant, n));
    xsQuadratic = abs(root - Extend(xsQuadratic, n));
    xsInvQuadratic = abs(root - Extend(xsInvQuadratic, n));
    xsInvLinear = abs(root - Extend(xsInvLinear, n));
    xsChebyshev = abs(root - Extend(xsChebyshev, n));
    xsTaylor = abs(root - Extend(xsTaylor, n));

    lineWidth = 1.3;
    f = figure;
    semilogy(xs, xsInvLinear, 'o',...
            'LineWidth', lineWidth,...
            'DisplayName', 'inverse linear interpolation method');
    hold on;
    semilogy(xs, xsSecant, '--',...
            'LineWidth', lineWidth,...
            'DisplayName', 'Secant method');
    semilogy(xs, xsQuadratic, '-s',...
            'LineWidth', lineWidth,...
            'DisplayName', 'Quadratic interpolation method');
    semilogy(xs, xsInvQuadratic, '-d',...
            'LineWidth', lineWidth,...
            'DisplayName', 'Inverse Quadratic interpolation method');
    semilogy(xs, xsNewton,...
            'LineWidth', lineWidth,...
            'DisplayName', 'Newton method');
    semilogy(xs, xsChebyshev, '-<',...
            'LineWidth', lineWidth,...
            'DisplayName', 'Chebyshev method');
    semilogy(xs, xsTaylor,'->',...
            'LineWidth', lineWidth,...
            'DisplayName', 'Taylor method');
    hold off;
    grid on;
    title('Methods of solving Transcendental equations');
    legend('show');
    print(f, '-dpng', '-r300', 'GraphOfConvergence');
end