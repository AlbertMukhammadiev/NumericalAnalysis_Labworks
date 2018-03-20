% approximates the root of the system of equations using the Newton
%   method
% arguments:
%   F - symbolic equation system
%   x0 - initial approximation vector
%   eps - the upper bound on the relative error, which affects the quality
%       of the approximation
function [ xs ] = NewtonMethod3x3(F, x0, eps )
    x = (argnames(F)).';
    Jf = jacobian(F);
    k = 1;
    xs = sym(x0);
    isntAccurate = true;
    while isntAccurate
        eqn = F(xs(1, k), xs(2, k), xs(3, k)) + ...
            + Jf(xs(1, k), xs(2, k), xs(3, k)) * (x - xs(:,k));
        sln = solve(eqn);
        sln = vpa([sln.x1; sln.x2; sln.x2]);
        xs = [xs sln];
        k = k + 1;
        distance = norm(xs(:,k) - xs(:,k - 1));
        isntAccurate = (distance > eps);
    end
end