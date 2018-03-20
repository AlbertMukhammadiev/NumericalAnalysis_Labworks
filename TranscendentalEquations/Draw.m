% Draws function 'func' and constant function = 0
function [ ] = Draw( func )
    f = symfun(sym(func), sym('x'));
    a = 0;
    b = 4;
    xs = a : 0.05 : b;
    hold on;
    grid on;
    plot(ones(20) * 0);
    plot(xs, f(xs));
end