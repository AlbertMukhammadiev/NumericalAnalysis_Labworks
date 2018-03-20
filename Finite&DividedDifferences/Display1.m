% prints the table of the finite differences
% for f(x), where x = 0.01 * i, i = 0,..10
function [ ] = Display1( )
n = 11;
delta = FiniteDifferences(0, 0.1, n, n - 1);
for i = 1 : n
    name(i) = "x" + int2str(i);
end
name(n + 1) = "max error";

fxTable = table(name', delta(:, 1));
fxTable.Properties.VariableNames([1 2]) = {'x' 'f'};
delta(:,1) = [ ];
deltaTable = array2table(delta);
fxTable = [fxTable deltaTable];
format long;
display(fxTable);
end