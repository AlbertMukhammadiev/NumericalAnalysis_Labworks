% Prints the table of the divided differences
% for y = f(x) for given array x.
function [ ] = Display2( )
x = [0.5 0.6 0.4 0.7 0.3];
y = f(x);

div_difference = DividedDifferences(x, y, length(x));
div_difference(:,1) = [ ];
ddTable = [table(y') array2table(div_difference)];
ddTable.Properties.VariableNames(1) = {'f'};
display(ddTable);
end