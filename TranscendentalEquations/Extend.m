% extends the array copying the last element, and returns first n elements
function [ xs ] = Extend( xs, n )
    tailNo = length(xs);
    for i = tailNo + 1 : n
        xs(i) = xs(tailNo);
    end
    xs = xs(1 : n);
end

