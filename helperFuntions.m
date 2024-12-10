function loss = cross_entropy_loss(output, target)
    % Compute cross entropy loss
    [~, num_classes] = size(output);
    target_one_hot = zeros(length(target), num_classes);
    for i = 1:length(target)
        target_one_hot(i,target(i)) = 1;
    end
    loss = -sum(sum(target_one_hot .* log(softmax(output)))) / length(target);
end

function sm = softmax(x)
    % Compute softmax
    ex = exp(x - max(x,[],2));
    sm = ex ./ sum(ex,2);
end

