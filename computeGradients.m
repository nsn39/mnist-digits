function [J, del_Theta1, del_Theta2] = computeGradients(X, y, lambda, size_a, size_b, size_c, Theta1, Theta2)
      
    % Calculate del_a3
    [a2, a3] = predict(X, size_a, size_b, size_c, Theta1, Theta2);
    del_a3 = y - a3;
    
    % Calculate del_a2
    del_a2 = (del_a3 * Theta2) .* a2 .* (1 .- a2);
    
    del_Theta2 = transpose(del_a3) * a2;
    
    m = size(X, 1);
    n = size(X, 2);
    a1 = [ones(m, 1) X];
    
    del_Theta1 = transpose(del_a2) * a1;
    del_Theta1 = del_Theta1(2:size(del_Theta1, 1), :);
    
    del_Theta1 = ((1/m) .* del_Theta1) + ((lambda/m) .* Theta1);
    del_Theta2 = ((1/m) .* del_Theta2) + ((lambda/m) .* Theta2);
    
    % Calculate cost of the network
    J = (-1/m) * sum(sum( y .* log(a3) + (1 - y) .* log(1 - a3) ));
    J += (lambda/(2*m)) * sum(sum(Theta1 .^ 2));
    J += (lambda/(2*m)) * sum(sum(Theta2 .^ 2));
    
endfunction
