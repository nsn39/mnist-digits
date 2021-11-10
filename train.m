function [new_Theta1, new_Theta2, cost] = train(iterations, alpha, lambda, X, y, size_a, size_b, size_c, Theta1, Theta2)
  cost  = zeros(iterations, 1);
  
  % Run the train routine for no of iterations
  for i = 1:iterations
    [J, del_Theta1, del_Theta2] = computeGradients(X, y, lambda, size_a, size_b, size_c, Theta1, Theta2);
    
    cost(i, 1) = J;
    
    Theta1 += alpha * del_Theta1;
    Theta2 += alpha * del_Theta2;
    
    disp(i);
  endfor
  
  new_Theta1 = Theta1;
  new_Theta2 = Theta2;
endfunction
