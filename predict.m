function [a2, a3] = predict(X, A, B, C, Theta1, Theta2)
  m = size(X, 1);
  z1 = [ones(m, 1) X];
  z2 = z1 * transpose(Theta1);
  a2 = sigmoid(z2);
  a2 = [ones(m, 1) a2];
  z3 = a2 * transpose(Theta2);
  a3 = sigmoid(z3);
  disp(a3(1, :));
endfunction
