function [Theta1, Theta2] = initializeWeights(size_a, size_b, size_c)
    pkg load statistics;
    #Theta1 = zeros(size_b, size_a + 1);
    #Theta2 = zeros(size_c, size_b + 1);
    #Theta1 = randn(size_b, size_a + 1);
    #Theta2 = randn(size_c, size_b + 1);
    Theta1 = normrnd(0, 200 ^ -0.5, size_b, size_a + 1);
    Theta2 = normrnd(0, 10 ^ -0.5, size_c, size_b + 1);
    #disp(Theta1);
    #disp(Theta2);
endfunction
