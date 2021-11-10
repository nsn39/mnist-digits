% ==============================================================================
% Load the dataset
trainData = loadDataset("mnist_train.csv");
testData = loadDataset("mnist_test.csv");

% ==============================================================================
% Preprocess the data first
m_train = size(trainData, 1);
n_train = size(trainData, 2);

m_test = size(testData, 1);
n_test = size(testData, 2);

X_train = trainData(2:m_train, 2:n_train);
y_train = trainData(2:m_train, 1);

X_test = testData(2:m_test, 2:n_test);
y_test = testData(2:m_test, 1);

% Final values of m and n
m_train = size(X_train, 1);
n_train = size(y_train, 2);

m_test = size(X_test, 1);
n_test = size(y_test, 2);

% Printing some sample images with labels.
#img_1 = reshape(X_train(5, :), 28, 28);
#imshow(img_1);
#disp(y_train(5,1));

%Normalizing all the pixel values.
mean_train = mean(X_train(:));
mean_test = mean(X_test(:));
std_train = std(X_train(:));
std_test = std(X_test(:));

X_train = X_train .- mean_train;
X_train = X_train ./ std_train;
X_test = X_test .- mean_test;
X_test = X_test ./ std_test;

disp(y_train);

% Mapping each label in y_train to a vector of 0 and 1.
temp = zeros(m_train, 10);
for i = 1:m_train
  for j = 0:9
    if y_train(i,1) == j
      temp(i,j+1) = 0.99;
    else
      temp(i,j+1) = 0.01;
    endif
  endfor
endfor

y_train = temp;


% ==============================================================================
% Initialize the weights of the neural network.
size_a = 784;
size_b = 200;
size_c = 10;
[Theta1, Theta2] = initializeWeights(size_a, size_b, size_c);

#disp(Theta1(1, :));
#disp(Theta2(1, :));

% Define some constants for training the network.
iterations = 200;
l_rate = 0.5;
lambda = 0;
before_training = time();

% Train your neural network.
[new_Theta1, new_Theta2, cost] = train(iterations, l_rate, lambda, X_train, y_train, size_a, size_b, size_c, Theta1, Theta2);
Theta1 = new_Theta1;
Theta2 = new_Theta2;

after_training = time();
% ==============================================================================
% Run predictions on trained model.
[a2, predictions] = predict(X_test, size_a, size_b, size_c, Theta1, Theta2);

for i = 1:size(predictions, 1)
  #disp(predictions(i, :));
endfor

% Convert matrix of 0s and 1s into vector of labels.
[one_v, predictions] = max(transpose(predictions)); 
% Convert all indicies in predictions from 1-10 to 0-9
predictions = predictions .- 1;
predictions = transpose(predictions);

% Display the misclassification rate.
accurate_labels = 0;
for i = 1:m_test
  if predictions(i, 1) == y_test(i, 1)
    accurate_labels += 1;
  endif
endfor

% ==============================================================================
% Plotting necessary graphs here.
cost = cost(:);
iter = 1:iterations;
plot(iter, cost);
title("Network cost along the iterations");

% ==============================================================================
% Debugging display scripts here.

disp("X_train");
disp(size(X_train));

disp("y_train");
disp(size(y_train));

disp("X_test");
disp(size(X_test));

disp("y_test");
disp(size(y_test));

disp("predictions:");
disp(size(predictions));

disp("Classification rate out of 10000");
disp(accurate_labels);

disp("Training done in seconds:");
disp(after_training - before_training);

disp("Predicted output classes");
#disp(predictions);
#disp(X_train(1, :));
disp(cost);
#disp(X_train);