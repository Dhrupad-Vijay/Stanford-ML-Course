% PREDICT Predict the label of an input given a trained neural network
% p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
% trained weights of a neural network (Theta1, Theta2) 
% Feedforward propagation

function p = predict(Theta1, Theta2, X)

[m, n] = size(X);                           % Record the X matrix dimensions
X = [ones(m,1) X];                          % Add the usual ones column

a2 = sigmoid(Theta1 * X');                  % Calculate activation values of 2nd hidden layer
a2 = [ones(1,m);a2];                        % Add ones row to a2 (25 x 5000 -> 26 x 5000) to act as input to next layer 

a3 = sigmoid(Theta2 * a2);                  % Calculate activation values of the 3rd hidden layer

[~,p] = max(a3,[],1);                       % Find the max element in each 10 element column and store the position value
p = p(:);                                   % Convert p into a column vector
end
