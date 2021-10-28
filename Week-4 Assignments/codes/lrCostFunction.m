% Computes the cost and gradient for regularized logistic regression without for loops

function [J, grad] = lrCostFunction(theta, X, y, lambda)

[m, n] = size(X);                                                                            % Assigns m and n values of the width and length of the dataset                                                                          % turns m into a vector. Needed to calculate the gradients
h = sigmoid(X * theta);                                                                      %Computing the sigmoid of the hypothesis  

J = (1/m).*((-y')*log(h) - ((1-y)')*log(1-h)) + (lambda/(2*m))*(theta(2:n)'*theta(2:n));    %Calculates cost (with regularized theta values, except theta(1))

grad = (1/m)*(X'*(h-y));
temp = theta; temp(1) = 0;
grad = grad + (lambda/m).*temp;                                             %Calculates gradients. We skip theta(1), because we add it artificially

grad = grad(:);                                                             %This line guarantees that the grad value is returned as a column vector.                                     

end
