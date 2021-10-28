% Predict the label for trained one-vs-all classifier/s. The labels are in the range 1..K, where K = size(all_theta, 1).
% p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions for each example in the matrix X.
% all_theta is a matrix where the i-th row is a trained logistic regression theta vector for the i-th class.

function p = predictOneVsAll(all_theta, X)

[m,n] = size(X);                                        % Recording the size of X
X = [ones(m,1), X];                                     % Adding the usual ones row
                                
h = sigmoid(X * all_theta');                            % Matrix of h(x) = g(z) ; matris is m x num_lables

[~,p] = max(h,[],2);                                    % Finding the max prob. in each row and storing the position in 'p'

end
