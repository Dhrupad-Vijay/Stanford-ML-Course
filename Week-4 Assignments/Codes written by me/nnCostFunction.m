function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...               % (25x401) Unrolling nn_params to get Theta1 
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...     % (10x26) Unrolling nn_params to get Theta1 
                 num_labels, (hidden_layer_size + 1));

[m n] = size(X);                           % storing dimensions of X
X = [ones(m,1) X];                         % (5000x401) Updating X to get a biases

y_mat = [1:num_labels]' == y';             % (10x5000) Expand the 'y' output values into a matrix of single values. If using a for loop (slower and harder) logical arrays will do the job


%
%           Forward Propagation
%
a1 = X';                                            % (401x5000) assign the trials into the 1st layer of the network
z2 = Theta1*a1;                                     % (25x5000) = (25x401)*(401x5000) find z2 for 2nd layer
a2 = sigmoid(z2);                                   % (25x5000) sigmoid of entire z2, to find activation values for 2nd layer
a2 = [ones(1,m);a2];                                % (26x5000) add the bias nodes to 2nd layer
z3 = Theta2*a2;                                     % (10x5000) = (10x26)*(26x5000) find z3 for 3rd layer
hx = sigmoid(z3);                                   % (10x5000) sigmoid of entire z3, to find activation values for 3rd layer

%
%           Cost Function
%

J = sum((1/m) .* (-y_mat .* log(hx) - (1 - y_mat) .* (log(1 - hx))),'all') + (lambda / (2 * m)) * (sum(Theta1(:,2:end).^2,'all') + sum(Theta2(:,2:end).^2,'all'));

%
%           Backward Propagation
%
delta3 = hx - y_mat;                                % (10x5000) taking difference of predicted and actual results, delta value for the 3rd layer
z2 = [ones(1,m);z2];                                % (26x5000)
delta2 = Theta2' * delta3 .* sigmoidGradient(z2);   % (26x5000) = (26x10) * (10x5000) .* (26x5000) calculating delta value for the 2nd layer
delta2 = delta2(2:(hidden_layer_size+1),:);         % (25x5000) <- (26x5000)
Theta1_grad = delta2*a1';    % unregularized        % (25x401) = (25x5000) * (5000x401) Finding the gradient for all theta1 values by going through all the trials
Theta2_grad = delta3*a2';    % unregularized        % (10x26) = (10x5000) * (5000x26) Finding the gradient for all theta2 values by going through all the trials

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = (1/m) .* (Theta1_grad + lambda .* (Theta1));  % regularized gradient matrix
Theta2_grad = (1/m) .* (Theta2_grad + lambda .* (Theta2));  % regularized gradient matrix

grad = [Theta1_grad(:) ; Theta2_grad(:)];           % unroll the gradients into a Theta1_grad and Theta2_grad 10285x1 vector


end
