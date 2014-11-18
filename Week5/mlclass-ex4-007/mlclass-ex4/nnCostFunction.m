function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % [25, 401]

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % [10, 26]

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% add ones to X
a1 = [ones(m, 1) X]; % [5000, 401]
z2 = a1 * Theta1'; % [5000, 25]
a2 = sigmoid(z2); % [5000, 25]
a2 = [ones(m, 1) a2]; % [5000, 26]
h = sigmoid(a2 * Theta2'); %[5000, 10]
h = h';
% transfer y
uniY = unique(y)';

Y = zeros(m, num_labels);

for i = 1:m
	Y(i, :) = (y(i) == uniY);
end
Y = Y'; %[10, 5000]

cosPositive = - Y .* log(h);
cosNegtive = (1-Y) .* log(1 - h);
cost = cosPositive - cosNegtive;

J = (1 / m) * sum(cost(:));

Theta1Tmp = Theta1(:, 2:end);
Theta2Tmp = Theta2(:, 2:end);
reg = lambda / (2 * m) * (sumsqr(Theta1Tmp) + sumsqr(Theta2Tmp));

J = J + reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
Delta1 = 0; % matrix associated with Theta1
Delta2 = 0; % matrix associated with Theta2

for t = 1:m
	a1 = [1; X(t, :)'];
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1; a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	yt = Y(:, t);
	d3 = a3 - yt;

	d2 = (Theta2Tmp' * d3) .* sigmoidGradient(z2);

	Delta1 = Delta1 + (d2 * a1');
	Delta2 = Delta2 + (d3 * a2');
end

Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1Tmp;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2Tmp;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
