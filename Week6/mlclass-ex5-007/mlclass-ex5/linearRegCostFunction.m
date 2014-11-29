function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;
z1 = h - y;
z = z1 .^ 2;

cost1 = (1 / (2 * m)) * (ones(1, m) * z);
regTheta = theta(2:end);
costReg = (lambda / (2 * m)) * sum(regTheta .^ 2);

J = cost1 + costReg;

% compute Grad
grad1 = (1 / m) * (z1' * X(:, 1));
gradOther = (1 / m) * (z1' * X(:, 2:end))' + (lambda / m) * regTheta;

grad = [grad1; gradOther]; 




% =========================================================================

grad = grad(:);

end
