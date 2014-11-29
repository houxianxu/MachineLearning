function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% Train the SVM
trainCs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
trainSigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

errors = zeros(64, 3);
n = 1;
n
for i = 1:size(trainCs, 2) 
	for j = 1:size(trainSigmas, 2)
		C = trainCs(i);
		sigma = trainSigmas(j);	
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		errorPredic = mean(double(predictions ~= yval));
		errors(n, 1) = C;
		errors(n, 2) = sigma;
		errors(n, 3) = errorPredic;
		n = n + 1;
	end
end
[maxVal, maxRowIndex] = min(errors(:, 3));
C = errors(maxRowIndex, 1);
sigma = errors(maxRowIndex, 2);







% =========================================================================

end
