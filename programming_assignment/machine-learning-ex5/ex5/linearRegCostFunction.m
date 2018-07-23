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

H = X*theta;

%number of parameters
n = length(theta);

%the first parameter is not considered i.e., theta(1)
I = eye(n,n);
I(1,1) = 0;
CP = I*theta;

J = (1/(2*m)).*((H-y)'*(H-y)) + (lambda/(2*m))*(CP'*CP);

%calculating the gradient

grad = (1/m)*(X'*(H-y)) + (lambda/m)*(CP);

% =========================================================================

grad = grad(:);

end
