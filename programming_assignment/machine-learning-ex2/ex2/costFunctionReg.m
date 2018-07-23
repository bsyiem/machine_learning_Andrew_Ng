function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%n is the number of parameters 
n = length(theta);

%H represents all the h(x) values in a vector
H = sigmoid(X*theta);

%Penalizing parameters
penalizing_params = theta(2:n);

%calculates the added regularized part
R = (lambda/(2*m))*(penalizing_params'*penalizing_params);

%Calculates the cost 
J = (-(1/m)*sum((y.*log(H))+(1-y).*log(1-H))) + R;



%calulating the gradient

% I is used to remove the first term from the theta parameter i.e., theta(1) as it is not penalized
% this method allows us to vectorize the calculation of the gradients.
I = eye(n,n);
I(1,1) = 0;



grad = (1/m)*(X'*(H-y)) + (lambda/m)*(I*theta);


% =============================================================

end
