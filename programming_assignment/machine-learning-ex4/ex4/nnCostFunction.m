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
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%% Feedforward propagation
a1 = [ones(m,1),X]';

z2 = Theta1*a1;
a2 = sigmoid(z2);

a2 = [ones(1,m);a2];

z3 = Theta2*a2;
a3 = sigmoid(z3);

% creates a num2str x m matrix where each row corresponds to the y(i) values
% y(i) is a num_labels X 1 vector where only the correct class has a value of 1
Y = zeros(num_labels,m);
for i = 1:m
  y_i = [1:num_labels]';  
  Y(:,i) = y_i==y(i);
end

%cost function
% without regularization
  % the inner sum() takes a sum over the columns i.e., over all the cost for each predicted class in each training example sperately
  % the outer sum() sums up over the cost over all the rows i.e., over all training examples. 

%with regularization: add sum(sum(Theta1(:,2:size(Theta1,2)).^2)) gives us the total sum of all the squared parameters values in Theta1
%excluding the parameters associated with the bias. i.e., when j = 0.    
J = (-1/m)*sum(sum((Y.*log(a3))+(1.-Y).*log(1.-a3))) + (lambda/(2*m))*(sum(sum(Theta1(:,2:size(Theta1,2)).^2)) + sum(sum(Theta2(:,2:size(Theta2,2)).^2)));


%backpropagation

delta_3 = a3 - Y; %num_labels x m matrix

% z2 is  hidden_layer_size x m matrix but (Theta2'*delta_3) is a (hidden_layer_size + 1) x m 
% this is because z2 does not contain the "bias"
% we therefore have to add a row of numbers called "dummy" to z2 
% we know that sigmoid(z2^i_0) for the bias == 1, so sigmoid(dummy) == 1
% this is removed after the calculation phase, hence it is not important what number we add, just that it's sigmoid == 1
% the sigmoidGradient() of the added row is 0 as 
  %sigmoid(dummy)(1 - sigmoid(dummy)) = 1 
  %as sigmoid(dummy) = 1
  % sigmoid(dummy)(1 - sigmoid(dummy)) = 0

dummy = 200; 
z2_temp = [ones(1,m)*dummy;z2];
delta_2 = (Theta2'*delta_3).*sigmoidGradient(z2_temp); %(hidden_layer_size + 1) x m matrix

%remove delta_2(0) as the error in the bias is not propogated back
delta_2 = delta_2(2:size(delta_2,1),:); % hidden_layer_size x m matrix


%Delta_2_ij = Delta_2_ij + delta_3_i * a_j for all i and j 
% gives errors for each unit in layer l in terms of the activation function output in layer l x errors in layer l+1

% Delta_2 =  num_labels x (hidden_layer_size + 1) matrix where an element at position (i,j) gives us the sum over m training examples 
% of the errors propagated to unit "i" of layer (2+1) from unit "j" in layer 2 
Delta_2 = (delta_3*a2');

%the Delta term for layer 1
%similar to Delta_2
Delta_1 = (delta_2*a1');

%the gradients
Theta1_grad = Delta_1 ./ m;
Theta2_grad = Delta_2 ./ m;

%grads with regularization
Theta1_grad = Theta1_grad + (lambda/m).*[zeros(size(Theta1,1),1),Theta1(:,2:size(Theta1,2))];
Theta2_grad = Theta2_grad + (lambda/m).*[zeros(size(Theta2,1),1),Theta2(:,2:size(Theta2,2))];
 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end