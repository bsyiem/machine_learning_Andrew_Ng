function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



%size(Theta1) = [25, n+1 ]       ----- +1 because of bias 
%size(Theta2) = [num_labels, 25+1]    ----- +1 because of bias

% Theta1 * a1 gives us a (25 * m) matrix where
% each column k corresponds to the z^(2)_rowNum in the next layer  
% z^j_i =  Theta1_10*x^k_0 + Theta1_11*x^k_1 .... + Theta1_1n*x^k_n 
% where k = the k^th training data point 


% X = m * n matrix 
% Theta1 = 25 * n + 1
% we need to add the x_0 = 1 elements to each row of X to turn it to a m * (n+1) matrix
% a1 = X' after adding x_0 to each row, making it a (n+1)*m matrix
a1 = [ones(m,1),X]';

%z2 =  (25 * m) matrix
z2 = Theta1 * a1;
a2 = sigmoid(z2);

% add bias unit to layer 2
% a2 = (25+1 * m) matrix
a2 = [ones(1,m);a2];

%z3 =  (10 * m) matrix
z3 = Theta2 * a2;
a3 = sigmoid(z3);

%take column wise maximum and return the [max,index]
[ignored,p_tentative] = max(a3,[],1);
p = p_tentative(:);

% =========================================================================


end
