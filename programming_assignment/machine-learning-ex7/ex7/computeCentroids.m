function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

temp = [];
relevant_points = [];

for i = 1:K
  %temp stores 1 for all rows wher idx == i and 0 otherwise
  temp = (idx == i);
  %relevant_points stores x(j) where c(j) == i
  relevant_points = X .* temp; 
  
  %sum(temp) returns the number of points that are assigned to cluster i as temp only stores 0 or 1
  centroids(i,:) =  sum(relevant_points)/sum(temp);
end
% =============================================================


end

