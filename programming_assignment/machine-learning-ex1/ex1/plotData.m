function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

figure; % open a new figure window

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the training data into a figure using the 
%               "figure" and "plot" commands. Set the axes labels using
%               the "xlabel" and "ylabel" commands. Assume the 
%               population and revenue data have been passed in
%               as the x and y arguments of this function.
%
% Hint: You can use the 'rx' option with plot to have the markers
%       appear as red crosses. Furthermore, you can make the
%       markers larger by using plot(..., 'rx', 'MarkerSize', 10);


plot(x,y,"xr");  %plot the y versus x with "x" markers and using the color "r"ed
xlabel('Population of City in 10,000s');  %setting xlabel
ylabel('Profit in $10,000s');   %setting ylabel
axis([4 24 -5 25]);  %setting the axis ranges of x and y

%setting step size for the x axis
xbounds = xlim(); 
set(gca,'xtick',xbounds(1):2:xbounds(2));   

%setting step size for the y axis
ybounds = ylim(); 
set(gca,'ytick',ybounds(1):5:ybounds(2));   


% ============================================================

end
