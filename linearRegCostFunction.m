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
%H = sigmoid(X*theta);
%T = y.*log(H) + (1 - y).*log(1 - H);
thetaf = theta(2:end).^2;
thetag = theta;
thetag(1,1)=0;


m = size(X,1); % number of training examples
predictions = X * theta;
sumerror = X'*(predictions - y);
sqrErrors = (predictions - y).^2;
J2 = 1 / (2 * m) * sum(sqrErrors);

J1 = lambda / (2 * m)*sum(thetaf) ;
J = J1 + J2;




grad1 = sumerror / ( m) ;
grad2 = lambda*thetag / m;

%

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	


    % ============================================================

    % Save the cost J in every iteration   
% =========================================================================

gradf = grad1 + grad2;
grad = gradf(:)

end

