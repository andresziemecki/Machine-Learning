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

for k=1:length(theta)
    for i=1:m
        grad(k) = grad(k) + (sigmoid(X(i,:)*theta)-y(i))*X(i,k);
        if (k>1)
            grad(k) = grad(k) + theta(k)*lambda/m;
        end
    end
end
grad=1/m*grad;


Sumatoria = 0;

for i=1:m
    Sumatoria = Sumatoria - y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta));
end

theta(1) = 0;
J = (1/m)*(Sumatoria + (lambda/2)*sum(theta.*theta));

    




% =============================================================

end
