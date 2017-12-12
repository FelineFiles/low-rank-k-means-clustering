function [L, L_grad] = lagrangian( K, Y, lambda_c1, lambda_c2, nu, t, k)
%LANGRANGIAN computes the lagrangian of low ranked k-means
% Precondition: K is the kernel matrix of the data, Y is the current primal
% variable, lambda_c1 is the current dual variable for constraint 1, 
% lambda_c2 is the current dual variable for constraint 2, and nu is the 
% current dual variable for the equality constraint
% Postcondition: returns the lagrangian of our k-means algo

% Some helper variables
n = size(K, 1); % Get the number of data points
wun = ones(n, 1);
Id = eye(n);
Z = Y*Y';

L = -sum(sum(K'.*Z)); % Contribution from objective

% From constraint 1
temp = trace(Z)-k;
L = L + lambda_c1 * temp + t/2 * temp^2; 

% From constraint 2
for i=1:n
    temp = Y(i,:)*sum(Y)'-1;
    L = L + lambda_c2(i)*temp + t/2 * temp^2;
end;

% From constraint 3
L = L + sum(sum( max(0, nu-t*Y).^2 - nu.^2 ))/(2*t);

% If gradient is required
if nargout > 1
    % From objective 
    L_grad = -2*K*Y;
    
    % From constraint 1 
    L_grad = L_grad + 2*(lambda_c1 + t*(sum(sum(Y.*Y))-k))*Y;
    
    % From constraint 2
    for i=1:n
        grad_help = zeros(n);
        grad_help(i,:) = wun/2;
        grad_help(:,i) = wun/2;
        grad_help(i,i) = 1;
        L_grad=L_grad+2*(lambda_c2(i)+t*(Y(i,:)*sum(Y)'-1)) ...
            * grad_help*Y;
    end;
    
    % From constraint 3
    L_grad = L_grad + -max(0, nu-t*Y);
end;

end

