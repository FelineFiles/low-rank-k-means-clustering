function [idx, C, sumd, D] = low_rank_k_means( X, k )
%LOW_RANK_K_MEANS runs low-rank k-means clustering on X
% Precondition: X is a n-by-p matrix where the rows are the input data, k 
% is the number of desired clusters.
% Postcondition: Perform a low-rank k-means cluster on X. Return 
% idx a n-by-1 vector containing cluster indices of each observations. 
% C is a k-by-p vector containing the centroid of each cluster in the rows.
% sumd ia the k-by-1 vector that stores the within cluster sums of point to
% centroid distance. D is a n-by-k matrix that stores the distance from
% each point to every centroid.

% Helper variables
[n, p] = size(X); % Get the number of data points
wun = ones(n, 1);
Id = eye(n);

% Step 0: Initialize the needed vectors and matrices
K = X*X'; % Get the Gram Matrix
Y = rand(n,k); % Y is our primal variable
lambda_c1 = 1; % The dual variable for the constraint |Y|_F^2 = k
lambda_c2 = -ones(n, 1); % The dual variable for the constraint YY'1 = 1
nu = ones(n, k); % The dual variable for -Y<=0 elementwise
tol = 10^-5; % Our tolerance
num_ite = 0; % Number of iterations
t = 1.5; % Step Size
MAX_ITE = 1000; % Maximum number of iterations. 1000 should be more than enough

% We will hot start our Y using regular k-means
temp = kmeans(X, k);
Y = zeros(n,k);
Y(sub2ind(size(Y), (1:n)', temp)) = 1;
Y = Y/(norm(Y, 'fro')/sqrt(k));
Y_temp = Y;

% Initialize our infeasibility
infeasibility = (k-norm(Y, 'fro')^2)^2 + norm(wun-Y*(Y'*wun))^2 ...
        + norm(max(0,-Y), 'fro')^2;
L_min = 0;

% Next starts our iterate method
last_obj = Inf;
for num_ite = 1:MAX_ITE
    % Primal update. We will Matlab's use BFGS.
    L = @(input) lagrangian( K, input, lambda_c1, lambda_c2, nu, t, k);
    options = optimoptions('fminunc','Algorithm', 'quasi-newton', 'GradObj','on', 'TolFun', 0.0004, 'MaxIter', 1000);
    [Y, L_min, ~, ~, grad] = fminunc(L, Y, options);
    Z = Y*Y';
    
    % Next, dual update. Begin by computing the amount of infeasibility
    curr_infeasibility = (k-norm(Y, 'fro')^2)^2 + norm(wun-Y*(Y'*wun))^2 ...
        + norm(max(0,-Y), 'fro')^2;
    % If current infeasibility is small, update the duals
    if (curr_infeasibility < 0.25*infeasibility)
        % First constraint
        lambda_c1 = lambda_c1 + t*(trace(Z)-k);
        % Second constraint
        for i=1:n
            lambda_c2(i) = lambda_c2(i) + t*(trace((Id(i, :)*Z) * wun)-1);
        end;
        % Third constraint
        nu = max(nu-t*Y, 0);
        
        infeasibility = curr_infeasibility;
        t = t;
    else 
        % Else the infeasibilit is big, so we increase the penalty
        % parameter
        t = sqrt(10)*t;
    end;
    
    % Define some helper variables
    L_min = lagrangian( K, Y, lambda_c1, lambda_c2, nu, t, k);
    curr_obj = -trace(K.^Z);
    cons_error = abs(last_obj-curr_obj)/max(1, abs(curr_obj));
    rel_error = abs(L_min-curr_obj)/max(1, abs(L_min));
    
    % Debug info
    fprintf('Iteration Number: %i\n', num_ite);
    fprintf('Constraint 1 Error: %d\n', (k-norm(Y, 'fro')^2)^2);
    fprintf('Constraint 2 Error: %d\n', norm(wun-Y*Y'*wun)^2);
    fprintf('Constraint 3 Error: %d\n', norm(max(0,-Y), 'fro')^2);
    fprintf('t: %d\n', t);
    fprintf('Objective Value: %d\n', curr_obj);
    fprintf('Lagrangian Value: %d\n', L_min);
    fprintf('Relative Error: %d\n', rel_error);
    fprintf('Consecutive Error: %d\n', cons_error);
    
    % Terminating condition
    if rel_error < tol
        fprintf('Difference between Lagrangian and objective is small\n');
        break;
    elseif cons_error < tol/100
        fprintf('Difference between consecutive objective is small\n');
        break;
    end;
    
    % Update the last objective value
    last_obj = curr_obj;
end;

% Post Processing
[~, idx] = max(Y');
idx = idx';
C = zeros(k, p);
for i=1:k
    class = idx==i;
    C(i,:) = mean(X(class,:));
end;
sumd = zeros(k,1);
D = zeros(n,k);
for i=1:n
    for j=1:k
        D(i,j) = norm(X(i,:)-C(j,:)); 
        if idx(i)==j
            sumd(j) = sumd(j) + norm(X(i,:)-C(j,:));
        end;
    end;
    sumd(i) = norm((X(class,:)-repmat(C(k,:), sum(class),1)), 'fro');
end;
