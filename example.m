%% Load Data
load fisheriris
X = meas(:, 3:4);
k=3;

%% Classify using low rank k-means
labels = low_rank_k_means(X, k);

%% Plot
figure(1);
class = labels==1;
plot(X(class, 1), X(class,2), 'ro');
hold on;
class = labels==2;
plot(X(class, 1), X(class,2), 'bo');
class = labels==3;
plot(X(class, 1), X(class,2), 'ko');
class = labels==4;
plot(X(class, 1), X(class,2), 'go');
class = labels==5;
plot(X(class, 1), X(class,2), 'mo');

title('Low-rank k-means');
