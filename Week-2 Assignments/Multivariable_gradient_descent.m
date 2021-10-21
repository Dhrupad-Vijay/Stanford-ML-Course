
data = load('ex1data2.txt');
X1 = data(:, 1:2);
y1 = data(:, 3);
m1 = length(y1);
%tht = zeros(size(X1,2)+1,1);

function [mu1, sigma2, X] = normal(X1)

mu1 = mean(X);
X = X - ones(m1,1)*mu1';
sigma2 = std(X);
X = X./sigma;
X = [ones(m1,1) X];

end

function jcost = cost(X,y,tht)

temp = sum((X*tht - y).^2);
jcost = (1/(2*m1))*(temp);

end

function [tht,J_rest] = graddec(X, y, theta, alpha, num_iters)

j = 0;
for i = 1:num_iters

err = (X*tht - y)
tht = tht - (alpha/m1).*(X' * err);

end
end

