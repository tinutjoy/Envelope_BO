
clear
close all

% Data Set Up
load mnist_uint8;
cast = @double;

train_x = cast(train_x) / 255;
test_x  = cast(test_x)  / 255;
train_y = cast(train_y);
test_y  = cast(test_y);

normalize
[train_x, mu, sigma]    = zscore(train_x);
test_x                  = normalize(test_x, mu, sigma);

Source_Percentage = 0.4; % percentage of source data

train_source_idx = round(Source_Percentage * size(train_x,1));
train_source_x = train_x(1:train_source_idx,:);
train_source_y = train_y(1:train_source_idx,:);

% define the objective function
fx = @(x) mlp_tuning(x,train_x,train_y,test_x,test_y); % target function
fx_source = @(x) mlp_tuning(x,train_source_x,train_source_y,test_x,test_y); %source function

% define the problem
problem_type = 'min'; % min: minimization | max: maximization

% select acquisition function
acq = 'GPUCB'; % options - EI: Expected improvement | GPUCB: GP-UCB 

% define problem space
lb = [100 -3 -3 0.1 100 1];     % lower bound of the input space
ub = [800 0 0 0.7 1000 3];   % upper bound of the input space
bounds = [lb' ub'];



min_init_target_points = size(bounds,1); % intial target points - setting it to the number of parameters
source_points = 30; %source points for transfer learning

%Generate source points

x_source = generateSamples(source_points, lb,ub);

y_source=[];

for tt=1:source_points
    
    y_source = [y_source;fx_source(x_source(tt,:))];
end




%setting for BO
maxiter = 30; %Number of recommendations
numiter = 10; % Number of runs with diff. intitialisations

best_y_mat = [];

ii = 1;
while ii <= numiter
    
    x_init = generateSamples(min_init_target_points, lb,ub); % intial target points
    y_init = [];
    for tt = 1:size(x_init,1)
        y_init = [y_init;fx(x_init(tt,:))];
    end
    
    [ x_recom,y_recom,best_y ] = envelopeBo( x_init,y_init,fx, bounds, x_source,y_source,maxiter, acq, problem_type);
    best_y_mat = [best_y_mat;  reshape(best_y,[1,length(best_y)])];
    
    ii = ii + 1;
    
end


bins = 1:maxiter;

best_y_average = mean(best_y_mat);
best_y_std_error = std(best_y_mat)./ sqrt(numiter);

errorbar(bins, best_y_average, best_y_std_error);
xlabel('Number of iterations');
ylabel('Test Error');
