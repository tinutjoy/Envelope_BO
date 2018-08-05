function [ x_target,y_target,best_y ] = envelopeBo( x_target,y_target,fx,bounds, x_source,y_source,maxiter, acq_str, problem_type)
% Transfel learning with Envelope-BO
 
cfse = gpcf_sexp('lengthScale',1.4,'magnSigma2',0.2^2);
% cfse = gpcf_sexp('lengthScale',[5 5],'lengthScale_prior',prior_t('s2',4),'magnSigma2',.1,'magnSigma2_prior',prior_sqrtt('s2',10^2));
lik = lik_gaussian('sigma2', 0.001);
gp = gp_set('cf', cfse, 'lik', lik, 'jitterSigma2', 1e-9);
opt = optimset('TolX',1e-4,'TolFun',1e-4,'Display','iter');
gp_target=gp;
 
% fit a gp at the source points
gp_source = gp;
gp_source = gp_optim(gp_source,x_source,y_source,'opt',opt, 'optimf', @fminlbfgs);
 
 
 
%combine data
x_comb = [ x_source ;x_target];
y_comb = [y_source;y_target];
 
 
%acq optimise settings
dopt.maxevals = 2000;
dopt.maxits = 1000;
dopt.showits = 0;
 
% choosing the problem
if strcmp(problem_type,'min')
    incumbent = @(y) min(y); % target function
elseif strcmp(problem_type,'max')
    incumbent = @(y) max(y); % target function
end
 
% choosing the acquisition
if strcmp(acq_str,'EI')
    FLAG = 0; % target function
elseif strcmp(acq_str,'GPUCB')
    FLAG = 1; % target function
end
 
 
 
%initial set up
 
inc = incumbent( y_target );
 
%Intial source noise 
 
%Change this setting when relatedness between functions change
 
alpha_prior = 5;
beta_prior = 3;
source_noise_var = 0.2; 
 
ii = 1;
best_y = [];
z =[];
 
numSource_points = size(x_source,1);
 
while ii <= maxiter
    
    [invC, a, gp_target] = fitGPCombined(gp_target,x_comb,y_comb, source_noise_var, numSource_points);
    
    
    if FLAG == 0
        problem.f = @(x_test) expectedimprovement(x_test', gp_target, x_comb, a, invC, inc, problem_type);
    else
        problem.f = @(x_test) gpUCB(x_test', gp_target, x_comb, a, invC, problem_type, ii);
        
    end
    %x_max=unifrnd(bounds(:,1),bounds(:,2));
    [~, x_new, ~] = direct(problem,bounds, dopt);
    
    % put new sample point to the list of evaluation points
    x_target(end+1,:) = x_new';
    x_comb(end+1,:) = x_new';
    y_target(end+1,:) = fx(x_target(end,:));     % calculate the function value at query point
    y_comb(end+1,:) = y_target(end);
    
    %compute source noise
    source_noise_var = estimateSourceNoise(gp_source,x_source,y_source,x_target,y_target, alpha_prior,beta_prior);
 
    inc = incumbent(y_target);
    
    ii=ii+1;
    best_y = [best_y; inc];
    
end
 
 
 

