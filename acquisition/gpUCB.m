function [ GPUCB ] = gpUCB( x_new, gp, x, a, invC, problem_type, numEvals )
%   GPUCB acquisition function
if ~isempty(x)
    
    [Ef, Varf] = predictGP(x_new,gp, x, a, invC);
    
    numDims = size(x_new, 2); 
    
    delta = 0.01; % (remember probability of 1 - delta).
    
    t = numEvals;
    beta_t = 2 *log(t^2 * 2*pi^2/(3*delta)) + ...
        2*numDims*log(t^2 *numDims * 2 * sqrt(4*numDims/delta) );
    
    if strcmp(problem_type,'min')
        
        GPUCB = -Ef + (sqrt(beta_t) * sqrt(Varf));
    
    elseif strcmp(problem_type,'max')

        GPUCB = Ef + (sqrt(beta_t) * sqrt(Varf));

    end
else
    GPUCB = 1;
end
GPUCB = -GPUCB;

