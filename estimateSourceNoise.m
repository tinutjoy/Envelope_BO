
function  source_noise_var = estimateSourceNoise(gp_source,x_source,y_source,x_target, y_target,alpha_prior,beta_prior)

[ySource_predcited] = gp_pred(gp_source,x_source,y_source,x_target);
alpha = alpha_prior+(length(y_target)/2);
diff_sq = sum((y_target-ySource_predcited).^2);
beta = beta_prior + (diff_sq/2);
source_noise_var = beta/(alpha+1);
