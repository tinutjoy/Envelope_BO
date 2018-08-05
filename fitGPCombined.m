function  [invC, a, gp_target] = fitGPCombined(gp_target,x_comb,y_comb, source_noise_var, numSource_points)

%     if i1>1
%         gp_target = gp_optim(gp_target,x_comb,y_comb);
%         [gpia,pth,th]=gp_ia(gp_target,x_comb,y_comb);
%         gp_target = gp_unpak(gp_target,sum(bsxfun(@times,pth,th)));
%     end
[K, C] = gp_trcov(gp_target,x_comb);

% Adding source noise varaince to observations from source task
C(1:numSource_points,1:numSource_points) = C(1:numSource_points,1:numSource_points) + (source_noise_var.*eye(numSource_points));
[U,S,V] = svd(C);
invC = V * (S\U');
a = C\y_comb;