function x = generateSamples(numSamples, lb,ub)
   
%
%   input: lb - lower bound
%          ub - upper bound
%   output:x -  sampels generated


nDim = length(ub);

x = zeros(numSamples, nDim);

for ii = 1:numSamples       
    
    x(ii,:) = lb' + (ub' - lb') .* rand(1,nDim);    
end

end