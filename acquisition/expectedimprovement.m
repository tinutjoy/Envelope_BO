function EI = expectedimprovement(x_new, gp, x, a, invC, inc, problem_type)


% Calculate expected improvement if there are training points for it
if ~isempty(x)
    [Ef, Varf] = predictGP(x_new,gp, x, a, invC);

    
    % expected improvement
    if strcmp(problem_type,'min')
        
        posvar=find(Varf>0);
        CDFpart = zeros(size(Varf));
        PDFpart = zeros(size(Varf));
        tmp = ( inc - Ef(posvar))./sqrt(Varf(posvar));
        CDFpart(posvar) = normcdf(tmp);
        CDFpart(~posvar) = (inc - Ef(~posvar))>0;
        PDFpart(posvar) = normpdf(tmp);
        EI =  ( inc - Ef ).*CDFpart + sqrt(Varf).*PDFpart;
    elseif strcmp(problem_type,'max')
        posvar=find(Varf>0);
        CDFpart = zeros(size(Varf));
        PDFpart = zeros(size(Varf));
        tmp = (  Ef(posvar) - inc)./sqrt(Varf(posvar));
        CDFpart(posvar) = normcdf(tmp);
        CDFpart(~posvar) = (Ef(~posvar)- inc)>0;
        PDFpart(posvar) = normpdf(tmp);
        EI =  (  Ef - inc).*CDFpart + sqrt(Varf).*PDFpart;
    end 
else
    EI = 1;
end
EI = -EI;
