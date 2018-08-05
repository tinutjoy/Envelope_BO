function [Ef, Varf] =  predictGP(x_new,gp, x, a, invC)

Knx = gp_cov(gp,x_new,x);
Kn = gp_trvar(gp,x_new);
Ef = Knx*a; Ef=Ef(1:size(x_new,1));
invCKnxt = invC*Knx';
Varf = Kn - sum(Knx.*invCKnxt',2);
Varf=max(Varf(1:size(x_new,1)),0);


