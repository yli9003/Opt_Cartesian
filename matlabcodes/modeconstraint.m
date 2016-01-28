function [freqdiffsq,optgrad]=modeconstraint(dof,freqref,epsdata,Ndata,mpbdata,counter)

epsbar=dof(1:end-1);
decisionvar=dof(end);

epsdiff=epsdata.epsdiff;
epsbkg=epsdata.epsbkg;
Nx=Ndata.Nx;
Ny=Ndata.Ny;
dx=Ndata.dx;
dy=Ndata.dy;
[freq,grad,epsilon]=freqgrad(epsbar,epsdiff,epsbkg,Nx,Ny,dx,dy,mpbdata);

freqdiffsq=(freq-freqref)^2 - decisionvar;
optgrad=[2*(freq-freqref)*grad;-1];

tmp=counter.count;
counter.count=tmp+1;
counter.epsilon=epsilon;
printout(counter);

end