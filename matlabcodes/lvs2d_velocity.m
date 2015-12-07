function [Vn,nvec]=lvs2d_velocity(E,Eadj,epsdiff,epsbkg,epsbar,dx,dy)
%Usage supply the vector fields E and Eadj in the form E.x, E.y and
%E.z where each is a matrix where columns represent x coordinates
%and rows represent y coordinates. Same thing for epsbar except
%that it is a scalar field.
%this means for a matrix A with [Ny x Nx], the 1d form is A=A(:)
%with indices where all the y coords come first for each x coord and to
%revert to the original form do A = reshape(A,Ny,Nx)

[nvecx,nvecy]=gradient(epsbar,dx,dy);
[Ny,Nx]=size(nvecx);
for i=1:Nx,
   for j=1:Ny,
      tmp=nvecx(j,i)^2 + nvecy(j,i)^2;
      if tmp > 0,
         nvecx(j,i)=nvecx(j,i)/sqrt(tmp);
         nvecy(j,i)=nvecy(j,i)/sqrt(tmp);
      end
   end
end
boundary=nvecx;
index=(nvecy ~= 0);
boundary(index)=nvecy(index);
boundary=logical(boundary);

Eperpen.x = ( nvecx.*E.x + nvecy.*E.y ) .* nvecx;
Eperpen.y = ( nvecx.*E.x + nvecy.*E.y ) .* nvecy;
Eadjperpen.x = ( nvecx.*Eadj.x + nvecy.*Eadj.y ) .* nvecx;
Eadjperpen.y = ( nvecx.*Eadj.x + nvecy.*Eadj.y ) .* nvecy;

Eparallel.x = E.x - Eperpen.x;
Eparallel.y = E.y - Eperpen.y;
Eadjparallel.x = Eadj.x - Eadjperpen.x;
Eadjparallel.y = Eadj.y - Eadjperpen.y;

epsilon=epsdiff*epsbar + epsbkg;
Dperpen.x = epsilon .* Eperpen.x;
Dperpen.y = epsilon .* Eperpen.y;
adjDperpen.x = epsilon .* Eadjperpen.x;
adjDperpen.y = epsilon .* Eadjperpen.y;

VnE = epsdiff * ( Eparallel.x .* Eadjparallel.x + Eparallel.y .* Eadjparallel.y );
eps1=epsbkg;
eps2=epsbkg + epsdiff;
VnD = (1/eps1 - 1/eps2) * ( Dperpen.x .* adjDperpen.x + Dperpen.y .* adjDperpen.y );

Vn = real(VnE + VnD);
Vn = boundary .* Vn;
Vnavg=sum(sum(Vn))/sum(sum(boundary));
Vn((boundary~=0)) = Vn((boundary~=0)) - Vnavg;
nvec.x=nvecx;
nvec.y=nvecy;
end