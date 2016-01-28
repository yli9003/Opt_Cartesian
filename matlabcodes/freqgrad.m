function [freq,grad,epsilon]=freqgrad(epsbar,epsdiff,epsbkg,Nx,Ny,dx,dy,mpbdata)
%Usage: this base function calculates the [eigenfreq, gradient array
%of freq, 2d epsilon matrix] calling MPB. You have to supply the
%binary array epsbar, scalars epsdiff and epsbkg as well as Nx, Ny,
%dx, dy. The information about the mpb call should be given in the
%datatype mpbdata.[mpbctlfile, mpbepsout, mpbEout, modenum, np]. Be
%sure to correctly specify the mode number (the code automatically
%adds +6 to specify the correct column number)

mpbctlfile = mpbdata.mpbctlfile;
mpbepsout = mpbdata.mpbepsout;
mpbEout = mpbdata.mpbEout;
modenum = mpbdata.modenum;
np = mpbdata.np;

epsilon=epsdiff*epsbar+epsbkg;
epsilon=reshape(epsilon,Ny,Nx);
hdf5write('tmp.h5','/eps',epsilon);

runmpb=['mpirun -np ',num2str(np),' mpb-mpi numbands=',num2str(modenum),' ',mpbctlfile,' | grep freqs | grep -v kmag | cut -d" " -f',num2str(modenum+6)];
%runmpb=['mpirun -np ',num2str(np),' mpb-mpi ',mpbctlfile,' | grep freqs | grep -v kmag | cut -d" " -f',num2str(modenum+6)];

[status,freq]=system(runmpb);

epsilon=hdf5read(mpbepsout,'/data');
E.x = hdf5read(mpbEout,'/x.r') + 1i * hdf5read(mpbEout,'/x.i');
E.y = hdf5read(mpbEout,'/y.r') + 1i * hdf5read(mpbEout,'/y.i');
E.z = hdf5read(mpbEout,'/z.r') + 1i * hdf5read(mpbEout,'/z.i');

dotE = conj(E.x) .* E.x + conj(E.y) .* E.y + conj(E.z) .* E.z;
norm = dx*dy* sum(sum(epsilon .* dotE ));
grad = -freq/2 * epsdiff * dotE /norm;
grad = grad(:);

cleanfiles=['rm -rf tmp.h5 ',mpbepsout,' ',mpbEout];
system(cleanfiles);

end