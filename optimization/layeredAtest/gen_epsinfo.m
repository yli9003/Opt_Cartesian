signal.n=3;
signal.nsub=1.45;
signal.nair=1.0;

idler.n=3;
idler.nsub=1.45;
idler.nair=1.0;

pump.n=3;
pump.nsub=1.45;
pump.nair=1.0;

eps1=idler;
eps2=pump;
eps3=signal;

nlayers=3;
!rm -rf epsinfo.txt

for i=1:nlayers,

    if i<nlayers,
        system(['echo "-eps1diff[',num2str(i),'] ',num2str(eps1.n^2-eps1.nsub^2),' ">> epsinfo.txt']);
        system(['echo "-eps1bkg[',num2str(i),'] ',num2str(eps1.nsub^2),' ">> epsinfo.txt']);

        system(['echo "-eps2diff[',num2str(i),'] ',num2str(eps2.n^2-eps2.nsub^2),' ">> epsinfo.txt']);
        system(['echo "-eps2bkg[',num2str(i),'] ',num2str(eps2.nsub^2),' ">> epsinfo.txt']);

        system(['echo "-eps3diff[',num2str(i),'] ',num2str(eps3.n^2-eps3.nsub^2),' ">> epsinfo.txt']);
        system(['echo "-eps3bkg[',num2str(i),'] ',num2str(eps3.nsub^2),' ">> epsinfo.txt']);
    else
        system(['echo "-eps1diff[',num2str(i),'] ',num2str(eps1.n^2-eps1.nair^2),' ">> epsinfo.txt']);
        system(['echo "-eps1bkg[',num2str(i),'] ',num2str(eps1.nair^2),' ">> epsinfo.txt']);

        system(['echo "-eps2diff[',num2str(i),'] ',num2str(eps2.n^2-eps2.nair^2),' ">> epsinfo.txt']);
        system(['echo "-eps2bkg[',num2str(i),'] ',num2str(eps2.nair^2),' ">> epsinfo.txt']);    

        system(['echo "-eps3diff[',num2str(i),'] ',num2str(eps3.n^2-eps3.nair^2),' ">> epsinfo.txt']);
        system(['echo "-eps3bkg[',num2str(i),'] ',num2str(eps3.nair^2),' ">> epsinfo.txt']);
    end

end

system(['echo "-eps1sub ',num2str(eps1.nsub^2),' ">> epsinfo.txt']);
system(['echo "-eps1air ',num2str(eps1.nair^2),' ">> epsinfo.txt']);
system(['echo "-eps1mid ',num2str(eps1.nsub^2),' ">> epsinfo.txt']);
!echo "-eps1subdiff 0" >> epsinfo.txt
!echo "-eps1airdiff 0" >> epsinfo.txt
!echo "-eps1middiff 0" >> epsinfo.txt

system(['echo "-eps2sub ',num2str(eps2.nsub^2),' ">> epsinfo.txt']);
system(['echo "-eps2air ',num2str(eps2.nair^2),' ">> epsinfo.txt']);
system(['echo "-eps2mid ',num2str(eps2.nsub^2),' ">> epsinfo.txt']);
!echo "-eps2subdiff 0" >> epsinfo.txt
!echo "-eps2airdiff 0" >> epsinfo.txt
!echo "-eps2middiff 0" >> epsinfo.txt
       
system(['echo "-eps3sub ',num2str(eps3.nsub^2),' ">> epsinfo.txt']);
system(['echo "-eps3air ',num2str(eps3.nair^2),' ">> epsinfo.txt']);
system(['echo "-eps3mid ',num2str(eps3.nsub^2),' ">> epsinfo.txt']);
!echo "-eps3subdiff 0" >> epsinfo.txt
!echo "-eps3airdiff 0" >> epsinfo.txt
!echo "-eps3middiff 0" >> epsinfo.txt

!grep eps1 >> tmp
!mv tmp epsinfo.txt
%replace eps1 with eps in emacs