!./test_exec -nlayers 3 -Nzo[1] 60 -Nzo[2] 80 -Nzo[3] 100 -options_file epsinfo.txt | tee output.dat

!grep -v "%" epsF.m | grep -v "=" | grep -v ";" > epsF.txt
!grep -v "%" epsDiff.m | grep -v "=" | grep -v ";" > epsDiff.txt
!grep -v "%" epsBkg.m | grep -v "=" | grep -v ";" > epsBkg.txt
!grep -v "%" epsFull.m | grep -v "=" | grep -v ";" > epsFull.txt
!rm -rf epsF.m epsDiff.m epsBkg.m epsFull.m *.h5 *.png

vectoh5('epsF.txt','epsF.h5',150,1,250);
vectoh5('epsDiff.txt','epsDiff.h5',150,1,250);
vectoh5('epsBkg.txt','epsBkg.h5',150,1,250);
vectoh5('epsFull.txt','epsFull.h5',150,1,250);

!h5topng epsF.h5:Fxr
!h5topng epsDiff.h5:Fxr
!h5topng epsBkg.h5:Fxr
!h5topng epsFull.h5:Fxr
