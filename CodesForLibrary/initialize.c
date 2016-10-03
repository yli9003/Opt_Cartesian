#include <petsc.h>
#include <time.h>
#include "libOPT.h"
#include <complex.h>
#include "petsctime.h"

#undef __FUNCT__ 
#define __FUNCT__ "readfromflags"
PetscErrorCode readfromflags(Universals *params)
{
  
  PetscBool flg;
  
  int Mx,My,Mz,Lz,Mzslab,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,Nxyz,DegFree;
  double hx,hy,hz,hxyz;
  int BCPeriod,LowerPMLx,LowerPMLy,LowerPMLz,Nxo,Nyo,Nzo;
  double Qabs;
  char initialdatafile[PETSC_MAX_PATH_LEN], filenameComm[PETSC_MAX_PATH_LEN];
  double sigmax,sigmay,sigmaz;
  int outputbase,readlubsfromfile;

  PetscOptionsGetInt(PETSC_NULL,"-Mx",&Mx,&flg);  MyCheckAndOutputInt(flg,Mx,"Mx","Mx");
  PetscOptionsGetInt(PETSC_NULL,"-My",&My,&flg);  MyCheckAndOutputInt(flg,My,"My","My");
  PetscOptionsGetInt(PETSC_NULL,"-Mz",&Mz,&flg);  MyCheckAndOutputInt(flg,Mz,"Mz","Mz");
  PetscOptionsGetInt(PETSC_NULL,"-Mzslab",&Mzslab,&flg);  MyCheckAndOutputInt(flg,Mzslab,"Mzslab","Mzslab");
  PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg);  MyCheckAndOutputInt(flg,Nx,"Nx","Nx");
  PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg);  MyCheckAndOutputInt(flg,Ny,"Ny","Ny");
  PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg);  MyCheckAndOutputInt(flg,Nz,"Nz","Nz");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlx",&Npmlx,&flg);  MyCheckAndOutputInt(flg,Npmlx,"Npmlx","Npmlx");
  PetscOptionsGetInt(PETSC_NULL,"-Npmly",&Npmly,&flg);  MyCheckAndOutputInt(flg,Npmly,"Npmly","Npmly");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlz",&Npmlz,&flg);  MyCheckAndOutputInt(flg,Npmlz,"Npmlz","Npmlz");
  getint("-Lz",&Lz,0);
  Nxyz=Nx*Ny*Nz;
  DegFree = Mx*My*((Mzslab==0)?Mz:1) + Lz;

  getreal("-hx",&hx,0.02);
  getreal("-hy",&hy,hx);
  getreal("-hz",&hz,hx);
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;

  double RRT=1e-25;
  sigmax = pmlsigma(RRT,(double) Npmlx*hx);
  sigmay = pmlsigma(RRT,(double) Npmly*hy);
  sigmaz = pmlsigma(RRT,(double) Npmlz*hz);

  getint("-BCPeriod",&BCPeriod,3);

  getint("-LowerPMLx",&LowerPMLx,1);
  getint("-LowerPMLy",&LowerPMLy,1);
  getint("-LowerPMLz",&LowerPMLz,1);
  getint("-Nxo",&Nxo,LowerPMLx*(Nx-Mx)/2);
  getint("-Nyo",&Nyo,LowerPMLy*(Ny-My)/2);
  getint("-Nzo",&Nzo,LowerPMLz*(Nz-Mz)/2);
  
  getreal("-Qabs",&Qabs,1e16);
  if (Qabs>1e15) Qabs=1.0/0.0;

  PetscOptionsGetString(PETSC_NULL,"-filenameprefix",filenameComm,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,filenameComm,"filenameprefix","Filename Prefix");
  PetscOptionsGetString(PETSC_NULL,"-initdatfile",initialdatafile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdatafile,"initialdatafile","Inputdata file");

  getint("-outputbase",&outputbase,50);
  getint("-readlubsfromfile",&readlubsfromfile,0);

  Universals tmp={Mx,My,Mz,Lz,Mzslab,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,Nxyz,DegFree,hx,hy,hz,hxyz,sigmax,sigmay,sigmaz,BCPeriod,LowerPMLx,LowerPMLy,LowerPMLz,Nxo,Nyo,Nzo,Qabs,NULL,NULL,0,0};
  tmp.outputbase=outputbase;
  tmp.readlubsfromfile=readlubsfromfile;
  strcpy(tmp.initialdatafile,initialdatafile);
  strcpy(tmp.filenameComm,filenameComm);

  *params=tmp;

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "setupMatVecs"
PetscErrorCode setupMatVecs(Universals params, Mat *A, Mat *C, Mat *D, Vec *vR, Vec *vI, Vec *weight, Vec *epsSReal, Vec *epsFReal)
{

  Mat Atmp, Ctmp, Dtmp;
  boosterinterp(PETSC_COMM_WORLD, &Atmp, params.Nx,params.Ny,params.Nz, params.Nxo,params.Nyo,params.Nzo, params.Mx,params.My,params.Mz,params.Mzslab, params.Lz, 0);
  int Arows, Acols;
  MatGetSize(Atmp,&Arows,&Acols);
  PetscPrintf(PETSC_COMM_WORLD,"****Dimensions of A is %d by %d \n",Arows,Acols);

  CongMat(PETSC_COMM_WORLD, &Ctmp, 6*params.Nxyz);
  ImagIMat(PETSC_COMM_WORLD, &Dtmp, 6*params.Nxyz);

  Vec vRtmp, vItmp;
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*params.Nxyz, &vRtmp);
  GetRealPartVec(vRtmp,6*params.Nxyz);
  VecDuplicate(vRtmp,&vItmp);
  MatMult(Dtmp,vRtmp,vItmp);

  Vec weighttmp;
  VecDuplicate(vRtmp,&weighttmp);
  GetWeightVecGeneralSym(weighttmp, params.Nx,params.Ny,params.Nz, params.LowerPMLx,params.LowerPMLy,params.LowerPMLz);

  Vec epsS, epsF;
  MatCreateVecs(Atmp,&epsS,&epsF);

  *A=Atmp;
  *C=Ctmp;
  *D=Dtmp;
  *vR=vRtmp;
  *vI=vItmp;
  *weight=weighttmp;
  *epsSReal=epsS;
  *epsFReal=epsF;

  return 0;

}

#undef __FUNCT__
#define __FUNCT__ "makemaxwell"
PetscErrorCode makemaxwell(char file[PETSC_MAX_PATH_LEN], Universals params, Mat A, Mat D, Vec vR, Vec weight, Maxwell *fdfd)
{

  FILE *fp;
  fp = fopen(file,"r");
  
  int blochcondition;
  double beta[3];
  int bx[2], by[2], bz[2];
  double freq,omega;
  double epsdiffU, epsdiffM, epsdiffL, epsbkgU, epsbkgM, epsbkgL;
  char Jfile[PETSC_MAX_PATH_LEN];

  int err;
  err=fscanf(fp,"blochcondition: %d\n",&blochcondition);
  err=fscanf(fp,"betax: %lf\n",beta);
  err=fscanf(fp,"betay: %lf\n",beta+1);
  err=fscanf(fp,"betaz: %lf\n",beta+2);
  err=fscanf(fp,"bxl: %d\n",bx);
  err=fscanf(fp,"bxu: %d\n",bx+1);
  err=fscanf(fp,"byl: %d\n",by);
  err=fscanf(fp,"byu: %d\n",by+1);
  err=fscanf(fp,"bzl: %d\n",bz);
  err=fscanf(fp,"bzu: %d\n",bz+1);
  err=fscanf(fp,"freq: %lf\n",&freq);
  err=fscanf(fp,"epsdiffU: %lf\n",&epsdiffU);
  err=fscanf(fp,"epsdiffM: %lf\n",&epsdiffM);
  err=fscanf(fp,"epsdiffL: %lf\n",&epsdiffL);
  err=fscanf(fp,"epsbkgU: %lf\n",&epsbkgU);
  err=fscanf(fp,"epsbkgM: %lf\n",&epsbkgM);
  err=fscanf(fp,"epsbkgL: %lf\n",&epsbkgL);
  err=fscanf(fp,"Jfile: %s\n",Jfile);

  PetscPrintf(PETSC_COMM_WORLD,"blochcondition: %d\n",blochcondition);
  PetscPrintf(PETSC_COMM_WORLD,"betax: %lf\n",beta[0]);
  PetscPrintf(PETSC_COMM_WORLD,"betay: %lf\n",beta[1]);
  PetscPrintf(PETSC_COMM_WORLD,"betaz: %lf\n",beta[2]);
  PetscPrintf(PETSC_COMM_WORLD,"bxl: %d\n",bx[0]);
  PetscPrintf(PETSC_COMM_WORLD,"bxu: %d\n",bx[1]);
  PetscPrintf(PETSC_COMM_WORLD,"byl: %d\n",by[0]);
  PetscPrintf(PETSC_COMM_WORLD,"byu: %d\n",by[1]);
  PetscPrintf(PETSC_COMM_WORLD,"bzl: %d\n",bz[0]);
  PetscPrintf(PETSC_COMM_WORLD,"bzu: %d\n",bz[1]);
  PetscPrintf(PETSC_COMM_WORLD,"freq: %lf\n",freq);
  PetscPrintf(PETSC_COMM_WORLD,"epsdiffU: %lf\n",epsdiffU);
  PetscPrintf(PETSC_COMM_WORLD,"epsdiffM: %lf\n",epsdiffM);
  PetscPrintf(PETSC_COMM_WORLD,"epsdiffL: %lf\n",epsdiffL);
  PetscPrintf(PETSC_COMM_WORLD,"epsbkgU: %lf\n",epsbkgU);
  PetscPrintf(PETSC_COMM_WORLD,"epsbkgM: %lf\n",epsbkgM);
  PetscPrintf(PETSC_COMM_WORLD,"epsbkgL: %lf\n",epsbkgL);
  PetscPrintf(PETSC_COMM_WORLD,"Jfile: %s\n",Jfile);

  fclose(fp);

  omega=2*PI*freq;
  Mat M;
  Vec muinvpml;
  double *muinv;
  int add=1;
  MuinvPMLGeneral(PETSC_COMM_SELF, &muinvpml, params.Nx,params.Ny,params.Nz,params.Npmlx,params.Npmly,params.Npmlz, params.sigmax,params.sigmay,params.sigmaz, omega, params.LowerPMLx,params.LowerPMLy,params.LowerPMLz);
  muinv = (double *) malloc(sizeof(double)*6*params.Nxyz);
  AddMuAbsorption(muinv,muinvpml,params.Qabs,add);
  if(blochcondition){
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M, params.Nx,params.Ny,params.Nz, params.hx,params.hy,params.hz, bx,by,bz, muinv, params.BCPeriod, beta);
  }else{
    MoperatorGeneral(PETSC_COMM_WORLD, &M, params.Nx,params.Ny,params.Nz, params.hx,params.hy,params.hz, bx,by,bz, muinv, params.BCPeriod);
  }

  Vec epsdiff, epsbkg, epspml, epspmlQ, epscoef;
  VecDuplicate(vR,&epsdiff);
  VecDuplicate(vR,&epsbkg);
  VecDuplicate(vR,&epspml);
  VecDuplicate(vR,&epspmlQ);
  VecDuplicate(vR,&epscoef);
  
  VecSet(epsdiff,0.0);
  VecSet(epsbkg,0.0);
  makethreelayeredepsdiff(epsdiff, params.Nx,params.Ny,params.Nz, params.Nzo, params.Mz, epsdiffU,epsdiffM,epsdiffL);
  makethreelayeredepsbkg(epsbkg, params.Nx,params.Ny,params.Nz, params.Nzo, params.Mz, epsbkgU,epsbkgM,epsbkgL);

  EpsPMLGeneral(PETSC_COMM_WORLD, epspml, params.Nx,params.Ny,params.Nz,params.Npmlx,params.Npmly,params.Npmlz, params.sigmax,params.sigmay,params.sigmaz, omega, params.LowerPMLx,params.LowerPMLy,params.LowerPMLz);
  EpsCombine(D, NULL, epspml, epspmlQ, epscoef, params.Qabs, omega, epsdiff);
  
  VecDestroy(&muinvpml);
  VecDestroy(&epspml);
  free(muinv);

  Vec J, weightedJ, b, x;
  VecDuplicate(vR,&J);
  VecDuplicate(vR,&weightedJ);
  VecDuplicate(vR,&b);
  VecDuplicate(vR,&x);

  /*
  double *Jdist;
  FILE *Jptf;
  int i;
  Jdist = (double *) malloc(6*params.Nxyz*sizeof(double));
  Jptf = fopen(Jfile,"r");
  for (i=0;i<6*params.Nxyz;i++)
    {
      err=fscanf(Jptf,"%lf",&Jdist[i]);
    }
  fclose(Jptf);
  ArrayToVec(Jdist,J);
  free(Jdist);*/

  VecSet(J,1.0);
  VecPointwiseMult(weightedJ,weight,J);
  MatMult(D,J,b);
  VecScale(b,omega);

  Maxwell tmp={M,epsdiff,epsbkg,epspmlQ,epscoef,J,weightedJ,b,x,omega};
  *fdfd=tmp;

  return 0;

}
