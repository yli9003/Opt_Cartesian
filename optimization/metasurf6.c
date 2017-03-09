#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

double RRT, sigmax, sigmay, sigmaz;
int mma_verbose;
int initdirect, maxit;

int count=1;
Mat A,B,C,D;
Vec vR, weight, vgradlocal, epsFReal;
VecScatter scatter;
IS from, to;
char filenameComm[PETSC_MAX_PATH_LEN];

int pSIMP;
double bproj, etaproj;
Mat Hfilt;
KSP kspH;
int itsH;

typedef struct {
  double *epsdiff;
  double *epsbkg;
  double epssub;
  double epssubdiff;
  double epsair;
  double epsairdiff;
  double epsmid;
  double epsmiddiff;
} epsinfo;

double mintrans;

/*------------------------------------------------------*/

PetscErrorCode setupKSP(MPI_Comm comm, KSP *ksp, PC *pc, int solver, int iteronly);
double pfunc(int DegFree, double *epsopt, double *grad, void *data);
double pfunc2(int DegFree, double *epsopt, double *grad, void *data);

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize ------*/
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;
  int iteronly=0;

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0) 
    mma_verbose=1;

/*****************************************************-------Set up the options parameters-------------********************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
  PetscBool flg;

  PetscOptionsGetInt(PETSC_NULL,"-initdirect",&initdirect,&flg);  MyCheckAndOutputInt(flg,initdirect,"initdirect","Initial number of direct LU solves");
  PetscOptionsGetInt(PETSC_NULL,"-maxit",&maxit,&flg);  MyCheckAndOutputInt(flg,maxit,"maxit","maximum krylov iterations before invoking direct solve");

  ////metaphase
  double metaphase1, metaphase2, metaphase3, metaphase4, metaphase5;
  getreal("-metaphase1",&metaphase1,0);
  getreal("-metaphase2",&metaphase2,0);
  getreal("-metaphase3",&metaphase3,0);
  getreal("-metaphase4",&metaphase4,0);
  getreal("-metaphase4",&metaphase5,0);

  int blochcondition;
  double beta1[3]={0,0,0}, beta2[3]={0,0,0}, beta3[3]={0,0,0}, beta4[3]={0,0,0}, beta5[3]={0,0,0};
  getint("-blochcondition",&blochcondition,0);
  getreal("-beta1x",beta1,0);
  getreal("-beta1y",beta1+1,0);
  getreal("-beta1z",beta1+2,0);
  getreal("-beta2x",beta2,0);
  getreal("-beta2y",beta2+1,0);
  getreal("-beta2z",beta2+2,0);
  getreal("-beta3x",beta3,0);
  getreal("-beta3y",beta3+1,0);
  getreal("-beta3z",beta3+2,0);
  getreal("-beta4x",beta4,0);
  getreal("-beta4y",beta4+1,0);
  getreal("-beta4z",beta4+2,0);
  getreal("-beta5x",beta5,0);
  getreal("-beta5y",beta5+1,0);
  getreal("-beta5z",beta5+2,0);

  int Mx, My, Mzslab, Nx, Ny, Nz, Npmlx, Npmly, Npmlz, Nxyz, DegFree;
  double hx, hy, hz, hxyz;
  PetscOptionsGetInt(PETSC_NULL,"-Mx",&Mx,&flg);  MyCheckAndOutputInt(flg,Mx,"Mx","Mx");
  PetscOptionsGetInt(PETSC_NULL,"-My",&My,&flg);  MyCheckAndOutputInt(flg,My,"My","My");

  PetscOptionsGetInt(PETSC_NULL,"-Mzslab",&Mzslab,&flg);  MyCheckAndOutputInt(flg,Mzslab,"Mzslab","Mzslab");

  PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg);  MyCheckAndOutputInt(flg,Nx,"Nx","Nx");
  PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg);  MyCheckAndOutputInt(flg,Ny,"Ny","Ny");
  PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg);  MyCheckAndOutputInt(flg,Nz,"Nz","Nz");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlx",&Npmlx,&flg);  MyCheckAndOutputInt(flg,Npmlx,"Npmlx","Npmlx");
  PetscOptionsGetInt(PETSC_NULL,"-Npmly",&Npmly,&flg);  MyCheckAndOutputInt(flg,Npmly,"Npmly","Npmly");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlz",&Npmlz,&flg);  MyCheckAndOutputInt(flg,Npmlz,"Npmlz","Npmlz");

  int LowerPMLx, LowerPMLy, LowerPMLz;
  getint("-LowerPMLx",&LowerPMLx,1);
  getint("-LowerPMLy",&LowerPMLy,1);
  getint("-LowerPMLz",&LowerPMLz,1);

  int Nxo,Nyo;
  getint("-Nxo",&Nxo,LowerPMLx*(Nx-Mx)/2);
  getint("-Nyo",&Nyo,LowerPMLy*(Ny-My)/2);

  int nlayers;
  int *Mz, *Nzo;
  epsinfo eps1, eps2, eps3, eps4, eps5;
  int i;
  DegFree=0;
  getint("-nlayers",&nlayers,1);
  Mz =(int *) malloc(nlayers*sizeof(int));
  Nzo=(int *) malloc(nlayers*sizeof(int));
  eps1.epsdiff=(double *) malloc(nlayers*sizeof(double));
  eps2.epsdiff=(double *) malloc(nlayers*sizeof(double));
  eps3.epsdiff=(double *) malloc(nlayers*sizeof(double));
  eps4.epsdiff=(double *) malloc(nlayers*sizeof(double));
  eps5.epsdiff=(double *) malloc(nlayers*sizeof(double));
  eps1.epsbkg=(double *) malloc(nlayers*sizeof(double));
  eps2.epsbkg=(double *) malloc(nlayers*sizeof(double));
  eps3.epsbkg=(double *) malloc(nlayers*sizeof(double));
  eps4.epsbkg=(double *) malloc(nlayers*sizeof(double));
  eps5.epsbkg=(double *) malloc(nlayers*sizeof(double));
  char tmpflg[PETSC_MAX_PATH_LEN];
  for (i=0;i<nlayers;i++) {
    sprintf(tmpflg,"-Mz[%d]",i+1);
    getint(tmpflg,Mz+i,10);
    sprintf(tmpflg,"-Nzo[%d]",i+1);
    getint(tmpflg,Nzo+i,LowerPMLz*(Nz-Mz[i])/2);
    DegFree=DegFree+Mx*My*((Mzslab==0)?Mz[i]:1);
  
    sprintf(tmpflg,"-eps1diff[%d]",i+1);
    getreal(tmpflg,eps1.epsdiff+i,3.6575);
    sprintf(tmpflg,"-eps1bkg[%d]",i+1);
    getreal(tmpflg,eps1.epsbkg+i,2.1025);

    sprintf(tmpflg,"-eps2diff[%d]",i+1);
    getreal(tmpflg,eps2.epsdiff+i,3.6575);
    sprintf(tmpflg,"-eps2bkg[%d]",i+1);
    getreal(tmpflg,eps2.epsbkg+i,2.1025);

    sprintf(tmpflg,"-eps3diff[%d]",i+1);
    getreal(tmpflg,eps3.epsdiff+i,3.6575);
    sprintf(tmpflg,"-eps3bkg[%d]",i+1);
    getreal(tmpflg,eps3.epsbkg+i,2.1025);

    sprintf(tmpflg,"-eps4diff[%d]",i+1);
    getreal(tmpflg,eps4.epsdiff+i,3.6575);
    sprintf(tmpflg,"-eps4bkg[%d]",i+1);
    getreal(tmpflg,eps4.epsbkg+i,2.1025);

    sprintf(tmpflg,"-eps5diff[%d]",i+1);
    getreal(tmpflg,eps5.epsdiff+i,3.6575);
    sprintf(tmpflg,"-eps5bkg[%d]",i+1);
    getreal(tmpflg,eps5.epsbkg+i,2.1025);

  }
  getreal("-eps1subdiff",&eps1.epssubdiff,0);
  getreal("-eps1airdiff",&eps1.epsairdiff,0);
  getreal("-eps1middiff",&eps1.epsmiddiff,0);
  getreal("-eps1sub",&eps1.epssub,2.1025);
  getreal("-eps1air",&eps1.epsair,1.0);
  getreal("-eps1mid",&eps1.epsmid,2.1025);

  getreal("-eps2subdiff",&eps2.epssubdiff,0);
  getreal("-eps2airdiff",&eps2.epsairdiff,0);
  getreal("-eps2middiff",&eps2.epsmiddiff,0);
  getreal("-eps2sub",&eps2.epssub,2.1025);
  getreal("-eps2air",&eps2.epsair,1.0);
  getreal("-eps2mid",&eps2.epsmid,2.1025);

  getreal("-eps3subdiff",&eps3.epssubdiff,0);
  getreal("-eps3airdiff",&eps3.epsairdiff,0);
  getreal("-eps3middiff",&eps3.epsmiddiff,0);
  getreal("-eps3sub",&eps3.epssub,2.1025);
  getreal("-eps3air",&eps3.epsair,1.0);
  getreal("-eps3mid",&eps3.epsmid,2.1025);

  getreal("-eps4subdiff",&eps4.epssubdiff,0);
  getreal("-eps4airdiff",&eps4.epsairdiff,0);
  getreal("-eps4middiff",&eps4.epsmiddiff,0);
  getreal("-eps4sub",&eps4.epssub,2.1025);
  getreal("-eps4air",&eps4.epsair,1.0);
  getreal("-eps4mid",&eps4.epsmid,2.1025);

  getreal("-eps5subdiff",&eps5.epssubdiff,0);
  getreal("-eps5airdiff",&eps5.epsairdiff,0);
  getreal("-eps5middiff",&eps5.epsmiddiff,0);
  getreal("-eps5sub",&eps5.epssub,2.1025);
  getreal("-eps5air",&eps5.epsair,1.0);
  getreal("-eps5mid",&eps5.epsmid,2.1025);

  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  getreal("-hy",&hy,hx);
  getreal("-hz",&hz,hx);
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;

  RRT=1e-25;
  sigmax = pmlsigma(RRT,(double) Npmlx*hx);
  sigmay = pmlsigma(RRT,(double) Npmly*hy);
  sigmaz = pmlsigma(RRT,(double) Npmlz*hz);

  int BCPeriod;
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod");

  Nxyz=Nx*Ny*Nz;

  int b1x[2], b1y[2], b1z[2], b2x[2], b2y[2], b2z[2], b3x[2], b3y[2], b3z[2], b4x[2], b4y[2], b4z[2], b5x[2], b5y[2], b5z[2];
  PetscOptionsGetInt(PETSC_NULL,"-b1xl",b1x,&flg);    MyCheckAndOutputInt(flg,b1x[0],"b1xl","BC at x lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1xu",b1x+1,&flg);  MyCheckAndOutputInt(flg,b1x[1],"b1xu","BC at x upper for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1yl",b1y,&flg);    MyCheckAndOutputInt(flg,b1y[0],"b1yl","BC at y lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1yu",b1y+1,&flg);  MyCheckAndOutputInt(flg,b1y[1],"b1yu","BC at y upper for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1zl",b1z,&flg);    MyCheckAndOutputInt(flg,b1z[0],"b1zl","BC at z lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1zu",b1z+1,&flg);  MyCheckAndOutputInt(flg,b1z[1],"b1zu","BC at z upper for mode 1");

  PetscOptionsGetInt(PETSC_NULL,"-b2xl",b2x,&flg);    MyCheckAndOutputInt(flg,b2x[0],"b2xl","BC at x lower for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2xu",b2x+1,&flg);  MyCheckAndOutputInt(flg,b2x[1],"b2xu","BC at x upper for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2yl",b2y,&flg);    MyCheckAndOutputInt(flg,b2y[0],"b2yl","BC at y lower for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2yu",b2y+1,&flg);  MyCheckAndOutputInt(flg,b2y[1],"b2yu","BC at y upper for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2zl",b2z,&flg);    MyCheckAndOutputInt(flg,b2z[0],"b2zl","BC at z lower for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2zu",b2z+1,&flg);  MyCheckAndOutputInt(flg,b2z[1],"b2zu","BC at z upper for mode 2");

  PetscOptionsGetInt(PETSC_NULL,"-b3xl",b3x,&flg);    MyCheckAndOutputInt(flg,b3x[0],"b3xl","BC at x lower for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3xu",b3x+1,&flg);  MyCheckAndOutputInt(flg,b3x[1],"b3xu","BC at x upper for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3yl",b3y,&flg);    MyCheckAndOutputInt(flg,b3y[0],"b3yl","BC at y lower for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3yu",b3y+1,&flg);  MyCheckAndOutputInt(flg,b3y[1],"b3yu","BC at y upper for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3zl",b3z,&flg);    MyCheckAndOutputInt(flg,b3z[0],"b3zl","BC at z lower for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3zu",b3z+1,&flg);  MyCheckAndOutputInt(flg,b3z[1],"b3zu","BC at z upper for mode 3");

  PetscOptionsGetInt(PETSC_NULL,"-b4xl",b4x,&flg);    MyCheckAndOutputInt(flg,b4x[0],"b4xl","BC at x lower for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4xu",b4x+1,&flg);  MyCheckAndOutputInt(flg,b4x[1],"b4xu","BC at x upper for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4yl",b4y,&flg);    MyCheckAndOutputInt(flg,b4y[0],"b4yl","BC at y lower for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4yu",b4y+1,&flg);  MyCheckAndOutputInt(flg,b4y[1],"b4yu","BC at y upper for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4zl",b4z,&flg);    MyCheckAndOutputInt(flg,b4z[0],"b4zl","BC at z lower for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4zu",b4z+1,&flg);  MyCheckAndOutputInt(flg,b4z[1],"b4zu","BC at z upper for mode 4");

  PetscOptionsGetInt(PETSC_NULL,"-b5xl",b5x,&flg);    MyCheckAndOutputInt(flg,b5x[0],"b5xl","BC at x lower for mode 5");
  PetscOptionsGetInt(PETSC_NULL,"-b5xu",b5x+1,&flg);  MyCheckAndOutputInt(flg,b5x[1],"b5xu","BC at x upper for mode 5");
  PetscOptionsGetInt(PETSC_NULL,"-b5yl",b5y,&flg);    MyCheckAndOutputInt(flg,b5y[0],"b5yl","BC at y lower for mode 5");
  PetscOptionsGetInt(PETSC_NULL,"-b5yu",b5y+1,&flg);  MyCheckAndOutputInt(flg,b5y[1],"b5yu","BC at y upper for mode 5");
  PetscOptionsGetInt(PETSC_NULL,"-b5zl",b5z,&flg);    MyCheckAndOutputInt(flg,b5z[0],"b5zl","BC at z lower for mode 5");
  PetscOptionsGetInt(PETSC_NULL,"-b5zu",b5z+1,&flg);  MyCheckAndOutputInt(flg,b5z[1],"b5zu","BC at z upper for mode 5");

  int outputbase;
  PetscOptionsGetInt(PETSC_NULL,"-outputbase",&outputbase,&flg); MyCheckAndOutputInt(flg,outputbase,"outputbase","outputbase");

  int J1dir, J2dir, J3dir, J4dir, J5dir;
  getint("-J1dir",&J1dir,2);
  getint("-J2dir",&J2dir,J1dir);
  getint("-J3dir",&J3dir,J1dir);
  getint("-J4dir",&J4dir,J1dir);
  getint("-J5dir",&J5dir,J1dir);

  double freq1, freq2, freq3, freq4, freq5, omega1, omega2, omega3, omega4, omega5;
  getreal("-freq1",&freq1,1.0);
  getreal("-freq2",&freq2,1.0);
  getreal("-freq3",&freq3,1.0);
  getreal("-freq4",&freq4,1.0);
  getreal("-freq5",&freq5,1.0);
  omega1=2.0*PI*freq1;
  omega2=2.0*PI*freq2;
  omega3=2.0*PI*freq3;
  omega4=2.0*PI*freq4;
  omega5=2.0*PI*freq5;

  //same block source position for all three currents
  double jlx, jux, jly, juy, jlz, juz;
  getreal("-jlx",&jlx,0);
  getreal("-jux",&jux,Nx*hx);
  getreal("-jly",&jly,0);
  getreal("-juy",&juy,Ny*hy);
  getreal("-jlz",&jlz,Npmlz*hz+1/(5*freq1));
  getreal("-juz",&juz,jlz+hz);

  double Jmag1, Jmag2, Jmag3, Jmag4, Jmag5;
  getreal("-Jmag1",&Jmag1,1.0);
  getreal("-Jmag2",&Jmag2,Jmag1);
  getreal("-Jmag3",&Jmag3,Jmag1);
  getreal("-Jmag4",&Jmag4,Jmag1);
  getreal("-Jmag5",&Jmag5,Jmag1);

  int ixref, iyref, izref, icref;
  getint("-ixref",&ixref,floor(Nx/2));
  getint("-iyref",&iyref,floor(Ny/2));
  getint("-izref",&izref,floor(Nz-Npmlz-1/(5*freq1*hz)));
  getint("-icref",&icref,2);

  PetscPrintf(PETSC_COMM_WORLD,"omegas, DegFree: %g, %g, %g, %g, %g, %d \n", omega1, omega2, omega3, omega4, omega5, DegFree);

  double Qabs;
  PetscOptionsGetReal(PETSC_NULL,"-Qabs",&Qabs,&flg);  MyCheckAndOutputDouble(flg,Qabs,"Qabs","Qabs");
  if (Qabs>1e15) Qabs=1.0/0.0;

  char initialdatafile[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-filenameprefix",filenameComm,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,filenameComm,"filenameprefix","Filename Prefix");
  PetscOptionsGetString(PETSC_NULL,"-initdatfile",initialdatafile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdatafile,"initialdatafile","Inputdata file");

  int solver;
  PetscOptionsGetInt(PETSC_NULL,"-solver",&solver,&flg);  MyCheckAndOutputInt(flg,solver,"solver","LU Direct solver choice (0 PASTIX, 1 MUMPS, 2 SUPERLU_DIST)");

  double sH, nR;
  int dimH;
  getint("-pSIMP",&pSIMP,1);
  getreal("-bproj",&bproj,0);
  getreal("-etaproj",&etaproj,0.5);
  getreal("-sH",&sH,-1);
  getreal("-nR",&nR,0);
  getint("-dimH",&dimH,1);

  int readlubsfromfile;
  getint("-readlubsfromfile",&readlubsfromfile,0);

/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
  
  /*------Set up the A, C, D matrices--------------*/
  Mat B1, B2;
  layeredA(PETSC_COMM_WORLD,&A, Nx,Ny,Nz, nlayers,Nxo,Nyo,Nzo, Mx,My,Mz, Mzslab);

  int Arows, Acols;
  MatGetSize(A,&Arows,&Acols);
  PetscPrintf(PETSC_COMM_WORLD,"****Dimensions of A is %d by %d \n",Arows,Acols);

  CongMat(PETSC_COMM_WORLD, &C, 6*Nxyz);
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  GetProjMat(PETSC_COMM_WORLD,&B1,J3dir-1,J1dir-1,Nx,Ny,Nz);
  GetProjMat(PETSC_COMM_WORLD,&B2,J3dir-1,J2dir-1,Nx,Ny,Nz);

  /*-----Set up vR, weight------*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &vR);CHKERRQ(ierr);
  GetRealPartVec(vR,6*Nxyz);

  ierr = VecDuplicate(vR,&weight); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vR, "vR");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) weight, "weight");CHKERRQ(ierr);

  GetWeightVecGeneralSym(weight,Nx,Ny,Nz,LowerPMLx,LowerPMLy,LowerPMLz); 

  /*----Set up the universal parts of M1, M2 and M3-------*/
  Mat M1, M2, M3, M4, M5;
  Vec muinvpml1, muinvpml2, muinvpml3, muinvpml4, muinvpml5;
  MuinvPMLGeneral(PETSC_COMM_SELF, &muinvpml1,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega1,LowerPMLx,LowerPMLy,LowerPMLz);
  MuinvPMLGeneral(PETSC_COMM_SELF, &muinvpml2,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega2,LowerPMLx,LowerPMLy,LowerPMLz); 
  MuinvPMLGeneral(PETSC_COMM_SELF, &muinvpml3,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega3,LowerPMLx,LowerPMLy,LowerPMLz); 
  MuinvPMLGeneral(PETSC_COMM_SELF, &muinvpml4,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega4,LowerPMLx,LowerPMLy,LowerPMLz); 
  MuinvPMLGeneral(PETSC_COMM_SELF, &muinvpml5,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega5,LowerPMLx,LowerPMLy,LowerPMLz); 
  double *muinv1, *muinv2, *muinv3, *muinv4, *muinv5;
  muinv1 = (double *) malloc(sizeof(double)*6*Nxyz);
  muinv2 = (double *) malloc(sizeof(double)*6*Nxyz);
  muinv3 = (double *) malloc(sizeof(double)*6*Nxyz);
  muinv4 = (double *) malloc(sizeof(double)*6*Nxyz);
  muinv5 = (double *) malloc(sizeof(double)*6*Nxyz);
  int add=1;
  AddMuAbsorption(muinv1,muinvpml1,Qabs,add);
  AddMuAbsorption(muinv2,muinvpml2,Qabs,add);
  AddMuAbsorption(muinv3,muinvpml3,Qabs,add);
  AddMuAbsorption(muinv4,muinvpml4,Qabs,add);
  AddMuAbsorption(muinv5,muinvpml5,Qabs,add);

  if(blochcondition){
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M1, Nx,Ny,Nz, hx,hy,hz, b1x,b1y,b1z, muinv1, BCPeriod, beta1);
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M2, Nx,Ny,Nz, hx,hy,hz, b2x,b2y,b2z, muinv2, BCPeriod, beta2);
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M3, Nx,Ny,Nz, hx,hy,hz, b3x,b3y,b3z, muinv3, BCPeriod, beta3);
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M4, Nx,Ny,Nz, hx,hy,hz, b4x,b4y,b4z, muinv4, BCPeriod, beta4);
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M5, Nx,Ny,Nz, hx,hy,hz, b5x,b5y,b5z, muinv5, BCPeriod, beta5);
  }else{
    MoperatorGeneral(PETSC_COMM_WORLD, &M1, Nx,Ny,Nz, hx,hy,hz, b1x,b1y,b1z, muinv1, BCPeriod);
    MoperatorGeneral(PETSC_COMM_WORLD, &M2, Nx,Ny,Nz, hx,hy,hz, b2x,b2y,b2z, muinv2, BCPeriod);
    MoperatorGeneral(PETSC_COMM_WORLD, &M3, Nx,Ny,Nz, hx,hy,hz, b3x,b3y,b3z, muinv3, BCPeriod);
    MoperatorGeneral(PETSC_COMM_WORLD, &M4, Nx,Ny,Nz, hx,hy,hz, b4x,b4y,b4z, muinv4, BCPeriod);
    MoperatorGeneral(PETSC_COMM_WORLD, &M5, Nx,Ny,Nz, hx,hy,hz, b5x,b5y,b5z, muinv5, BCPeriod);
  }
  ierr = PetscObjectSetName((PetscObject) M1, "M1"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) M2, "M2"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) M3, "M3"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) M4, "M4"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) M5, "M5"); CHKERRQ(ierr);

  /*----Set up the epsilon PML vectors--------*/
  Vec unitx,unity,unitz;
  ierr = VecDuplicate(vR,&unitx);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unity);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unitz);CHKERRQ(ierr);
  GetUnitVec(unitx,0,6*Nxyz);
  GetUnitVec(unity,1,6*Nxyz);
  GetUnitVec(unitz,2,6*Nxyz);

  Vec epsI, epsII, epsIII, epsIV, epsV;
  PetscPrintf(PETSC_COMM_WORLD,"***NOTE: Anisotropic epsilon not supported in this version. \n");
  ierr = VecDuplicate(vR,&epsI);  CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsII); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsIII);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsIV);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsV);CHKERRQ(ierr);

  VecSet(epsI,0.0);
  layeredepsdiff(epsI,   Nx,Ny,Nz, nlayers,Nzo,Mz, eps1.epsdiff, eps1.epssubdiff, eps1.epsairdiff, eps1.epsmiddiff);
  VecSet(epsII,0.0); 
  layeredepsdiff(epsII,  Nx,Ny,Nz, nlayers,Nzo,Mz, eps2.epsdiff, eps2.epssubdiff, eps2.epsairdiff, eps2.epsmiddiff);
  VecSet(epsIII,0.0);
  layeredepsdiff(epsIII, Nx,Ny,Nz, nlayers,Nzo,Mz, eps3.epsdiff, eps3.epssubdiff, eps3.epsairdiff, eps3.epsmiddiff);
  VecSet(epsIV,0.0);
  layeredepsdiff(epsIV, Nx,Ny,Nz, nlayers,Nzo,Mz, eps4.epsdiff, eps4.epssubdiff, eps4.epsairdiff, eps4.epsmiddiff);
  VecSet(epsV,0.0);
  layeredepsdiff(epsV, Nx,Ny,Nz, nlayers,Nzo,Mz, eps5.epsdiff, eps5.epssubdiff, eps5.epsairdiff, eps5.epsmiddiff);

  Vec epspml1, epspml2, epspml3, epspml4, epspml5, epspmlQ1, epspmlQ2, epspmlQ3, epspmlQ4, epspmlQ5, epscoef1, epscoef2, epscoef3, epscoef4, epscoef5;
  ierr = VecDuplicate(vR,&epspml1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml3);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml4);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml5);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ3);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ4);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ5);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef3);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef4);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef5);CHKERRQ(ierr);

  EpsPMLGeneral(PETSC_COMM_WORLD, epspml1,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega1, LowerPMLx,LowerPMLy,LowerPMLz);
  EpsPMLGeneral(PETSC_COMM_WORLD, epspml2,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega2, LowerPMLx,LowerPMLy,LowerPMLz);
  EpsPMLGeneral(PETSC_COMM_WORLD, epspml3,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega3, LowerPMLx,LowerPMLy,LowerPMLz);
  EpsPMLGeneral(PETSC_COMM_WORLD, epspml4,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega4, LowerPMLx,LowerPMLy,LowerPMLz);
  EpsPMLGeneral(PETSC_COMM_WORLD, epspml5,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega5, LowerPMLx,LowerPMLy,LowerPMLz);

  EpsCombine(D, weight, epspml1, epspmlQ1, epscoef1, Qabs, omega1, epsI);
  EpsCombine(D, weight, epspml2, epspmlQ2, epscoef2, Qabs, omega2, epsII);
  EpsCombine(D, weight, epspml3, epspmlQ3, epscoef3, Qabs, omega3, epsIII);
  EpsCombine(D, weight, epspml4, epspmlQ4, epscoef4, Qabs, omega4, epsIV);
  EpsCombine(D, weight, epspml5, epspmlQ5, epscoef5, Qabs, omega5, epsV);

  /*-----Set up epsmedium, epsSReal, epsFReal, epsC, epsCi, epsP, vgrad, vgradlocal ------*/
  Vec epsmedium1, epsmedium2, epsmedium3, epsmedium4, epsmedium5;
  ierr = VecDuplicate(vR,&epsmedium1); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsmedium2); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsmedium3); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsmedium4); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsmedium5); CHKERRQ(ierr);

  VecSet(epsmedium1,0.0);
  layeredepsbkg(epsmedium1, Nx,Ny,Nz, nlayers,Nzo,Mz, eps1.epsbkg, eps1.epssub, eps1.epsair, eps1.epsmid);
  VecSet(epsmedium2,0.0); 
  layeredepsbkg(epsmedium2, Nx,Ny,Nz, nlayers,Nzo,Mz, eps2.epsbkg, eps2.epssub, eps2.epsair, eps2.epsmid);
  VecSet(epsmedium3,0.0);
  layeredepsbkg(epsmedium3, Nx,Ny,Nz, nlayers,Nzo,Mz, eps3.epsbkg, eps3.epssub, eps3.epsair, eps3.epsmid);
  VecSet(epsmedium4,0.0);
  layeredepsbkg(epsmedium4, Nx,Ny,Nz, nlayers,Nzo,Mz, eps4.epsbkg, eps4.epssub, eps4.epsair, eps4.epsmid);
  VecSet(epsmedium5,0.0);
  layeredepsbkg(epsmedium5, Nx,Ny,Nz, nlayers,Nzo,Mz, eps5.epsbkg, eps5.epssub, eps5.epsair, eps5.epsmid);

  Vec epsSReal, vgrad;
  ierr = MatCreateVecs(A,&epsSReal, &epsFReal); CHKERRQ(ierr);
  ierr = VecDuplicate(epsSReal, &vgrad); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) epsmedium1,  "epsmedium1");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsmedium2,  "epsmedium2");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsmedium3,  "epsmedium3");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsmedium4,  "epsmedium4");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsmedium5,  "epsmedium5");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsFReal, "epsFReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vgrad, "vgrad");CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, DegFree, &vgradlocal); CHKERRQ(ierr);

  /*---------Set up J, b and weightedJ-------------*/
  Vec J1, J2, J3, J4, J5, b1, b2, b3, b4, b5, weightedJ1, weightedJ2, weightedJ3, weightedJ4, weightedJ5;

  SourceBlock(PETSC_COMM_WORLD, &J1, Nx,Ny,Nz, hx,hy,hz, jlx,jux,jly,juy,jlz,juz, 1.0/hz,J1dir);
  SourceBlock(PETSC_COMM_WORLD, &J2, Nx,Ny,Nz, hx,hy,hz, jlx,jux,jly,juy,jlz,juz, 1.0/hz,J2dir);
  SourceBlock(PETSC_COMM_WORLD, &J3, Nx,Ny,Nz, hx,hy,hz, jlx,jux,jly,juy,jlz,juz, 1.0/hz,J3dir);
  SourceBlock(PETSC_COMM_WORLD, &J4, Nx,Ny,Nz, hx,hy,hz, jlx,jux,jly,juy,jlz,juz, 1.0/hz,J4dir);
  SourceBlock(PETSC_COMM_WORLD, &J5, Nx,Ny,Nz, hx,hy,hz, jlx,jux,jly,juy,jlz,juz, 1.0/hz,J5dir);

  int Jopt;
  getint("-Jopt",&Jopt,0);

  if(Jopt==1){
    char inputsrc1[PETSC_MAX_PATH_LEN], inputsrc2[PETSC_MAX_PATH_LEN], inputsrc3[PETSC_MAX_PATH_LEN], inputsrc4[PETSC_MAX_PATH_LEN], inputsrc5[PETSC_MAX_PATH_LEN];
    PetscOptionsGetString(PETSC_NULL,"-inputsrc1",inputsrc1,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrc1,"inputsrc1","Input source current 1");
    PetscOptionsGetString(PETSC_NULL,"-inputsrc2",inputsrc2,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrc2,"inputsrc2","Input source current 2");
    PetscOptionsGetString(PETSC_NULL,"-inputsrc3",inputsrc3,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrc3,"inputsrc3","Input source current 3");
    PetscOptionsGetString(PETSC_NULL,"-inputsrc4",inputsrc4,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrc4,"inputsrc4","Input source current 4");
    PetscOptionsGetString(PETSC_NULL,"-inputsrc5",inputsrc5,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrc5,"inputsrc5","Input source current 5");

    double *Jdist1, *Jdist2, *Jdist3, *Jdist4, *Jdist5;
    FILE *Jptf1, *Jptf2, *Jptf3, *Jptf4, *Jptf5;
    int inJi;
    Jdist1 = (double *) malloc(6*Nxyz*sizeof(double));
    Jdist2 = (double *) malloc(6*Nxyz*sizeof(double));
    Jdist3 = (double *) malloc(6*Nxyz*sizeof(double));
    Jdist4 = (double *) malloc(6*Nxyz*sizeof(double));
    Jdist5 = (double *) malloc(6*Nxyz*sizeof(double));

    Jptf1 = fopen(inputsrc1,"r");
    Jptf2 = fopen(inputsrc2,"r");
    Jptf3 = fopen(inputsrc3,"r");
    Jptf4 = fopen(inputsrc4,"r");
    Jptf5 = fopen(inputsrc5,"r");
    for (inJi=0;inJi<6*Nxyz;inJi++)
      { 
        fscanf(Jptf1,"%lf",&Jdist1[inJi]);
        fscanf(Jptf2,"%lf",&Jdist2[inJi]);
        fscanf(Jptf3,"%lf",&Jdist3[inJi]);
        fscanf(Jptf4,"%lf",&Jdist4[inJi]);
        fscanf(Jptf5,"%lf",&Jdist5[inJi]);
      }
    fclose(Jptf1);
    fclose(Jptf2);
    fclose(Jptf3);
    fclose(Jptf4);
    fclose(Jptf5);
    ArrayToVec(Jdist1,J1);
    ArrayToVec(Jdist2,J2);
    ArrayToVec(Jdist3,J3);
    ArrayToVec(Jdist4,J4);
    ArrayToVec(Jdist5,J5);
    free(Jdist1);
    free(Jdist2);
    free(Jdist3);
    free(Jdist4);
    free(Jdist5);
  }
  VecScale(J1,Jmag1);
  VecScale(J2,Jmag2);
  VecScale(J3,Jmag3);
  VecScale(J4,Jmag4);
  VecScale(J5,Jmag5);

  ierr = VecDuplicate(vR,&b1);CHKERRQ(ierr);
  ierr = MatMult(D,J1,b1);CHKERRQ(ierr);
  VecScale(b1,omega1);

  ierr = VecDuplicate(vR,&b2);CHKERRQ(ierr);
  ierr = MatMult(D,J2,b2);CHKERRQ(ierr);
  VecScale(b2,omega2);

  ierr = VecDuplicate(vR,&b3);CHKERRQ(ierr);
  ierr = MatMult(D,J3,b3);CHKERRQ(ierr);
  VecScale(b3,omega3);

  ierr = VecDuplicate(vR,&b4);CHKERRQ(ierr);
  ierr = MatMult(D,J4,b4);CHKERRQ(ierr);
  VecScale(b4,omega4);

  ierr = VecDuplicate(vR,&b5);CHKERRQ(ierr);
  ierr = MatMult(D,J5,b5);CHKERRQ(ierr);
  VecScale(b5,omega5);

  VecDuplicate(vR,&weightedJ1);
  VecDuplicate(vR,&weightedJ2);
  VecDuplicate(vR,&weightedJ3);
  VecDuplicate(vR,&weightedJ4);
  VecDuplicate(vR,&weightedJ5);
  VecPointwiseMult(weightedJ1,J1,weight);
  VecPointwiseMult(weightedJ2,J2,weight);
  VecPointwiseMult(weightedJ3,J3,weight);
  VecPointwiseMult(weightedJ4,J4,weight);
  VecPointwiseMult(weightedJ5,J5,weight);

  //make VecPt
  Vec VecPt;
  VecDuplicate(vR,&VecPt);
  MakeVecPt(VecPt,Nx,Ny,Nz,ixref,iyref,izref,icref-1);

  /****set up x ****/
  Vec x1, x2, x3, x4, x5;
  ierr = VecDuplicate(vR,&x1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&x2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&x3);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&x4);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&x5);CHKERRQ(ierr);

  /*--------Create index sets for the vec scatter -------*/
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&to); CHKERRQ(ierr);

  /*--------Setup the KSP variables ---------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the KSP variables.--------\n ");CHKERRQ(ierr);
  KSP ksp1, ksp2, ksp3, ksp4, ksp5;
  PC pc1, pc2, pc3, pc4, pc5; 
  setupKSP(PETSC_COMM_WORLD,&ksp1,&pc1,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp2,&pc2,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp3,&pc3,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp4,&pc4,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp5,&pc5,solver,iteronly);
  KSP refksp1, refksp2, refksp3, refksp4, refksp5;
  PC refpc1, refpc2, refpc3, refpc4, refpc5; 
  setupKSP(PETSC_COMM_WORLD,&refksp1,&refpc1,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&refksp2,&refpc2,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&refksp3,&refpc3,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&refksp4,&refpc4,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&refksp5,&refpc5,solver,iteronly);
  int its1=100, its2=100, its3=100, its4=100, its5=100;
  int refits1=100, refits2=100, refits3=100, refits4=100, refits5=100;

  int optsuperpose=1;
  MetaSurfGroup meta1={Nx,Ny,Nz,hxyz, epsSReal,epsFReal, omega1, M1, A,b1,J1,x1,weightedJ1, epspmlQ1,epsmedium1,epsI  ,epscoef1, ksp1,&its1,refksp1,&refits1, metaphase1,optsuperpose,NULL,NULL, VecPt, outputbase,&filenameComm[0]};
  MetaSurfGroup meta2={Nx,Ny,Nz,hxyz, epsSReal,epsFReal, omega2, M2, A,b2,J2,x2,weightedJ2, epspmlQ2,epsmedium2,epsII ,epscoef2, ksp2,&its2,refksp2,&refits2, metaphase2,optsuperpose,NULL,NULL, VecPt, outputbase,&filenameComm[0]};
  MetaSurfGroup meta3={Nx,Ny,Nz,hxyz, epsSReal,epsFReal, omega3, M3, A,b3,J3,x3,weightedJ3, epspmlQ3,epsmedium3,epsIII,epscoef3, ksp3,&its3,refksp3,&refits3, metaphase3,optsuperpose,NULL,NULL, VecPt, outputbase,&filenameComm[0]};
  MetaSurfGroup meta4={Nx,Ny,Nz,hxyz, epsSReal,epsFReal, omega4, M4, A,b4,J4,x4,weightedJ4, epspmlQ4,epsmedium4,epsIV,epscoef4, ksp4,&its4,refksp4,&refits4, metaphase4,optsuperpose,NULL,NULL, VecPt, outputbase,&filenameComm[0]};
  MetaSurfGroup meta5={Nx,Ny,Nz,hxyz, epsSReal,epsFReal, omega5, M5, A,b5,J5,x5,weightedJ5, epspmlQ5,epsmedium5,epsV,epscoef5, ksp5,&its5,refksp5,&refits5, metaphase5,optsuperpose,NULL,NULL, VecPt, outputbase,&filenameComm[0]};

  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the KSP variables DONE!--------\n ");CHKERRQ(ierr);
  /*--------Setup the KSP variables DONE. ---------------*/

  /*--------Setup Helmholtz filter---------*/
  PC pcH;
  GetH1d(PETSC_COMM_WORLD,&Hfilt,DegFree,sH,nR,&kspH,&pcH);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the Hfilt DONE!--------\n ");CHKERRQ(ierr);
  /*--------Setup Helmholtz filter DONE---------*/

  /*---------Setup the epsopt and grad arrays----------------*/
  double *epsopt;
  FILE *ptf;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  ptf = fopen(initialdatafile,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");
  for (i=0;i<DegFree;i++)
    { 
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);

  double *grad;
  grad = (double *) malloc(DegFree*sizeof(double));

  /*---------Setup Done!---------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Everything set up! Ready to calculate the overlap and gradient.--------\n ");CHKERRQ(ierr);

/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/

  int Job;
  getint("-Job",&Job,1);

  if(Job==1){

    double frac;
    getreal("-penalfrac",&frac,1.0);

    double dummyvar;
    PetscOptionsGetReal(PETSC_NULL,"-dummyvar",&dummyvar,&flg);  MyCheckAndOutputDouble(flg,dummyvar,"dummyvar","Initial value of dummy variable t");
    int DegFreeAll=DegFree+1;
    double *epsoptAll;
    epsoptAll = (double *) malloc(DegFreeAll*sizeof(double));
    for (i=0;i<DegFree;i++){ epsoptAll[i]=epsopt[i]; }
    epsoptAll[DegFreeAll-1]=dummyvar;
  
    double mylb=0, myub=1.0;
    int maxeval, maxtime, mynloptalg;
    double maxf;
    nlopt_opt  opt;
    int mynloptlocalalg;
    nlopt_opt local_opt;
    nlopt_result result;

    PetscOptionsGetInt(PETSC_NULL,"-maxeval",&maxeval,&flg);  MyCheckAndOutputInt(flg,maxeval,"maxeval","max number of evaluation");
    PetscOptionsGetInt(PETSC_NULL,"-maxtime",&maxtime,&flg);  MyCheckAndOutputInt(flg,maxtime,"maxtime","max time of evaluation");
    PetscOptionsGetInt(PETSC_NULL,"-mynloptalg",&mynloptalg,&flg);  MyCheckAndOutputInt(flg,mynloptalg,"mynloptalg","The algorithm used ");
    PetscOptionsGetInt(PETSC_NULL,"-mynloptlocalalg",&mynloptlocalalg,&flg);  MyCheckAndOutputInt(flg,mynloptlocalalg,"mynloptlocalalg","The local optimization algorithm used ");

    double *lb=NULL, *ub=NULL;
    lb = (double *) malloc(DegFreeAll*sizeof(double));
    ub = (double *) malloc(DegFreeAll*sizeof(double));
    if(!readlubsfromfile) {
      for(i=0;i<DegFree;i++)
	{
	  lb[i] = mylb;
	  ub[i] = myub;
	}
    }else {
      char lbfile[PETSC_MAX_PATH_LEN], ubfile[PETSC_MAX_PATH_LEN];
      PetscOptionsGetString(PETSC_NULL,"-lbfile",lbfile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,lbfile,"lbfile","Lower-bound file");
      PetscOptionsGetString(PETSC_NULL,"-ubfile",ubfile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,ubfile,"ubfile","Upper-bound file");
      ptf = fopen(lbfile,"r");
      for (i=0;i<DegFree;i++)
	{ 
	  fscanf(ptf,"%lf",&lb[i]);
	}
      fclose(ptf);

      ptf = fopen(ubfile,"r");
      for (i=0;i<DegFree;i++)
	{ 
	  fscanf(ptf,"%lf",&ub[i]);
	}
      fclose(ptf);

    }
    //make sure that the pixels near boundaries are fixed
    int *fixedendptsarray;
    fixedendptsarray = (int *) malloc(nlayers*sizeof(int));
    int ilayer;
    int fixedendpts;
    getint("-fixedendpts",&fixedendpts,0);
    if(fixedendpts==0){
      for(ilayer=0;ilayer<nlayers;ilayer++){
	sprintf(tmpflg,"-fixed[%d]",ilayer+1);
	getint(tmpflg,fixedendptsarray+ilayer,5);
      }
    }else{
      for(ilayer=0;ilayer<nlayers;ilayer++){
	fixedendptsarray[ilayer]=fixedendpts;
      }
    }
    for(ilayer=0;ilayer<nlayers;ilayer++){
      for(i=0;i<fixedendptsarray[ilayer];i++){
	lb[ilayer*Mx+i]=0;
	lb[(ilayer+1)*Mx-1-i]=0;
	ub[ilayer*Mx+i]=0;
	ub[(ilayer+1)*Mx-1-i]=0;
      }
    }
    //make sure that the pixels near boundaries are fixed

    lb[DegFreeAll-1]=-3;
    ub[DegFreeAll-1]=3;

    opt = nlopt_create(mynloptalg, DegFreeAll);
    nlopt_set_lower_bounds(opt,lb);
    nlopt_set_upper_bounds(opt,ub);
    nlopt_set_maxeval(opt,maxeval);
    nlopt_set_maxtime(opt,maxtime);
    if (mynloptalg==11) nlopt_set_vector_storage(opt,4000);
    if (mynloptlocalalg)
      { 
	PetscPrintf(PETSC_COMM_WORLD,"-----------Running with a local optimizer.---------\n"); 
	local_opt=nlopt_create(mynloptlocalalg,DegFreeAll);
	nlopt_set_ftol_rel(local_opt, 1e-14);
	nlopt_set_maxeval(local_opt,100000);
	nlopt_set_local_optimizer(opt,local_opt);
      }

    int nummodes;
    getint("-nummodes",&nummodes,2);
    getreal("-mintrans",&mintrans,0);
    if(nummodes==1){
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta1,1e-8);
    }else if(nummodes==2){
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta1,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta2,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta2,1e-8);
    }else if(nummodes==3){
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta1,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta2,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta2,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta3,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta3,1e-8);
    }else if(nummodes==4){
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta1,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta2,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta2,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta3,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta3,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta4,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta4,1e-8);
    }else if(nummodes==5){
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta1,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta2,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta2,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta3,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta3,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta4,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta4,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta5,1e-8);
      if(mintrans>0) nlopt_add_inequality_constraint(opt,transmissionmetaconstr,&meta5,1e-8);
    }else{
      PetscPrintf(PETSC_COMM_WORLD,"Supply nummodes between 1 to 5 \n");
    }     

    if(frac<1.0) nlopt_add_inequality_constraint(opt,pfunc,&frac,1e-8);
    nlopt_set_max_objective(opt,minimaxobjfun,NULL);   
    
    result = nlopt_optimize(opt,epsoptAll,&maxf);

    PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);

    PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);

  }

  if(Job==2){

    /*---------Calculate the overlap and gradient--------*/
    double beta=0;
    double s1, ds, s2, epscen;
    int posMj;
    getint("-posMj",&posMj,0);
    getreal("-s1",&s1,0);
    getreal("-s2",&s2,1);
    getreal("-ds",&ds,0.01);
    for (epscen=s1;epscen<s2;epscen+=ds)
      {
        epsopt[posMj]=epscen;
        beta = metasurface(DegFree,epsopt,grad,&meta1);
        PetscPrintf(PETSC_COMM_WORLD,"epscen: %g objfunc: %g objfunc-grad: %g \n", epsopt[posMj], beta, grad[posMj]);
      }

  }





  ///Job 3 Printing

  if(Job==3){

    /*
    metasurface(DegFree,epsopt,grad,&meta1);
    OutputVec(PETSC_COMM_WORLD,meta1.x,"exmField1",".m");
    metasurface(DegFree,epsopt,grad,&meta2);
    OutputVec(PETSC_COMM_WORLD,meta2.x,"exmField2",".m");
    metasurface(DegFree,epsopt,grad,&meta3);
    OutputVec(PETSC_COMM_WORLD,meta3.x,"exmField3",".m");
    */
    metasurface(DegFree,epsopt,grad,&meta4);
    OutputVec(PETSC_COMM_WORLD,meta4.x,"exmField4",".m");
    metasurface(DegFree,epsopt,grad,&meta5);
    OutputVec(PETSC_COMM_WORLD,meta5.x,"exmField5",".m");

    Vec eps1Full, eps2Full, eps3Full, eps4Full, eps5Full;
    VecDuplicate(vR,&eps1Full);
    VecDuplicate(vR,&eps2Full);
    VecDuplicate(vR,&eps3Full);
    VecDuplicate(vR,&eps4Full);
    VecDuplicate(vR,&eps5Full);


    ArrayToVec(epsopt,epsSReal);
    MatMult(A,epsSReal,epsFReal);

    VecPointwiseMult(eps1Full,epsFReal,epsI);
    VecAXPY(eps1Full,1.0,epsmedium1);
    VecPointwiseMult(eps1Full,eps1Full,vR);
    OutputVec(PETSC_COMM_WORLD, eps1Full, "eps1Full",".m");

    VecPointwiseMult(eps2Full,epsFReal,epsII);
    VecAXPY(eps2Full,1.0,epsmedium2);
    VecPointwiseMult(eps2Full,eps2Full,vR);
    OutputVec(PETSC_COMM_WORLD, eps2Full, "eps2Full",".m");

    VecPointwiseMult(eps3Full,epsFReal,epsIII);
    VecAXPY(eps3Full,1.0,epsmedium3);
    VecPointwiseMult(eps3Full,eps3Full,vR);
    OutputVec(PETSC_COMM_WORLD, eps3Full, "eps3Full",".m");

    VecPointwiseMult(eps4Full,epsFReal,epsIV);
    VecAXPY(eps4Full,1.0,epsmedium4);
    VecPointwiseMult(eps4Full,eps4Full,vR);
    OutputVec(PETSC_COMM_WORLD, eps4Full, "eps4Full",".m");

    VecPointwiseMult(eps5Full,epsFReal,epsV);
    VecAXPY(eps5Full,1.0,epsmedium5);
    VecPointwiseMult(eps5Full,eps5Full,vR);
    OutputVec(PETSC_COMM_WORLD, eps5Full, "eps5Full",".m");

    VecDestroy(&eps1Full);
    VecDestroy(&eps2Full);
    VecDestroy(&eps3Full);
    VecDestroy(&eps4Full);
    VecDestroy(&eps5Full);

    OutputVec(PETSC_COMM_WORLD,J1,"J",".m");
    OutputVec(PETSC_COMM_WORLD,VecPt,"VecPt",".m");

    double *tmpepsopt;
    tmpepsopt = (double *) malloc(DegFree*sizeof(double));
    for (i=0;i<DegFree;i++){
      tmpepsopt[i]=0;
    };
    metasurface(DegFree,tmpepsopt,grad,&meta1);
    OutputVec(PETSC_COMM_WORLD,meta1.x,"refField1",".m");
    metasurface(DegFree,tmpepsopt,grad,&meta2);
    OutputVec(PETSC_COMM_WORLD,meta2.x,"refField2",".m");
    metasurface(DegFree,tmpepsopt,grad,&meta3);
    OutputVec(PETSC_COMM_WORLD,meta3.x,"refField3",".m");



  }












  ///Job 3 Printing Done


  if(Job==4){

    /*---------Optimization--------*/
    double frac;
    getreal("-penalfrac",&frac,1.0);

    double mylb=0, myub=1.0;
    int maxeval, maxtime, mynloptalg;
    double maxf;
    nlopt_opt  opt;
    int mynloptlocalalg;
    nlopt_opt local_opt;
    nlopt_result result;

    PetscOptionsGetInt(PETSC_NULL,"-maxeval",&maxeval,&flg);  MyCheckAndOutputInt(flg,maxeval,"maxeval","max number of evaluation");
    PetscOptionsGetInt(PETSC_NULL,"-maxtime",&maxtime,&flg);  MyCheckAndOutputInt(flg,maxtime,"maxtime","max time of evaluation");
    PetscOptionsGetInt(PETSC_NULL,"-mynloptalg",&mynloptalg,&flg);  MyCheckAndOutputInt(flg,mynloptalg,"mynloptalg","The algorithm used ");
    PetscOptionsGetInt(PETSC_NULL,"-mynloptlocalalg",&mynloptlocalalg,&flg);  MyCheckAndOutputInt(flg,mynloptlocalalg,"mynloptlocalalg","The local optimization algorithm used ");

    double *lb=NULL, *ub=NULL;
    lb = (double *) malloc(DegFree*sizeof(double));
    ub = (double *) malloc(DegFree*sizeof(double));
    for(i=0;i<DegFree;i++)
      {
	lb[i] = 0;
	ub[i] = 1;
      }
    //make sure that the pixels near boundaries are fixed
    int fixedendpts;
    getint("-fixedendpts",&fixedendpts,5);
    for(i=0;i<fixedendpts;i++){
      lb[i]=0;
      lb[DegFree-i-1]=0;
      ub[i]=0;
      ub[DegFree-i-1]=0;
    }
    //make sure that the pixels near boundaries are fixed
    
    opt = nlopt_create(mynloptalg, DegFree);
    nlopt_set_lower_bounds(opt,lb);
    nlopt_set_upper_bounds(opt,ub);
    nlopt_set_maxeval(opt,maxeval);
    nlopt_set_maxtime(opt,maxtime);
    if (mynloptalg==11) nlopt_set_vector_storage(opt,4000);
    if (mynloptlocalalg)
      { 
	PetscPrintf(PETSC_COMM_WORLD,"-----------Running with a local optimizer.---------\n"); 
	local_opt=nlopt_create(mynloptlocalalg,DegFree);
	nlopt_set_ftol_rel(local_opt, 1e-14);
	nlopt_set_maxeval(local_opt,100000);
	nlopt_set_local_optimizer(opt,local_opt);
      }

    if(frac<1.0) nlopt_add_inequality_constraint(opt,pfunc2,&frac,1e-8);
    nlopt_set_max_objective(opt,metasurface,&meta1);   

    result = nlopt_optimize(opt,epsopt,&maxf);

    PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);

    PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);
    
  }

/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

/* ----------------------Destroy Vecs and Mats----------------------------*/ 
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = MatDestroy(&B1); CHKERRQ(ierr);
  ierr = MatDestroy(&B2); CHKERRQ(ierr);
  ierr = MatDestroy(&C); CHKERRQ(ierr);
  ierr = MatDestroy(&D); CHKERRQ(ierr);
  ierr = MatDestroy(&M1); CHKERRQ(ierr);  
  ierr = MatDestroy(&M2); CHKERRQ(ierr);
  ierr = MatDestroy(&M3); CHKERRQ(ierr);
  ierr = MatDestroy(&M4); CHKERRQ(ierr);
  ierr = MatDestroy(&M5); CHKERRQ(ierr);
  ierr = MatDestroy(&Hfilt); CHKERRQ(ierr);

  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&weight); CHKERRQ(ierr);

  ierr = VecDestroy(&unitx); CHKERRQ(ierr);
  ierr = VecDestroy(&unity); CHKERRQ(ierr);
  ierr = VecDestroy(&unitz); CHKERRQ(ierr);
  ierr = VecDestroy(&epsI); CHKERRQ(ierr);
  ierr = VecDestroy(&epsII); CHKERRQ(ierr);
  ierr = VecDestroy(&epsIII); CHKERRQ(ierr);
  ierr = VecDestroy(&epsIV); CHKERRQ(ierr);
  ierr = VecDestroy(&epsV); CHKERRQ(ierr);

  ierr = VecDestroy(&muinvpml1); CHKERRQ(ierr);
  ierr = VecDestroy(&muinvpml2); CHKERRQ(ierr);
  ierr = VecDestroy(&muinvpml3); CHKERRQ(ierr);
  ierr = VecDestroy(&muinvpml4); CHKERRQ(ierr);
  ierr = VecDestroy(&muinvpml5); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml1); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml2); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml3); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml4); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml5); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ1); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ2); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ3); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ4); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ5); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef1); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef2); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef3); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef4); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef5); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium1); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium2); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium3); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium4); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium5); CHKERRQ(ierr);
  ierr = VecDestroy(&epsSReal); CHKERRQ(ierr);
  ierr = VecDestroy(&epsFReal); CHKERRQ(ierr);
  ierr = VecDestroy(&vgrad); CHKERRQ(ierr);  
  ierr = VecDestroy(&vgradlocal); CHKERRQ(ierr);

  ierr = VecDestroy(&x1); CHKERRQ(ierr);
  ierr = VecDestroy(&x2); CHKERRQ(ierr);
  ierr = VecDestroy(&x3); CHKERRQ(ierr);
  ierr = VecDestroy(&x4); CHKERRQ(ierr);
  ierr = VecDestroy(&x5); CHKERRQ(ierr);
  ierr = VecDestroy(&b1);CHKERRQ(ierr);
  ierr = VecDestroy(&b2); CHKERRQ(ierr);
  ierr = VecDestroy(&b3); CHKERRQ(ierr);
  ierr = VecDestroy(&b4); CHKERRQ(ierr);
  ierr = VecDestroy(&b5); CHKERRQ(ierr);
  ierr = VecDestroy(&J1); CHKERRQ(ierr);
  ierr = VecDestroy(&J2); CHKERRQ(ierr);
  ierr = VecDestroy(&J3); CHKERRQ(ierr);
  ierr = VecDestroy(&J4); CHKERRQ(ierr);
  ierr = VecDestroy(&J5); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ1); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ2); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ3); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ4); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ5); CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp1);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp2);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp3);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp4);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp5);CHKERRQ(ierr);
  ierr = KSPDestroy(&refksp1);CHKERRQ(ierr);
  ierr = KSPDestroy(&refksp2);CHKERRQ(ierr);
  ierr = KSPDestroy(&refksp3);CHKERRQ(ierr);
  ierr = KSPDestroy(&refksp4);CHKERRQ(ierr);
  ierr = KSPDestroy(&refksp5);CHKERRQ(ierr);

  ierr = KSPDestroy(&kspH);CHKERRQ(ierr);

  ISDestroy(&from);
  ISDestroy(&to);

  free(muinv1);
  free(muinv2);
  free(muinv3);
  free(muinv4);
  free(muinv5);
  free(epsopt);
  free(grad);

  /*------------ finalize the program -------------*/

  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Barrier(PETSC_COMM_WORLD);
  }
  
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}

double pfunc(int DegFree, double *epsopt, double *grad, void *data)
{
  int i;
  double sumeps;
  double max=DegFree/4;
  double *tmp  = (double *) data;
  double frac= *tmp;

  sumeps=0.0;
  for (i=0;i<DegFree-1;i++){
    sumeps+=fabs(epsopt[i]*(1-epsopt[i]));
    grad[i]=1-2*epsopt[i];
  }
  grad[DegFree-1]=0;

  PetscPrintf(PETSC_COMM_WORLD,"******the current binaryindex is %1.6e \n",sumeps);
  PetscPrintf(PETSC_COMM_WORLD,"******the current binaryexcess  is %1.6e \n",sumeps-frac*max);

  return sumeps - frac*max;
}

double pfunc2(int DegFree, double *epsopt, double *grad, void *data)
{
  int i;
  double sumeps;
  double max=DegFree/4;
  double *tmp  = (double *) data;
  double frac= *tmp;

  sumeps=0.0;
  for (i=0;i<DegFree;i++){
    sumeps+=fabs(epsopt[i]*(1-epsopt[i]));
    grad[i]=1-2*epsopt[i];
  }

  PetscPrintf(PETSC_COMM_WORLD,"******the current binaryindex is %1.6e \n",sumeps);
  PetscPrintf(PETSC_COMM_WORLD,"******the current binaryexcess  is %1.6e \n",sumeps-frac*max);

  return sumeps - frac*max;
}

PetscErrorCode setupKSP(MPI_Comm comm, KSP *kspout, PC *pcout, int solver, int iteronly)
{
  PetscErrorCode ierr;
  KSP ksp;
  PC pc; 
  
  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  if (solver==0) {
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERPASTIX);CHKERRQ(ierr);
  }
  else if (solver==1){
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
  }
  else {
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU_DIST);CHKERRQ(ierr);
  }
  ierr = KSPSetTolerances(ksp,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxit);CHKERRQ(ierr);

  if (iteronly==1){
  ierr = KSPSetType(ksp, KSPLSQR);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPMonitorSet(ksp,KSPMonitorTrueResidualNorm,NULL,0);CHKERRQ(ierr);
  }

  ierr = PCSetFromOptions(pc);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  *kspout=ksp;
  *pcout=pc;

  PetscFunctionReturn(0);

}

