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

/*------------------------------------------------------*/

PetscErrorCode setupKSP(MPI_Comm comm, KSP *ksp, PC *pc, int solver, int iteronly);
double pfunc(int DegFree, double *epsopt, double *grad, void *data);

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

  int blochcondition;
  double beta1[3]={0,0,0}, beta2[3]={0,0,0}, beta3[3]={0,0,0};
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

  int Mx, My, Mz, Mzslab, Nx, Ny, Nz, Npmlx, Npmly, Npmlz, Nxyz, DegFree;
  double hx, hy, hz, hxyz;
  int cx, cy, cz;
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

  int LowerPMLx, LowerPMLy, LowerPMLz;
  getint("-LowerPMLx",&LowerPMLx,1);
  getint("-LowerPMLy",&LowerPMLy,1);
  getint("-LowerPMLz",&LowerPMLz,1);

  int Lz;
  getint("-Lz",&Lz,0);

  int DegFree1, DegFree2;
  Nxyz=Nx*Ny*Nz;
  DegFree1 = Mx*My*((Mzslab==0)?Mz:1);
  DegFree2 = Lz;
  DegFree = DegFree1 + DegFree2;

  int b1x[2], b1y[2], b1z[2], b2x[2], b2y[2], b2z[2], b3x[2], b3y[2], b3z[2];
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

  PetscOptionsGetInt(PETSC_NULL,"-cx",&cx,&flg);  MyCheckAndOutputInt(flg,cx,"cx","cx");
  PetscOptionsGetInt(PETSC_NULL,"-cy",&cy,&flg);  MyCheckAndOutputInt(flg,cy,"cy","cy");
  PetscOptionsGetInt(PETSC_NULL,"-cz",&cz,&flg);  MyCheckAndOutputInt(flg,cz,"cz","cz");  

  int outputbase;
  PetscOptionsGetInt(PETSC_NULL,"-outputbase",&outputbase,&flg); MyCheckAndOutputInt(flg,outputbase,"outputbase","outputbase");

  int J1dir, J2dir, J3dir;
  PetscOptionsGetInt(PETSC_NULL,"-J1dir",&J1dir,&flg);  MyCheckAndOutputInt(flg,J1dir,"J1dir","J1dir");
  PetscOptionsGetInt(PETSC_NULL,"-J2dir",&J2dir,&flg);  MyCheckAndOutputInt(flg,J2dir,"J2dir","J2dir");
  PetscOptionsGetInt(PETSC_NULL,"-J3dir",&J3dir,&flg);  MyCheckAndOutputInt(flg,J3dir,"J3dir","J3dir");

  double freq1, freq2, freq3, omega1, omega2, omega3;
  getreal("-freq1",&freq1,1.0);
  getreal("-freq2",&freq2,1.0);
  getreal("-freq3",&freq3,freq1+freq2);
  omega1=2.0*PI*freq1;
  omega2=2.0*PI*freq2;
  omega3=2.0*PI*freq3;

  double eps1diffU, eps1diffM, eps1diffL, eps2diffU, eps2diffM, eps2diffL, eps3diffU, eps3diffM, eps3diffL;
  PetscOptionsGetReal(PETSC_NULL,"-eps1diffU",&eps1diffU,&flg); MyCheckAndOutputDouble(flg,eps1diffU,"eps1diffU","eps1diffU");
  PetscOptionsGetReal(PETSC_NULL,"-eps1diffM",&eps1diffM,&flg); MyCheckAndOutputDouble(flg,eps1diffM,"eps1diffM","eps1diffM");
  PetscOptionsGetReal(PETSC_NULL,"-eps1diffL",&eps1diffL,&flg); MyCheckAndOutputDouble(flg,eps1diffL,"eps1diffL","eps1diffL");
  PetscOptionsGetReal(PETSC_NULL,"-eps2diffU",&eps2diffU,&flg); MyCheckAndOutputDouble(flg,eps2diffU,"eps2diffU","eps2diffU");
  PetscOptionsGetReal(PETSC_NULL,"-eps2diffM",&eps2diffM,&flg); MyCheckAndOutputDouble(flg,eps2diffM,"eps2diffM","eps2diffM");
  PetscOptionsGetReal(PETSC_NULL,"-eps2diffL",&eps2diffL,&flg); MyCheckAndOutputDouble(flg,eps2diffL,"eps2diffL","eps2diffL");
  PetscOptionsGetReal(PETSC_NULL,"-eps3diffU",&eps3diffU,&flg); MyCheckAndOutputDouble(flg,eps3diffU,"eps3diffU","eps3diffU");
  PetscOptionsGetReal(PETSC_NULL,"-eps3diffM",&eps3diffM,&flg); MyCheckAndOutputDouble(flg,eps3diffM,"eps3diffM","eps3diffM");
  PetscOptionsGetReal(PETSC_NULL,"-eps3diffL",&eps3diffL,&flg); MyCheckAndOutputDouble(flg,eps3diffL,"eps3diffL","eps3diffL");

  double eps1bkgU, eps1bkgM, eps1bkgL, eps2bkgU, eps2bkgM, eps2bkgL, eps3bkgU, eps3bkgM, eps3bkgL;
  PetscOptionsGetReal(PETSC_NULL,"-eps1bkgU",&eps1bkgU,&flg); MyCheckAndOutputDouble(flg,eps1bkgU,"eps1bkgU","eps1bkgU");
  PetscOptionsGetReal(PETSC_NULL,"-eps1bkgM",&eps1bkgM,&flg); MyCheckAndOutputDouble(flg,eps1bkgM,"eps1bkgM","eps1bkgM");
  PetscOptionsGetReal(PETSC_NULL,"-eps1bkgL",&eps1bkgL,&flg); MyCheckAndOutputDouble(flg,eps1bkgL,"eps1bkgL","eps1bkgL");
  PetscOptionsGetReal(PETSC_NULL,"-eps2bkgU",&eps2bkgU,&flg); MyCheckAndOutputDouble(flg,eps2bkgU,"eps2bkgU","eps2bkgU");
  PetscOptionsGetReal(PETSC_NULL,"-eps2bkgM",&eps2bkgM,&flg); MyCheckAndOutputDouble(flg,eps2bkgM,"eps2bkgM","eps2bkgM");
  PetscOptionsGetReal(PETSC_NULL,"-eps2bkgL",&eps2bkgL,&flg); MyCheckAndOutputDouble(flg,eps2bkgL,"eps2bkgL","eps2bkgL");
  PetscOptionsGetReal(PETSC_NULL,"-eps3bkgU",&eps3bkgU,&flg); MyCheckAndOutputDouble(flg,eps3bkgU,"eps3bkgU","eps3bkgU");
  PetscOptionsGetReal(PETSC_NULL,"-eps3bkgM",&eps3bkgM,&flg); MyCheckAndOutputDouble(flg,eps3bkgM,"eps3bkgM","eps3bkgM");
  PetscOptionsGetReal(PETSC_NULL,"-eps3bkgL",&eps3bkgL,&flg); MyCheckAndOutputDouble(flg,eps3bkgL,"eps3bkgL","eps3bkgL");

  int Nxo,Nyo,Nzo;
  getint("-Nxo",&Nxo,LowerPMLx*(Nx-Mx)/2);
  getint("-Nyo",&Nyo,LowerPMLy*(Ny-My)/2);
  getint("-Nzo",&Nzo,LowerPMLz*(Nz-Mz)/2);

  PetscPrintf(PETSC_COMM_WORLD,"omegas, DegFree: %g, %g, %g, %d \n", omega1, omega2, omega3, DegFree);

  double Qabs;
  PetscOptionsGetReal(PETSC_NULL,"-Qabs",&Qabs,&flg);  MyCheckAndOutputDouble(flg,Qabs,"Qabs","Qabs");
  if (Qabs>1e15) Qabs=1.0/0.0;

  double Jmag1, Jmag2;
  getreal("-Jmag1",&Jmag1,1.0);
  getreal("-Jmag2",&Jmag2,1.0);

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
  boosterinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, Nxo,Nyo,Nzo, Mx,My,Mz,Mzslab, Lz, 0);

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
  Mat M1, M2, M3;
  Vec muinvpml1, muinvpml2, muinvpml3;
  MuinvPMLGeneral(PETSC_COMM_SELF, &muinvpml1,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega1,LowerPMLx,LowerPMLy,LowerPMLz);
  MuinvPMLGeneral(PETSC_COMM_SELF, &muinvpml2,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega2,LowerPMLx,LowerPMLy,LowerPMLz); 
  MuinvPMLGeneral(PETSC_COMM_SELF, &muinvpml3,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega3,LowerPMLx,LowerPMLy,LowerPMLz); 
  double *muinv1, *muinv2, *muinv3;
  muinv1 = (double *) malloc(sizeof(double)*6*Nxyz);
  muinv2 = (double *) malloc(sizeof(double)*6*Nxyz);
  muinv3 = (double *) malloc(sizeof(double)*6*Nxyz);
  int add=1;
  AddMuAbsorption(muinv1,muinvpml1,Qabs,add);
  AddMuAbsorption(muinv2,muinvpml2,Qabs,add);
  AddMuAbsorption(muinv3,muinvpml3,Qabs,add);

  if(blochcondition){
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M1, Nx,Ny,Nz, hx,hy,hz, b1x,b1y,b1z, muinv1, BCPeriod, beta1);
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M2, Nx,Ny,Nz, hx,hy,hz, b2x,b2y,b2z, muinv2, BCPeriod, beta2);
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M3, Nx,Ny,Nz, hx,hy,hz, b3x,b3y,b3z, muinv3, BCPeriod, beta3);
  }else{
    MoperatorGeneral(PETSC_COMM_WORLD, &M1, Nx,Ny,Nz, hx,hy,hz, b1x,b1y,b1z, muinv1, BCPeriod);
    MoperatorGeneral(PETSC_COMM_WORLD, &M2, Nx,Ny,Nz, hx,hy,hz, b2x,b2y,b2z, muinv2, BCPeriod);
    MoperatorGeneral(PETSC_COMM_WORLD, &M3, Nx,Ny,Nz, hx,hy,hz, b3x,b3y,b3z, muinv3, BCPeriod);
  }
  ierr = PetscObjectSetName((PetscObject) M1, "M1"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) M2, "M2"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) M3, "M3"); CHKERRQ(ierr);

  /*----Set up the epsilon PML vectors--------*/
  Vec unitx,unity,unitz;
  ierr = VecDuplicate(vR,&unitx);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unity);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unitz);CHKERRQ(ierr);
  GetUnitVec(unitx,0,6*Nxyz);
  GetUnitVec(unity,1,6*Nxyz);
  GetUnitVec(unitz,2,6*Nxyz);

  Vec epsI, epsII, epsIII;
  PetscPrintf(PETSC_COMM_WORLD,"***NOTE: Anisotropic epsilon not supported in this version. \n");
  ierr = VecDuplicate(vR,&epsI);  CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsII); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsIII);CHKERRQ(ierr);

  VecSet(epsI,0.0);
  makethreelayeredepsdiff(epsI,   Nx,Ny,Nz, Nzo, Mz, eps1diffU, eps1diffM, eps1diffL);
  VecSet(epsII,0.0); 
  makethreelayeredepsdiff(epsII,  Nx,Ny,Nz, Nzo, Mz, eps2diffU, eps2diffM, eps2diffL);
  VecSet(epsIII,0.0);
  makethreelayeredepsdiff(epsIII, Nx,Ny,Nz, Nzo, Mz, eps3diffU, eps3diffM, eps3diffL);

  Vec epspml1, epspml2, epspml3, epspmlQ1, epspmlQ2, epspmlQ3, epscoef1, epscoef2, epscoef3;
  ierr = VecDuplicate(vR,&epspml1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml3);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ3);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef3);CHKERRQ(ierr);

  EpsPMLGeneral(PETSC_COMM_WORLD, epspml1,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega1, LowerPMLx,LowerPMLy,LowerPMLz);
  EpsPMLGeneral(PETSC_COMM_WORLD, epspml2,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega2, LowerPMLx,LowerPMLy,LowerPMLz);
  EpsPMLGeneral(PETSC_COMM_WORLD, epspml3,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega3, LowerPMLx,LowerPMLy,LowerPMLz);

  EpsCombine(D, weight, epspml1, epspmlQ1, epscoef1, Qabs, omega1, epsI);
  EpsCombine(D, weight, epspml2, epspmlQ2, epscoef2, Qabs, omega2, epsII);
  EpsCombine(D, weight, epspml3, epspmlQ3, epscoef3, Qabs, omega3, epsIII);

  /*-----Set up epsmedium, epsSReal, epsFReal, epsC, epsCi, epsP, vgrad, vgradlocal ------*/
  Vec epsmedium1, epsmedium2, epsmedium3;
  ierr = VecDuplicate(vR,&epsmedium1); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsmedium2); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsmedium3); CHKERRQ(ierr);
  VecSet(epsmedium1,0.0);
  makethreelayeredepsbkg(epsmedium1, Nx,Ny,Nz, Nzo, Mz, eps1bkgU, eps1bkgM, eps1bkgL);
  VecSet(epsmedium2,0.0); 
  makethreelayeredepsbkg(epsmedium2, Nx,Ny,Nz, Nzo, Mz, eps2bkgU, eps2bkgM, eps2bkgL);
  VecSet(epsmedium3,0.0);
  makethreelayeredepsbkg(epsmedium3, Nx,Ny,Nz, Nzo, Mz, eps3bkgU, eps3bkgM, eps3bkgL);

  Vec epsSReal, vgrad;
  ierr = MatCreateVecs(A,&epsSReal, &epsFReal); CHKERRQ(ierr);
  ierr = VecDuplicate(epsSReal, &vgrad); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) epsmedium1,  "epsmedium1");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsmedium2,  "epsmedium2");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsmedium3,  "epsmedium3");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsFReal, "epsFReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vgrad, "vgrad");CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, DegFree, &vgradlocal); CHKERRQ(ierr);

  /*---------Set up J, b and weightedJ-------------*/
  Vec J1, J2, b1, b2, weightedJ1, weightedJ2;

  ierr = VecDuplicate(vR,&J1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) J1, "Source1");CHKERRQ(ierr);
  VecSet(J1,0.0);
  if (J1dir == 1)
    SourceSingleSetX(PETSC_COMM_WORLD, J1, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else if (J1dir ==2)
    SourceSingleSetY(PETSC_COMM_WORLD, J1, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else if (J1dir == 3)
    SourceSingleSetZ(PETSC_COMM_WORLD, J1, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else
    PetscPrintf(PETSC_COMM_WORLD," Please specify correct direction of current J1: x (1) , y (2) or z (3)\n "); 

  ierr = VecDuplicate(vR,&J2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) J2, "Source2");CHKERRQ(ierr);
  VecSet(J2,0.0);
  if (J2dir == 1)
    SourceSingleSetX(PETSC_COMM_WORLD, J2, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else if (J2dir ==2)
    SourceSingleSetY(PETSC_COMM_WORLD, J2, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else if (J2dir == 3)
    SourceSingleSetZ(PETSC_COMM_WORLD, J2, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else
    PetscPrintf(PETSC_COMM_WORLD," Please specify correct direction of current J2: x (1) , y (2) or z (3)\n "); 

  int Jopt;
  getint("-Jopt",&Jopt,0);

  if(Jopt==1){
    char inputsrc1[PETSC_MAX_PATH_LEN], inputsrc2[PETSC_MAX_PATH_LEN];
    PetscOptionsGetString(PETSC_NULL,"-inputsrc1",inputsrc1,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrc1,"inputsrc1","Input source current 1");
    PetscOptionsGetString(PETSC_NULL,"-inputsrc2",inputsrc2,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrc2,"inputsrc2","Input source current 2");

    double *Jdist1, *Jdist2;
    FILE *Jptf1, *Jptf2;
    int inJi;
    Jdist1 = (double *) malloc(6*Nxyz*sizeof(double));
    Jdist2 = (double *) malloc(6*Nxyz*sizeof(double));

    Jptf1 = fopen(inputsrc1,"r");
    Jptf2 = fopen(inputsrc2,"r");
    for (inJi=0;inJi<6*Nxyz;inJi++)
      { 
        fscanf(Jptf1,"%lf",&Jdist1[inJi]);
        fscanf(Jptf2,"%lf",&Jdist2[inJi]);
      }
    fclose(Jptf1);
    fclose(Jptf2);
    ArrayToVec(Jdist1,J1);
    ArrayToVec(Jdist2,J2);
    free(Jdist1);
    free(Jdist2);
  }
  VecScale(J1,Jmag1);
  VecScale(J2,Jmag2);

  ierr = VecDuplicate(vR,&b1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) b1, "rhs1");CHKERRQ(ierr);
  ierr = MatMult(D,J1,b1);CHKERRQ(ierr);
  VecScale(b1,omega1);
  ierr = VecDuplicate(vR,&weightedJ1); CHKERRQ(ierr);
  ierr = VecPointwiseMult(weightedJ1,J1,weight);
  ierr = PetscObjectSetName((PetscObject) weightedJ1, "weightedJ1");CHKERRQ(ierr);

  ierr = VecDuplicate(vR,&b2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) b2, "rhs2");CHKERRQ(ierr);
  ierr = MatMult(D,J2,b2);CHKERRQ(ierr);
  VecScale(b2,omega2);
  ierr = VecDuplicate(vR,&weightedJ2); CHKERRQ(ierr);
  ierr = VecPointwiseMult(weightedJ2,J2,weight);
  ierr = PetscObjectSetName((PetscObject) weightedJ2, "weightedJ2");CHKERRQ(ierr);

  /****set up x ****/
  Vec x1, x2;
  ierr = VecDuplicate(vR,&x1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&x2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x1, "E1");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x2, "E2");CHKERRQ(ierr);

  /*--------Create index sets for the vec scatter -------*/
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&to); CHKERRQ(ierr);

  /*--------Setup the KSP variables ---------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the KSP variables.--------\n ");CHKERRQ(ierr);
  KSP ksp1, ksp2, ksp3;
  PC pc1, pc2, pc3; 
  setupKSP(PETSC_COMM_WORLD,&ksp1,&pc1,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp2,&pc2,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp3,&pc3,solver,iteronly);

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
  int i;
  for (i=0;i<DegFree;i++)
    { 
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);

  //DEBUG
  ArrayToVec(epsopt,epsSReal);
  MatMult(A,epsSReal,epsFReal);
  VecPointwiseMult(epsFReal,epsFReal,epsI);
  VecAXPY(epsFReal,1.0,epsmedium1);
  OutputVec(PETSC_COMM_WORLD, epsFReal, "initial_","epsF.m");

  double *grad;
  grad = (double *) malloc(DegFree*sizeof(double));

  /*---------Setup Done!---------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Everything set up! Ready to calculate the overlap and gradient.--------\n ");CHKERRQ(ierr);


/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/

  double p1, p2;
  getreal("-p1",&p1,0);
  getreal("-p2",&p2,0);
  int its1=100, its2=100, its3=100;

  SFGdataGroup sfgdata={Nx,Ny,Nz,hxyz,epsSReal,epsFReal,omega1,omega2,omega3,M1,M2,M3,A,b1,b2,x1,x2,weightedJ1,weightedJ2,epspmlQ1,epspmlQ2,epspmlQ3,epsmedium1,epsmedium2,epsmedium3,epsI,epsII,epsIII,epscoef1,epscoef2,epscoef3,B1,B2,ksp1,ksp2,ksp3,&its1,&its2,&its3,p1,p2,outputbase};

  int Job;
  getint("-Job",&Job,1);

if (Job==1){

  PetscPrintf(PETSC_COMM_WORLD,"-----------Begin setting up the optimization problem.-----\n"); 
  double frac;
  getreal("-penalfrac",&frac,1.0);

  /*---------Optimization--------*/
  double mylb=0, myub=1.0, *lb=NULL, *ub=NULL;
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

  lb = (double *) malloc(DegFree*sizeof(double));
  ub = (double *) malloc(DegFree*sizeof(double));
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

  if(frac<1.0) nlopt_add_inequality_constraint(opt,pfunc,&frac,1e-8);

  int singleldos;
  getint("-singleldos",&singleldos,0);
  if(singleldos==0)
    nlopt_set_max_objective(opt,sfg_arbitraryPol,&sfgdata);
  else
    nlopt_set_max_objective(opt,sfg_singleldos,&sfgdata);

  result = nlopt_optimize(opt,epsopt,&maxf);			//********************optimization here!********************//

  if (result < 0) 
    PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);
  else 
    PetscPrintf(PETSC_COMM_WORLD,"found extremum  %0.16e\n",maxf); 

  PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);

  free(lb);
  free(ub);
  nlopt_destroy(opt);

  int rankA;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rankA);

  if(rankA==0)
    {
      ptf = fopen(strcat(filenameComm,"epsopt.txt"),"w");
        for (i=0;i<DegFree;i++)
          fprintf(ptf,"%0.16e \n",epsopt[i]);
	  fclose(ptf);
    } 

}

if (Job==2){

  /*---------Calculate the overlap and gradient--------*/
  int px, py, pz=0;
  double beta=0;
  double s1, ds, s2, epscen;

  getint("-px",&px,0);
  getint("-py",&py,0);
  getint("-pz",&pz,0);
  getreal("-s1",&s1,0);
  getreal("-s2",&s2,1);
  getreal("-ds",&ds,0.1);
  
  int posMj=(px*My+ py)*(Mzslab==0?Mz:1) + pz*(1-Mzslab);
  for (epscen=s1;epscen<s2;epscen+=ds)
  { 
      epsopt[posMj]=epscen; 
      beta = sfg_arbitraryPol(DegFree,epsopt,grad,&sfgdata);
      PetscPrintf(PETSC_COMM_WORLD,"epscen: %g objfunc: %g objfunc-grad: %g \n", epsopt[posMj], beta, grad[posMj]);
  }


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
  ierr = MatDestroy(&Hfilt); CHKERRQ(ierr);

  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&weight); CHKERRQ(ierr);

  ierr = VecDestroy(&unitx); CHKERRQ(ierr);
  ierr = VecDestroy(&unity); CHKERRQ(ierr);
  ierr = VecDestroy(&unitz); CHKERRQ(ierr);
  ierr = VecDestroy(&epsI); CHKERRQ(ierr);
  ierr = VecDestroy(&epsII); CHKERRQ(ierr);
  ierr = VecDestroy(&epsIII); CHKERRQ(ierr);

  ierr = VecDestroy(&muinvpml1); CHKERRQ(ierr);
  ierr = VecDestroy(&muinvpml2); CHKERRQ(ierr);
  ierr = VecDestroy(&muinvpml3); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml1); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml2); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml3); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ1); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ2); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ3); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef1); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef2); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef3); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium1); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium2); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium3); CHKERRQ(ierr);
  ierr = VecDestroy(&epsSReal); CHKERRQ(ierr);
  ierr = VecDestroy(&epsFReal); CHKERRQ(ierr);
  ierr = VecDestroy(&vgrad); CHKERRQ(ierr);  
  ierr = VecDestroy(&vgradlocal); CHKERRQ(ierr);

  ierr = VecDestroy(&x1); CHKERRQ(ierr);
  ierr = VecDestroy(&x2); CHKERRQ(ierr);
  ierr = VecDestroy(&b1);CHKERRQ(ierr);
  ierr = VecDestroy(&b2); CHKERRQ(ierr);
  ierr = VecDestroy(&J1); CHKERRQ(ierr);
  ierr = VecDestroy(&J2); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ1); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ2); CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp1);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp2);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp3);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspH);CHKERRQ(ierr);

  ISDestroy(&from);
  ISDestroy(&to);

  free(muinv1);
  free(muinv2);
  free(muinv3);
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

