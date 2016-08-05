#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

#define filteroverlap projsimpoverlap //projsimpoverlap, SIMPoverlap

int count=1;
int maxit;
int initdirect;
int mma_verbose;
double RRT, sigmax, sigmay, sigmaz;

/*------------------------------------------------------*/
int Nxyz, count;
double hxyz;
Mat B, C, D;
Vec vR, weight;
Vec vgradlocal;
Vec epsSReal,epsFReal;
IS from, to;
VecScatter scatter;
char filenameComm[PETSC_MAX_PATH_LEN];
int outputbase;

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

  int iteronly;
  PetscOptionsGetInt(PETSC_NULL,"-iteronly",&iteronly,&flg);
  if(flg) MyCheckAndOutputInt(flg,iteronly,"iteronly","iteronly");
  if(!flg) iteronly=0;

  int blochcondition=1;
  double beta[3]={0,0,0};
  PetscPrintf(PETSC_COMM_WORLD,"-------Use Bloch condition is %d\n",blochcondition);
  if(blochcondition){
    PetscOptionsGetReal(PETSC_NULL,"-betax",beta  ,&flg); MyCheckAndOutputDouble(flg,beta[0],"betax","Bloch vector component betax");
    PetscOptionsGetReal(PETSC_NULL,"-betay",beta+1,&flg); MyCheckAndOutputDouble(flg,beta[1],"betay","Bloch vector component betay");
    PetscOptionsGetReal(PETSC_NULL,"-betaz",beta+2,&flg); MyCheckAndOutputDouble(flg,beta[2],"betaz","Bloch vector component betaz");
  }

  int Mx,My,Mz,Mzslab,Npmlx,Npmly,Npmlz,Nx,Ny,Nz;
  double hx,hy,hz;
  PetscOptionsGetInt(PETSC_NULL,"-Mx",&Mx,&flg);  MyCheckAndOutputInt(flg,Mx,"Mx","Mx");
  PetscOptionsGetInt(PETSC_NULL,"-My",&My,&flg);  MyCheckAndOutputInt(flg,My,"My","My");
  PetscOptionsGetInt(PETSC_NULL,"-Mz",&Mz,&flg);  MyCheckAndOutputInt(flg,Mz,"Mz","Mz");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlx",&Npmlx,&flg);  MyCheckAndOutputInt(flg,Npmlx,"Npmlx","Npmlx");
  PetscOptionsGetInt(PETSC_NULL,"-Npmly",&Npmly,&flg);  MyCheckAndOutputInt(flg,Npmly,"Npmly","Npmly");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlz",&Npmlz,&flg);  MyCheckAndOutputInt(flg,Npmlz,"Npmlz","Npmlz");

  PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg);  MyCheckAndOutputInt(flg,Nx,"Nx","Nx");
  PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg);  MyCheckAndOutputInt(flg,Ny,"Ny","Ny");
  PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg);  MyCheckAndOutputInt(flg,Nz,"Nz","Nz");
  PetscOptionsGetInt(PETSC_NULL,"-Mzslab",&Mzslab,&flg);  MyCheckAndOutputInt(flg,Mzslab,"Mzslab","Mzslab");

  int DegFree, anisotropicDOF=0;
  Nxyz=Nx*Ny*Nz;
  DegFree = (anisotropicDOF ? 3 : 1 )*Mx*My*((Mzslab==0)?Mz:1);
  getreal("-hx",&hx,0.01);
  getreal("-hy",&hy,hx);
  getreal("-hz",&hz,hx);
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;

  RRT=1e-25;
  sigmax = pmlsigma(RRT,(double) Npmlx*hx);
  sigmay = pmlsigma(RRT,(double) Npmly*hy);
  sigmaz = pmlsigma(RRT,(double) Npmlz*hz);

  int BCPeriod, LowerPML;
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod");
  getint("-LowerPML",&LowerPML,1);

  int PrintEpsC;
  PetscOptionsGetInt(PETSC_NULL,"-outputbase",&outputbase,&flg); MyCheckAndOutputInt(flg,outputbase,"outputbase","outputbase");
  PetscOptionsGetInt(PETSC_NULL,"-PrintEpsC",&PrintEpsC,&flg); MyCheckAndOutputInt(flg,PrintEpsC,"PrintEpsC","PrintEpsC");

  double epsx,epsy,epsz,epsair,epssub;
  PetscOptionsGetReal(PETSC_NULL,"-epsx",&epsx,&flg); MyCheckAndOutputDouble(flg,epsx,"epsx","epsx");
  PetscOptionsGetReal(PETSC_NULL,"-epsy",&epsy,&flg); MyCheckAndOutputDouble(flg,epsy,"epsy","epsy");
  PetscOptionsGetReal(PETSC_NULL,"-epsz",&epsz,&flg); MyCheckAndOutputDouble(flg,epsz,"epsz","epsz");

  getreal("-epsair",&epsair,1.0);
  getreal("-epssub",&epssub,1.0);

  double Qabs;
  PetscOptionsGetReal(PETSC_NULL,"-Qabs",&Qabs,&flg);  MyCheckAndOutputDouble(flg,Qabs,"Qabs","Qabs");
  if (Qabs>1e15) Qabs=1.0/0.0;

  char initialdatafile[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-filenameprefix",filenameComm,PETSC_MAX_PATH_LEN,&flg);
  if (!flg) sprintf(filenameComm,"c3v_");
  if (flg) MyCheckAndOutputChar(flg,filenameComm,"filenameprefix","Filename Prefix");
  PetscOptionsGetString(PETSC_NULL,"-initdatfile",initialdatafile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdatafile,"initialdatafile","Inputdata file");

  int solver;
  PetscOptionsGetInt(PETSC_NULL,"-solver",&solver,&flg);  MyCheckAndOutputInt(flg,solver,"solver","LU Direct solver choice (0 PASTIX, 1 MUMPS, 2 SUPERLU_DIST)");

  int readlubsfromfile;
  getint("-readlubsfromfile",&readlubsfromfile,0);

/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
  

  Mat A,A1, A2;
  int Arows, Acols;

  myinterp(PETSC_COMM_WORLD, &A1,Nx,Ny,Nz,0,0,0,Nx,Ny,Nz,1,0);
  PetscPrintf(PETSC_COMM_WORLD,"*****A1 finished.\n");
  c3vinterp(PETSC_COMM_WORLD, &A2, Mx, My, Nx, Ny);
  PetscPrintf(PETSC_COMM_WORLD,"*****A2 finished.\n");
  MatGetSize(A1,&Arows,&Acols);
  PetscPrintf(PETSC_COMM_WORLD,"****Dimensions of A1 is %d by %d \n",Arows,Acols);
  MatGetSize(A2,&Arows,&Acols);
  PetscPrintf(PETSC_COMM_WORLD,"****Dimensions of A2 is %d by %d \n",Arows,Acols);

  MatMatMult(A1,A2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A);
  MatGetSize(A,&Arows,&Acols);
  PetscPrintf(PETSC_COMM_WORLD,"****Dimensions of A is %d by %d \n",Arows,Acols);

  GetDotMat(PETSC_COMM_WORLD, &B, Nx, Ny, Nz);
  CongMat(PETSC_COMM_WORLD, &C, 6*Nxyz);
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  ierr = MatCreateVecs(A,&epsSReal, &epsFReal); CHKERRQ(ierr);
  /*-----Set up vR, weight------*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &vR);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&weight); CHKERRQ(ierr);

  GetRealPartVec(vR,6*Nxyz);
  ierr = PetscObjectSetName((PetscObject) vR, "vR");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) weight, "weight");CHKERRQ(ierr);
  if(LowerPML==0)
    GetWeightVec(weight, Nx, Ny,Nz); 
  else
    VecSet(weight,1.0);

  /*--------Create index sets for the vec scatter -------*/
  ierr = VecCreateSeq(PETSC_COMM_SELF, DegFree, &vgradlocal); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&to); CHKERRQ(ierr);

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

//Job = 1 for multiple degenerate mode minimax optimization

if (Job==1){

  //read the bc's for the third mode and set up the third matrix
  //we can just use the muinv1 and epspmlQ for all the matrices since freqs and Qabs are the same
  //also set up the ksp3, ksp4, ksp5, ksp6

  Vec epspmlQ,epsmedium,epsDiff,epscoef,vgrad;
  Mat M1,M2,M3,M4,M5,M6;
  Vec x1,J1,weightedJ1,b1;
  Vec x2,J2,weightedJ2,b2;
  Vec x3,J3,weightedJ3,b3;
  Vec x4,J4,weightedJ4,b4;
  Vec x5,J5,weightedJ5,b5;
  Vec x6,J6,weightedJ6,b6;
  double omega1,omega2,omega3,omega4,omega5,omega6;
  KSP ksp1, ksp2, ksp3, ksp4, ksp5, ksp6;
  PC pc1, pc2, pc3, pc4, pc5, pc6;
  int its1=100, its2=100, its3=100, its4=100, its5=100, its6=100;
  Vec unitx,unity,unitz;
  Vec epspml;
  double omega;

  getreal("-freq1",&omega1,1.0);
  getreal("-freq2",&omega2,1.0);
  getreal("-freq3",&omega3,1.0);
  getreal("-freq4",&omega4,1.0);
  getreal("-freq5",&omega5,1.0);
  getreal("-freq6",&omega6,1.0);
  omega1=2*PI*omega1;
  omega2=2*PI*omega2;
  omega3=2*PI*omega3;
  omega4=2*PI*omega4;
  omega5=2*PI*omega5;
  omega6=2*PI*omega6;
  omega=omega1;

  VecDuplicate(vR,&epspmlQ);
  VecDuplicate(vR,&epsmedium);
  VecDuplicate(vR,&epsDiff);
  VecDuplicate(vR,&epscoef);
  VecDuplicate(vR,&x1);
  VecDuplicate(vR,&J1);
  VecDuplicate(vR,&weightedJ1);
  VecDuplicate(vR,&b1);
  VecDuplicate(vR,&x2);
  VecDuplicate(vR,&J2);
  VecDuplicate(vR,&weightedJ2);
  VecDuplicate(vR,&b2);
  VecDuplicate(vR,&x3);
  VecDuplicate(vR,&J3);
  VecDuplicate(vR,&weightedJ3);
  VecDuplicate(vR,&b3);
  VecDuplicate(vR,&x4);
  VecDuplicate(vR,&J4);
  VecDuplicate(vR,&weightedJ4);
  VecDuplicate(vR,&b4);
  VecDuplicate(vR,&x5);
  VecDuplicate(vR,&J5);
  VecDuplicate(vR,&weightedJ5);
  VecDuplicate(vR,&b5);
  VecDuplicate(vR,&x6);
  VecDuplicate(vR,&J6);
  VecDuplicate(vR,&weightedJ6);
  VecDuplicate(vR,&b6);
  VecDuplicate(vR,&unitx);
  VecDuplicate(vR,&unity);
  VecDuplicate(vR,&unitz);
  VecDuplicate(vR,&epspml);
  VecDuplicate(vR,&epspmlQ);
  VecDuplicate(vR,&epscoef);
  VecDuplicate(vR,&epsmedium); 
  VecDuplicate(vR,&epsDiff); 

  /*----Set up the epsilon and PML vectors--------*/
  GetUnitVec(unitx,0,6*Nxyz);
  GetUnitVec(unity,1,6*Nxyz);
  GetUnitVec(unitz,2,6*Nxyz);

  VecSet(epsDiff,0.0);
  VecAXPY(epsDiff,epsx,unitx);
  VecAXPY(epsDiff,epsy,unity);
  VecAXPY(epsDiff,epsz,unitz);

  EpsPMLFull(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega,LowerPML);
  EpsCombine(D, weight, epspml, epspmlQ, epscoef, Qabs, omega, epsDiff);

  GetMediumVecwithSub(epsmedium,Nz,Mz,epsair,epssub);

  VecDuplicate(epsSReal,&vgrad); 

/*-------------------------------------------------------------------------*/
  	ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);									
        ierr=MatMult(A,epsSReal,epsFReal); CHKERRQ(ierr);									
  	ierr=VecPointwiseMult(epsFReal,epsFReal,epsDiff); CHKERRQ(ierr);							
  	ierr = VecAXPY(epsFReal,1.0,epsmedium); CHKERRQ(ierr);
	ierr = MatMultTranspose(A,epsFReal,epsSReal); CHKERRQ(ierr);
  	OutputVec(PETSC_COMM_WORLD, epsFReal, "initial_","epsF.m");
	OutputVec(PETSC_COMM_WORLD, epsSReal, "initial_","epsS.m");
	//OutputMat(PETSC_COMM_WORLD, A2, "c3vMatrix",".m");
/*-------------------------------------------------------------------------*/

  double Jmag1, Jmag2, Jmag3, Jmag4, Jmag5, Jmag6;
  getreal("-Jmag1",&Jmag1,1.0);
  getreal("-Jmag2",&Jmag2,1.0);
  getreal("-Jmag3",&Jmag3,1.0);
  getreal("-Jmag4",&Jmag4,1.0);
  getreal("-Jmag5",&Jmag5,1.0);
  getreal("-Jmag6",&Jmag6,1.0);

  setupKSP(PETSC_COMM_WORLD,&ksp1,&pc1,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp2,&pc2,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp3,&pc3,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp4,&pc4,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp5,&pc5,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp6,&pc6,solver,iteronly);
  
  double frac;
  getreal("-penalfrac",&frac,1.0);

  int b1x[2], b1y[2], b1z[2];
  PetscOptionsGetInt(PETSC_NULL,"-b1xl",b1x,&flg);    MyCheckAndOutputInt(flg,b1x[0],"b1xl","BC at x lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1xu",b1x+1,&flg);  MyCheckAndOutputInt(flg,b1x[1],"b1xu","BC at x upper for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1yl",b1y,&flg);    MyCheckAndOutputInt(flg,b1y[0],"b1yl","BC at y lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1yu",b1y+1,&flg);  MyCheckAndOutputInt(flg,b1y[1],"b1yu","BC at y upper for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1zl",b1z,&flg);    MyCheckAndOutputInt(flg,b1z[0],"b1zl","BC at z lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1zu",b1z+1,&flg);  MyCheckAndOutputInt(flg,b1z[1],"b1zu","BC at z upper for mode 1");

  int b2x[2], b2y[2], b2z[2];
  PetscOptionsGetInt(PETSC_NULL,"-b2xl",b2x,&flg);    MyCheckAndOutputInt(flg,b2x[0],"b2xl","BC at x lower for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2xu",b2x+1,&flg);  MyCheckAndOutputInt(flg,b2x[1],"b2xu","BC at x upper for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2yl",b2y,&flg);    MyCheckAndOutputInt(flg,b2y[0],"b2yl","BC at y lower for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2yu",b2y+1,&flg);  MyCheckAndOutputInt(flg,b2y[1],"b2yu","BC at y upper for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2zl",b2z,&flg);    MyCheckAndOutputInt(flg,b2z[0],"b2zl","BC at z lower for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2zu",b2z+1,&flg);  MyCheckAndOutputInt(flg,b2z[1],"b2zu","BC at z upper for mode 2");

  int b3x[2], b3y[2], b3z[2];
  PetscOptionsGetInt(PETSC_NULL,"-b3xl",b3x,&flg);    MyCheckAndOutputInt(flg,b3x[0],"b3xl","BC at x lower for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3xu",b3x+1,&flg);  MyCheckAndOutputInt(flg,b3x[1],"b3xu","BC at x upper for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3yl",b3y,&flg);    MyCheckAndOutputInt(flg,b3y[0],"b3yl","BC at y lower for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3yu",b3y+1,&flg);  MyCheckAndOutputInt(flg,b3y[1],"b3yu","BC at y upper for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3zl",b3z,&flg);    MyCheckAndOutputInt(flg,b3z[0],"b3zl","BC at z lower for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3zu",b3z+1,&flg);  MyCheckAndOutputInt(flg,b3z[1],"b3zu","BC at z upper for mode 3");

  int b4x[2], b4y[2], b4z[2];
  PetscOptionsGetInt(PETSC_NULL,"-b4xl",b4x,&flg);    MyCheckAndOutputInt(flg,b4x[0],"b4xl","BC at x lower for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4xu",b4x+1,&flg);  MyCheckAndOutputInt(flg,b4x[1],"b4xu","BC at x upper for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4yl",b4y,&flg);    MyCheckAndOutputInt(flg,b4y[0],"b4yl","BC at y lower for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4yu",b4y+1,&flg);  MyCheckAndOutputInt(flg,b4y[1],"b4yu","BC at y upper for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4zl",b4z,&flg);    MyCheckAndOutputInt(flg,b4z[0],"b4zl","BC at z lower for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4zu",b4z+1,&flg);  MyCheckAndOutputInt(flg,b4z[1],"b4zu","BC at z upper for mode 4");

  int b5x[2], b5y[2], b5z[2];
  PetscOptionsGetInt(PETSC_NULL,"-b5xl",b5x,&flg);    MyCheckAndOutputInt(flg,b5x[0],"b5xl","BC at x lower for mode 5");
  PetscOptionsGetInt(PETSC_NULL,"-b5xu",b5x+1,&flg);  MyCheckAndOutputInt(flg,b5x[1],"b5xu","BC at x upper for mode 5");
  PetscOptionsGetInt(PETSC_NULL,"-b5yl",b5y,&flg);    MyCheckAndOutputInt(flg,b5y[0],"b5yl","BC at y lower for mode 5");
  PetscOptionsGetInt(PETSC_NULL,"-b5yu",b5y+1,&flg);  MyCheckAndOutputInt(flg,b5y[1],"b5yu","BC at y upper for mode 5");
  PetscOptionsGetInt(PETSC_NULL,"-b5zl",b5z,&flg);    MyCheckAndOutputInt(flg,b5z[0],"b5zl","BC at z lower for mode 5");
  PetscOptionsGetInt(PETSC_NULL,"-b5zu",b5z+1,&flg);  MyCheckAndOutputInt(flg,b5z[1],"b5zu","BC at z upper for mode 5");

  int b6x[2], b6y[2], b6z[2];
  PetscOptionsGetInt(PETSC_NULL,"-b6xl",b6x,&flg);    MyCheckAndOutputInt(flg,b6x[0],"b6xl","BC at x lower for mode 6");
  PetscOptionsGetInt(PETSC_NULL,"-b6xu",b6x+1,&flg);  MyCheckAndOutputInt(flg,b6x[1],"b6xu","BC at x upper for mode 6");
  PetscOptionsGetInt(PETSC_NULL,"-b6yl",b6y,&flg);    MyCheckAndOutputInt(flg,b6y[0],"b6yl","BC at y lower for mode 6");
  PetscOptionsGetInt(PETSC_NULL,"-b6yu",b6y+1,&flg);  MyCheckAndOutputInt(flg,b6y[1],"b6yu","BC at y upper for mode 6");
  PetscOptionsGetInt(PETSC_NULL,"-b6zl",b6z,&flg);    MyCheckAndOutputInt(flg,b6z[0],"b6zl","BC at z lower for mode 6");
  PetscOptionsGetInt(PETSC_NULL,"-b6zu",b6z+1,&flg);  MyCheckAndOutputInt(flg,b6z[1],"b6zu","BC at z upper for mode 6");

  Vec muinvpml;
  int add=0;
  double *muinv;

  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega,LowerPML);
  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  AddMuAbsorption(muinv,muinvpml,Qabs,add);

  MoperatorGeneralBloch(PETSC_COMM_WORLD, &M1, Nx,Ny,Nz, hx,hy,hz, b1x,b1y,b1z, muinv, BCPeriod, beta);
  MoperatorGeneralBloch(PETSC_COMM_WORLD, &M2, Nx,Ny,Nz, hx,hy,hz, b2x,b2y,b2z, muinv, BCPeriod, beta);
  MoperatorGeneralBloch(PETSC_COMM_WORLD, &M3, Nx,Ny,Nz, hx,hy,hz, b3x,b3y,b3z, muinv, BCPeriod, beta);
  MoperatorGeneralBloch(PETSC_COMM_WORLD, &M4, Nx,Ny,Nz, hx,hy,hz, b4x,b4y,b4z, muinv, BCPeriod, beta);
  MoperatorGeneralBloch(PETSC_COMM_WORLD, &M5, Nx,Ny,Nz, hx,hy,hz, b5x,b5y,b5z, muinv, BCPeriod, beta);
  MoperatorGeneralBloch(PETSC_COMM_WORLD, &M6, Nx,Ny,Nz, hx,hy,hz, b6x,b6y,b6z, muinv, BCPeriod, beta);

  //Read the J's from input;
  VecSet(J1,0.0);
  VecSet(J2,0.0);
  VecSet(J3,0.0);
  VecSet(J4,0.0);
  VecSet(J5,0.0);
  VecSet(J6,0.0);
  double *J1array,*J2array,*J3array,*J4array,*J5array,*J6array;
  FILE *J1ptf,*J2ptf,*J3ptf,*J4ptf,*J5ptf,*J6ptf;
  J1array = (double *) malloc(6*Nxyz*sizeof(double));
  J2array = (double *) malloc(6*Nxyz*sizeof(double));
  J3array = (double *) malloc(6*Nxyz*sizeof(double));
  J4array = (double *) malloc(6*Nxyz*sizeof(double));
  J5array = (double *) malloc(6*Nxyz*sizeof(double));
  J6array = (double *) malloc(6*Nxyz*sizeof(double));
  J1ptf = fopen("J1input.txt","r");
  J2ptf = fopen("J2input.txt","r");
  J3ptf = fopen("J3input.txt","r"); 
  J4ptf = fopen("J4input.txt","r"); 
  J5ptf = fopen("J5input.txt","r"); 
  J6ptf = fopen("J6input.txt","r"); 
  int inJi;
  PetscPrintf(PETSC_COMM_WORLD,"---reading J1, J2, J3, J4, J5 and J6-----\n");
  for (inJi=0;inJi<6*Nxyz;inJi++)
    { 
      fscanf(J1ptf,"%lf",&J1array[inJi]);
      fscanf(J2ptf,"%lf",&J2array[inJi]);
      fscanf(J3ptf,"%lf",&J3array[inJi]);	
      fscanf(J4ptf,"%lf",&J4array[inJi]);
      fscanf(J5ptf,"%lf",&J5array[inJi]);
      fscanf(J6ptf,"%lf",&J6array[inJi]);	
    }
  fclose(J1ptf);
  fclose(J2ptf);
  fclose(J3ptf);
  fclose(J4ptf);
  fclose(J5ptf);
  fclose(J6ptf);
  PetscPrintf(PETSC_COMM_WORLD,"---Done reading J2, J2, J3, J4, J5 and J6!-----\n");

  ArrayToVec(J1array,J1);
  ArrayToVec(J2array,J2);
  ArrayToVec(J3array,J3);
  ArrayToVec(J4array,J4);
  ArrayToVec(J5array,J5);
  ArrayToVec(J6array,J6);
  free(J1array);
  free(J2array);
  free(J3array);
  free(J4array);
  free(J5array);
  free(J6array);
  VecScale(J1,Jmag1);
  VecScale(J2,Jmag2);
  VecScale(J3,Jmag3);
  VecScale(J4,Jmag4);
  VecScale(J5,Jmag5);
  VecScale(J6,Jmag6);

  /*
  OutputVec(PETSC_COMM_WORLD,J1,"J1",".m");
  OutputVec(PETSC_COMM_WORLD,J2,"J2",".m");
  OutputVec(PETSC_COMM_WORLD,J3,"J3",".m");
  OutputVec(PETSC_COMM_WORLD,J4,"J4",".m");
  OutputVec(PETSC_COMM_WORLD,J5,"J5",".m");
  OutputVec(PETSC_COMM_WORLD,J6,"J6",".m");
  */

  VecPointwiseMult(weightedJ1,weight,J1);
  VecPointwiseMult(weightedJ2,weight,J2);
  VecPointwiseMult(weightedJ3,weight,J3);
  VecPointwiseMult(weightedJ4,weight,J4);
  VecPointwiseMult(weightedJ5,weight,J5);
  VecPointwiseMult(weightedJ6,weight,J6);

  MatMult(D,J1,b1);
  VecScale(b1,omega1);
  MatMult(D,J2,b2);
  VecScale(b2,omega2);
  MatMult(D,J3,b3);
  VecScale(b3,omega3);
  MatMult(D,J4,b4);
  VecScale(b4,omega4);
  MatMult(D,J5,b5);
  VecScale(b5,omega5);
  MatMult(D,J6,b6);
  VecScale(b6,omega6);

  LDOSdataGroup ldos1data={omega1,M1,A,x1,b1,weightedJ1,epspmlQ,epsmedium,epsDiff,&its1,epscoef,vgrad,ksp1};
  LDOSdataGroup ldos2data={omega2,M2,A,x2,b2,weightedJ2,epspmlQ,epsmedium,epsDiff,&its2,epscoef,vgrad,ksp2};
  LDOSdataGroup ldos3data={omega3,M3,A,x3,b3,weightedJ3,epspmlQ,epsmedium,epsDiff,&its3,epscoef,vgrad,ksp3};
  LDOSdataGroup ldos4data={omega4,M4,A,x4,b4,weightedJ4,epspmlQ,epsmedium,epsDiff,&its4,epscoef,vgrad,ksp4};
  LDOSdataGroup ldos5data={omega5,M5,A,x5,b5,weightedJ5,epspmlQ,epsmedium,epsDiff,&its5,epscoef,vgrad,ksp5};
  LDOSdataGroup ldos6data={omega6,M6,A,x6,b6,weightedJ6,epspmlQ,epsmedium,epsDiff,&its6,epscoef,vgrad,ksp6};

 /*---------Optimization--------*/
  double tstart;
  PetscOptionsGetReal(PETSC_NULL,"-tstart",&tstart,&flg);  MyCheckAndOutputDouble(flg,tstart,"tstart","Initial value of dummy variable t");
  int DegFreeAll=DegFree+1;
  double *epsoptAll, *gradAll;
  epsoptAll = (double *) malloc(DegFreeAll*sizeof(double));
  gradAll = (double *) malloc(DegFreeAll*sizeof(double));
  for (i=0;i<DegFree;i++){ epsoptAll[i]=epsopt[i]; }
  epsoptAll[DegFreeAll-1]=tstart;
  
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
  lb[DegFreeAll-1]=0;
  ub[DegFreeAll-1]=1.0/0.0;

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
  PetscOptionsGetInt(PETSC_NULL,"-nummodes",&nummodes,&flg);  MyCheckAndOutputInt(flg,nummodes,"nummodes","number of degenerate modes to optimize");

  if(nummodes==2){
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos1data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos2data,1e-8);
  }else if(nummodes==3){
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos1data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos2data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos3data,1e-8);
  }else if(nummodes==4){
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos1data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos2data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos3data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos4data,1e-8);
  }else if(nummodes==6){
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos1data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos2data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos3data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos4data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos5data,1e-8);
    nlopt_add_inequality_constraint(opt,ldoskconstraintnofilter,&ldos6data,1e-8);
  }

  if(frac<1.0) nlopt_add_inequality_constraint(opt,pfunc,&frac,1e-8);
  nlopt_set_max_objective(opt,maxminobjfun,NULL);   

  result = nlopt_optimize(opt,epsoptAll,&maxf);

  PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);

  PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);

  /*************************/
  int gtest, gpos;
  double s1, s2, ds, epscen, gbeta;
  getint("-gtest",&gtest,0);
  getint("-gpos",&gpos,0);
  getreal("-s1",&s1,0);
  getreal("-s2",&s2,1);
  getreal("-ds",&ds,0.01);
  if(gtest==1){
    for(epscen=s1;epscen<s2;epscen=epscen+ds){
      epsoptAll[gpos]=epscen;
      gbeta=ldoskconstraintnofilter(DegFreeAll,epsoptAll,gradAll,&ldos2data);
      PetscPrintf(PETSC_COMM_WORLD,"epscen: %g objfunc: %g objfunc-grad: %g \n", epsoptAll[gpos], gbeta, gradAll[gpos]);
    }
  }
  /*************************/

  free(epsoptAll);
  free(gradAll);
  free(lb);
  free(ub);
  nlopt_destroy(opt);

  VecDestroy(&epspmlQ);
  VecDestroy(&epsmedium);
  VecDestroy(&epsDiff);
  VecDestroy(&epscoef);
  VecDestroy(&x1);
  VecDestroy(&J1);
  VecDestroy(&weightedJ1);
  VecDestroy(&b1);
  VecDestroy(&x2);
  VecDestroy(&J2);
  VecDestroy(&weightedJ2);
  VecDestroy(&b2);
  VecDestroy(&x3);
  VecDestroy(&J3);
  VecDestroy(&weightedJ3);
  VecDestroy(&b3);
  VecDestroy(&x4);
  VecDestroy(&J4);
  VecDestroy(&weightedJ4);
  VecDestroy(&b4);
  VecDestroy(&x5);
  VecDestroy(&J5);
  VecDestroy(&weightedJ5);
  VecDestroy(&b5);
  VecDestroy(&x6);
  VecDestroy(&J6);
  VecDestroy(&weightedJ6);
  VecDestroy(&b6);
  VecDestroy(&unitx);
  VecDestroy(&unity);
  VecDestroy(&unitz);
  VecDestroy(&epspml);
  VecDestroy(&epspmlQ);
  VecDestroy(&epscoef);
  VecDestroy(&epsmedium); 
  VecDestroy(&epsDiff); 
  VecDestroy(&vgrad);

  free(muinv);
  KSPDestroy(&ksp1); 
  KSPDestroy(&ksp2); 
  KSPDestroy(&ksp3); 
  KSPDestroy(&ksp4); 
  KSPDestroy(&ksp5); 
  KSPDestroy(&ksp6); 

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
  ierr = MatDestroy(&A1); CHKERRQ(ierr);
  ierr = MatDestroy(&A2); CHKERRQ(ierr);
  ierr = MatDestroy(&B); CHKERRQ(ierr);
  ierr = MatDestroy(&C); CHKERRQ(ierr);
  ierr = MatDestroy(&D); CHKERRQ(ierr);

  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&weight); CHKERRQ(ierr);
  ierr = VecDestroy(&vgradlocal); CHKERRQ(ierr);
  VecDestroy(&epsSReal);
  VecDestroy(&epsFReal);

  ISDestroy(&from);
  ISDestroy(&to);

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

