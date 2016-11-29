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
  
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0) 
    mma_verbose=1;

  int nlayers=3;
  int Mx=100, My=1;
  int *Mz;
  int Mzslab, DegFree;
  Mz = (int *) malloc(nlayers*sizeof(int));
  Mz[0]=10;
  Mz[1]=7;
  Mz[2]=15;
  int Npmlx=25, Npmly=0, Npmlz=25;
  int Nx=150, Ny=1, Nz=250;
  int Nxo=(Nx-Mx)/2;
  int Nyo=0;
  int *Nzo;
  Nzo = (int *) malloc(nlayers*sizeof(int));
  Nzo[0]=60;
  Nzo[1]=80;
  Nzo[2]=100;
  Mzslab=1;
  DegFree = nlayers*Mx*My;

  layeredA(PETSC_COMM_WORLD,&A, Nx,Ny,Nz, nlayers,Nxo,Nyo,Nzo, Mx,My,Mz, Mzslab);
  Vec epsSReal;
  MatCreateVecs(A,&epsSReal, &epsFReal);
  double *epsopt;
  FILE *ptf;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  ptf = fopen("testinput.txt","r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");
  int i;
  for (i=0;i<DegFree;i++)
    {
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);

  ArrayToVec(epsopt,epsSReal);
  MatMult(A,epsSReal,epsFReal);  


  Vec epsBkg, epsDiff;
  VecDuplicate(epsFReal,&epsBkg);
  VecDuplicate(epsFReal,&epsDiff);
  double *epsbkg, *epsdiff;
  epsbkg = (double *) malloc(nlayers*sizeof(double));
  epsdiff = (double *) malloc(nlayers*sizeof(double));
  epsbkg[0]=1;
  epsbkg[1]=1;
  epsbkg[2]=1;
  epsdiff[0]=1.5;
  epsdiff[1]=2.0;
  epsdiff[2]=1.1;
  layeredepsbkg(epsBkg, Nx,Ny,Nz, nlayers,Nzo,Mz, epsbkg, 2,1,1.3);
  layeredepsdiff(epsDiff, Nx,Ny,Nz, nlayers,Nzo,Mz, epsdiff, 0,0,0);

  OutputVec(PETSC_COMM_WORLD, epsFReal, "epsF",".m");
  OutputVec(PETSC_COMM_WORLD, epsDiff, "epsDiff",".m");
  OutputVec(PETSC_COMM_WORLD, epsBkg, "epsBkg",".m");

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

