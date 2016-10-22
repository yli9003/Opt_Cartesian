#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

int mma_verbose;
int initdirect, maxit;

int count=1;
VecScatter scatter;
IS from, to;
Vec vgradlocal;
Mat B,C,D;
Vec vR, epsFReal;

int pSIMP;
double bproj, etaproj;
Mat Hfilt;
KSP kspH;
int itsH;

/*------------------------------------------------------*/

PetscErrorCode setupKSP(MPI_Comm comm, KSP *ksp, PC *pc, int solver, int iteronly);

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

  /*************************************************************/
  PetscBool flg;

  getint("-initdirect",&initdirect,3);
  getint("-maxit",&maxit,15);
  int solver;
  PetscOptionsGetInt(PETSC_NULL,"-solver",&solver,&flg);  
  if(!flg) solver=1;
  PetscPrintf(PETSC_COMM_WORLD,"LU Direct solver choice (0 PASTIX, 1 MUMPS, 2 SUPERLU_DIST): %d\n",solver);
  double sH, nR;
  int dimH;
  PC pcH;
  getint("-pSIMP",&pSIMP,1);
  getreal("-bproj",&bproj,0);
  getreal("-etaproj",&etaproj,0.5);
  getreal("-sH",&sH,-1);
  getreal("-nR",&nR,0);
  getint("-dimH",&dimH,1);
  /***************************************************************/

  Universals flagparams;
  readfromflags(&flagparams);

  Mat A;
  Vec vI, weight;
  Vec epsSReal;

  setupMatVecs(flagparams, &A, &C, &D, &vR, &vI, &weight, &epsSReal, &epsFReal);

  ierr=VecCreateSeq(PETSC_COMM_SELF, flagparams.DegFree, &vgradlocal); CHKERRQ(ierr); 
  ISCreateStride(PETSC_COMM_SELF,flagparams.DegFree,0,1,&from); 
  ISCreateStride(PETSC_COMM_SELF,flagparams.DegFree,0,1,&to); 

  GetH1d(PETSC_COMM_WORLD,&Hfilt,flagparams.DegFree,sH,nR,&kspH,&pcH);
  /****************************************************************************/

  double *epsopt;
  FILE *ptf;
  epsopt = (double *) malloc(flagparams.DegFree*sizeof(double));
  ptf = fopen(flagparams.initialdatafile,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");
  int i;
  for (i=0;i<flagparams.DegFree;i++)
    { 
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);
  /**********************************************************/

  char maxwellfile1[PETSC_MAX_PATH_LEN];
  Maxwell maxwell1;
  PetscOptionsGetString(PETSC_NULL,"-maxwellfile1",maxwellfile1,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,maxwellfile1,"maxwellfile1","maxwellfile1");
  makemaxwell(maxwellfile1,flagparams,A,D,vR,weight,&maxwell1);

  Vec tmpepsS, tmpepsF;
  MatCreateVecs(A,&tmpepsS,&tmpepsF);
  ArrayToVec(epsopt,tmpepsS);
  MatMult(A,tmpepsS,tmpepsF);
  VecPointwiseMult(tmpepsF,tmpepsF,maxwell1.epsdiff);
  VecAXPY(tmpepsF,1.0,maxwell1.epsbkg);
  OutputVec(PETSC_COMM_WORLD,tmpepsF,"epsF",".m");
  VecDestroy(&tmpepsS);
  VecDestroy(&tmpepsF);
 
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Barrier(PETSC_COMM_WORLD);

  
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
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

