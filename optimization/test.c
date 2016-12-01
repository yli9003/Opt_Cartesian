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

/*------------------------------------------------------*/

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

  int Mx=100, My=1;
  int Mzslab, DegFree;
  int Npmlx=25, Npmly=0, Npmlz=25;
  int Nx=150, Ny=1, Nz=250;
  int Nxo=(Nx-Mx)/2;
  int Nyo=0;
  Mzslab=1;

  int nlayers;
  int *Mz, *Nzo;
  epsinfo eps;
  int i;
  DegFree=0;
  getint("-nlayers",&nlayers,1);
  Mz =(int *) malloc(nlayers*sizeof(int));
  Nzo=(int *) malloc(nlayers*sizeof(int));
  eps.epsdiff=(double *) malloc(nlayers*sizeof(double));
  eps.epsbkg=(double *) malloc(nlayers*sizeof(double));
  char tmpflg[PETSC_MAX_PATH_LEN];
  for (i=0;i<nlayers;i++) {
    sprintf(tmpflg,"-Mz[%d]",i+1);
    getint(tmpflg,Mz+i,10);
    sprintf(tmpflg,"-Nzo[%d]",i+1);
    getint(tmpflg,Nzo+i,(Nz-Mz[i])/2);
    DegFree=DegFree+Mx*My*((Mzslab==0)?Mz[i]:1);

    sprintf(tmpflg,"-epsdiff[%d]",i+1);
    getreal(tmpflg,eps.epsdiff+i,3.6575);
    sprintf(tmpflg,"-epsbkg[%d]",i+1);
    getreal(tmpflg,eps.epsbkg+i,2.1025);

  }
  getreal("-epssubdiff",&eps.epssubdiff,0);
  getreal("-epsairdiff",&eps.epsairdiff,0);
  getreal("-epsmiddiff",&eps.epsmiddiff,0);
  getreal("-epssub",&eps.epssub,2.1025);
  getreal("-epsair",&eps.epsair,1.0);
  getreal("-epsmid",&eps.epsmid,2.1025);

  layeredA(PETSC_COMM_WORLD,&A, Nx,Ny,Nz, nlayers,Nxo,Nyo,Nzo, Mx,My,Mz, Mzslab);
  PetscPrintf(PETSC_COMM_WORLD,"layered A created. \n");
  Vec epsSReal;
  MatCreateVecs(A,&epsSReal, &epsFReal);
  double *epsopt;
  FILE *ptf;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  ptf = fopen("testinput.txt","r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");
  for (i=0;i<DegFree;i++)
    {
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);

  ArrayToVec(epsopt,epsSReal);
  MatMult(A,epsSReal,epsFReal);  


  Vec epsBkg, epsDiff, epsFull;
  VecDuplicate(epsFReal,&epsBkg);
  VecDuplicate(epsFReal,&epsDiff);
  VecDuplicate(epsFReal,&epsFull);
  layeredepsbkg(epsBkg, Nx,Ny,Nz, nlayers,Nzo,Mz, eps.epsbkg, eps.epssub, eps.epsair, eps.epsmid);
  layeredepsdiff(epsDiff, Nx,Ny,Nz, nlayers,Nzo,Mz, eps.epsdiff, eps.epssubdiff, eps.epsairdiff, eps.epsmiddiff);

  VecPointwiseMult(epsFull,epsFReal,epsDiff);
  VecAXPY(epsFull,1.0,epsBkg);

  OutputVec(PETSC_COMM_WORLD, epsFReal, "epsF",".m");
  OutputVec(PETSC_COMM_WORLD, epsDiff, "epsDiff",".m");
  OutputVec(PETSC_COMM_WORLD, epsBkg, "epsBkg",".m");
  OutputVec(PETSC_COMM_WORLD, epsFull, "epsFull",".m");

  /*------------ finalize the program -------------*/

  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Barrier(PETSC_COMM_WORLD);
  }
  
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}
