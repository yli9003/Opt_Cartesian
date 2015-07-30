#include <stdlib.h>
#include <petsc.h>
#include <slepc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

int maxit=20;
Mat A, B, C, D;
Vec vR;
int mma_verbose;
Vec epsFReal;

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize ------*/
  SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0) 
    mma_verbose=1;

  int DegFree, Mzslab;
  int Mx, My, Mz, Npmlx, Npmly, Npmlz, Nx, Ny, Nz, Nxyz;
  double hx, hy, hz;
  double omega;
  double epsx, epsy, epsz, epsair, epssub;
  double RRT, sigmax, sigmay, sigmaz;
  double Qabs;
  char initialdatafile[PETSC_MAX_PATH_LEN];
  Mat M;
  Vec epsSReal, epsDiff, epsmedium, epspmlQ;
  
/*****************************************************-------Set up the options parameters-------------********************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
  PetscBool flg;

  int blochcondition;
  double beta[3]={0,0,0};
  PetscOptionsGetInt(PETSC_NULL,"-blochcondition",&blochcondition,&flg);
  if(!flg) blochcondition=0;
  PetscPrintf(PETSC_COMM_WORLD,"-------Use Bloch condition is %d\n",blochcondition);
  if(blochcondition){
    PetscOptionsGetReal(PETSC_NULL,"-betax",beta  ,&flg); MyCheckAndOutputDouble(flg,beta[0],"betax","Bloch vector component betax");
    PetscOptionsGetReal(PETSC_NULL,"-betay",beta+1,&flg); MyCheckAndOutputDouble(flg,beta[1],"betay","Bloch vector component betay");
    PetscOptionsGetReal(PETSC_NULL,"-betaz",beta+2,&flg); MyCheckAndOutputDouble(flg,beta[2],"betaz","Bloch vector component betaz");
  }
  
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

  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  PetscOptionsGetReal(PETSC_NULL,"-hy",&hy,&flg);  MyCheckAndOutputDouble(flg,hy,"hy","hy");
  PetscOptionsGetReal(PETSC_NULL,"-hz",&hz,&flg);  MyCheckAndOutputDouble(flg,hz,"hz","hz");
  
  int BCPeriod, LowerPML;
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod");
  PetscOptionsGetInt(PETSC_NULL,"-LowerPML",&LowerPML,&flg);  MyCheckAndOutputInt(flg,LowerPML,"LowerPML","LowerPML");

  int bx[2], by[2], bz[2];
  PetscOptionsGetInt(PETSC_NULL,"-bxl",bx,&flg);    MyCheckAndOutputInt(flg,bx[0],"bxl","BC at x lower ");
  PetscOptionsGetInt(PETSC_NULL,"-bxu",bx+1,&flg);  MyCheckAndOutputInt(flg,bx[1],"bxu","BC at x upper ");
  PetscOptionsGetInt(PETSC_NULL,"-byl",by,&flg);    MyCheckAndOutputInt(flg,by[0],"byl","BC at y lower ");
  PetscOptionsGetInt(PETSC_NULL,"-byu",by+1,&flg);  MyCheckAndOutputInt(flg,by[1],"byu","BC at y upper ");
  PetscOptionsGetInt(PETSC_NULL,"-bzl",bz,&flg);    MyCheckAndOutputInt(flg,bz[0],"bzl","BC at z lower ");
  PetscOptionsGetInt(PETSC_NULL,"-bzu",bz+1,&flg);  MyCheckAndOutputInt(flg,bz[1],"bzu","BC at z upper ");

  double freq;
  PetscOptionsGetReal(PETSC_NULL,"-freq",&freq,&flg);
  if(!flg) freq=1.0;
  PetscPrintf(PETSC_COMM_WORLD,"-------freq: %g \n",freq);
  omega=2.0*PI*freq;

  PetscOptionsGetReal(PETSC_NULL,"-epsx",&epsx,&flg); MyCheckAndOutputDouble(flg,epsx,"epsx","epsx");
  PetscOptionsGetReal(PETSC_NULL,"-epsy",&epsy,&flg); MyCheckAndOutputDouble(flg,epsy,"epsy","epsy");
  PetscOptionsGetReal(PETSC_NULL,"-epsz",&epsz,&flg); MyCheckAndOutputDouble(flg,epsz,"epsz","epsz");

  PetscOptionsGetReal(PETSC_NULL,"-epsmed",&epsair,&flg);
  if(!flg) epsair=1.0;
  if(flg) MyCheckAndOutputDouble(flg,epsair,"epsair","Dielectric of surrounding medium");
  PetscOptionsGetReal(PETSC_NULL,"-epssub",&epssub,&flg);
  if(!flg) epssub=1.0; 
  if(flg) MyCheckAndOutputDouble(flg,epssub,"epssub","Dielectric of substrate at freq");

  RRT=1e-25;
  sigmax = pmlsigma(RRT,(double) Npmlx*hx);
  sigmay = pmlsigma(RRT,(double) Npmly*hy);
  sigmaz = pmlsigma(RRT,(double) Npmlz*hz);

  PetscPrintf(PETSC_COMM_WORLD,"sigma, omega, DegFree: %g, %g, %g, %g \n", sigmax, sigmay, sigmaz, omega);

  PetscOptionsGetReal(PETSC_NULL,"-Qabs",&Qabs,&flg);  MyCheckAndOutputDouble(flg,Qabs,"Qabs","Qabs");
  if (Qabs>1e15) Qabs=1.0/0.0;

  PetscOptionsGetString(PETSC_NULL,"-initdatfile",initialdatafile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdatafile,"initialdatafile","Inputdata file");

/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/


  /*------Set up the A, C, D matrices--------------*/
  Nxyz=Nx*Ny*Nz;
  DegFree = (Mzslab==0) * Mx*My*Mz + (Mzslab==1) * Mx*My;
  myinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, LowerPML*floor((Nx-Mx)/2),LowerPML*floor((Ny-My)/2),LowerPML*floor((Nz-Mz)/2), Mx,My,Mz,Mzslab,0);
  CongMat(PETSC_COMM_WORLD, &C, 6*Nxyz);
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  /*-----Set up vR------*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &vR);CHKERRQ(ierr);
  GetRealPartVec(vR,6*Nxyz);

  ierr = PetscObjectSetName((PetscObject) vR, "vR");CHKERRQ(ierr);
  
  /*----Set up the universal parts of M-------*/
  Vec munivpml;
  MuinvPMLFull(PETSC_COMM_SELF, &munivpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega,LowerPML);
  double *muinv;
  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  int add=0;
  AddMuAbsorption(muinv,munivpml,Qabs,add);

  if(blochcondition){
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M, Nx,Ny,Nz, hx,hy,hz, bx,by,bz, muinv, BCPeriod, beta);
  }else{
    MoperatorGeneral(PETSC_COMM_WORLD, &M, Nx,Ny,Nz, hx,hy,hz, bx,by,bz, muinv, BCPeriod);
  }
  ierr = PetscObjectSetName((PetscObject) M, "M"); CHKERRQ(ierr);


  /*----Set up the epsilon PML vectors--------*/
  Vec unitx,unity,unitz;
  ierr = VecDuplicate(vR,&unitx);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unity);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unitz);CHKERRQ(ierr);
  GetUnitVec(unitx,0,6*Nxyz);
  GetUnitVec(unity,1,6*Nxyz);
  GetUnitVec(unitz,2,6*Nxyz);

  ierr = VecDuplicate(vR,&epsDiff); CHKERRQ(ierr);

  VecSet(epsDiff,0.0);
  VecAXPY(epsDiff,epsx,unitx);
  VecAXPY(epsDiff,epsy,unity);
  VecAXPY(epsDiff,epsz,unitz);

  Vec epspml;
  ierr = VecDuplicate(vR,&epspml);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epspmlQ,"EpsPMLQ"); CHKERRQ(ierr);

  EpsPMLFull(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);
  ierr =MatMult(D,epspml,epspmlQ); CHKERRQ(ierr);
  ierr =VecScale(epspmlQ, 1.0/Qabs); CHKERRQ(ierr);
  ierr =VecAXPY(epspmlQ, 1.0, epspml);CHKERRQ(ierr);

  /*-----Set up epsmedium, epsSReal, epsFReal ------*/
  ierr = VecDuplicate(vR,&epsmedium); CHKERRQ(ierr);
  //GetMediumVecwithSub(epsmedium,Nz,Mz,epsair,epssub);
  GetMediumVec(epsmedium,Nz,Mz,epsair,epssub);

  ierr = MatCreateVecs(A,&epsSReal, &epsFReal); CHKERRQ(ierr);

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

  /*---------Setup Done!---------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Everything set up! Ready to calculate the overlap and gradient.--------\n ");CHKERRQ(ierr);


/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/

/*-------------------------------------------------------------------------*/
  ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);									
  ierr=MatMult(A,epsSReal,epsFReal); CHKERRQ(ierr);
  ModifyMatDiag(M, D, epsFReal, epsDiff, epsmedium, epspmlQ, omega, Nx, Ny, Nz);

  ierr=VecPointwiseMult(epsFReal,epsFReal,epsDiff); CHKERRQ(ierr);							
  ierr=VecAXPY(epsFReal,1.0,epsmedium); CHKERRQ(ierr);								
  ierr=VecPointwiseMult(epsFReal,epsFReal,epspmlQ); CHKERRQ(ierr);

  eigsolver(M,epsFReal,D);
/*-------------------------------------------------------------------------*/

  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = MatDestroy(&C); CHKERRQ(ierr);
  ierr = MatDestroy(&D); CHKERRQ(ierr);
  ierr = MatDestroy(&M); CHKERRQ(ierr);  

  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&unitx); CHKERRQ(ierr);
  ierr = VecDestroy(&unity); CHKERRQ(ierr);
  ierr = VecDestroy(&unitz); CHKERRQ(ierr);
  ierr = VecDestroy(&epsDiff); CHKERRQ(ierr);

  ierr = VecDestroy(&munivpml); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium); CHKERRQ(ierr);
  ierr = VecDestroy(&epsSReal); CHKERRQ(ierr);
  ierr = VecDestroy(&epsFReal); CHKERRQ(ierr);

  free(muinv);
  free(epsopt);

  /*------------ finalize the program -------------*/

  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Barrier(PETSC_COMM_WORLD);
  }
  
  ierr = SlepcFinalize(); CHKERRQ(ierr);

  return 0;
}
  


