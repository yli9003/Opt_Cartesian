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

PetscErrorCode ComplexVectorProduct(Vec va, Vec vb, Vec vout, Mat D);

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

  int Npmlx, Npmly, Npmlz, Nx, Ny, Nz, Nxyz;
  double hx, hy, hz;
  double omega;
  double RRT, sigmax, sigmay, sigmaz;
  char epsfile[PETSC_MAX_PATH_LEN];
  Mat M;
  Vec epsNoPML, epspml, epsC;
  
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
  
  PetscOptionsGetInt(PETSC_NULL,"-Npmlx",&Npmlx,&flg);  MyCheckAndOutputInt(flg,Npmlx,"Npmlx","Npmlx");
  PetscOptionsGetInt(PETSC_NULL,"-Npmly",&Npmly,&flg);  MyCheckAndOutputInt(flg,Npmly,"Npmly","Npmly");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlz",&Npmlz,&flg);  MyCheckAndOutputInt(flg,Npmlz,"Npmlz","Npmlz");

  PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg);  MyCheckAndOutputInt(flg,Nx,"Nx","Nx");
  PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg);  MyCheckAndOutputInt(flg,Ny,"Ny","Ny");
  PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg);  MyCheckAndOutputInt(flg,Nz,"Nz","Nz");

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

  RRT=1e-25;
  sigmax = pmlsigma(RRT,(double) Npmlx*hx);
  sigmay = pmlsigma(RRT,(double) Npmly*hy);
  sigmaz = pmlsigma(RRT,(double) Npmlz*hz);

  PetscPrintf(PETSC_COMM_WORLD,"sigma, omega: %g, %g, %g, %g \n", sigmax, sigmay, sigmaz, omega);

  PetscOptionsGetString(PETSC_NULL,"-epsfile",epsfile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,epsfile,"epsfile","Inputdata file");

/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/


  /*------Set up the C, D matrices--------------*/
  Nxyz=Nx*Ny*Nz;
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
  double Qabs=1.0/0.0;
  AddMuAbsorption(muinv,munivpml,Qabs,add);

  if(blochcondition){
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M, Nx,Ny,Nz, hx,hy,hz, bx,by,bz, muinv, BCPeriod, beta);
  }else{
    MoperatorGeneral(PETSC_COMM_WORLD, &M, Nx,Ny,Nz, hx,hy,hz, bx,by,bz, muinv, BCPeriod);
  }
  ierr = PetscObjectSetName((PetscObject) M, "M"); CHKERRQ(ierr);


  /*----Set up the epsilon and PML vectors--------*/
  ierr = VecDuplicate(vR,&epsNoPML); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsC);CHKERRQ(ierr);

  EpsPMLFull(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);

  /*---------Setup the epsArray----------------*/
  double *epsArray;
  FILE *ptf;
  epsArray = (double *) malloc(6*Nxyz*sizeof(double));
  ptf = fopen(epsfile,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input epsilon files \n");
  int i;
  for (i=0;i<6*Nxyz;i++)
    { 
      fscanf(ptf,"%lf",&epsArray[i]);
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
  ierr=ArrayToVec(epsArray, epsNoPML); CHKERRQ(ierr);								     

  ComplexVectorProduct(epsNoPML,epspml,epsC,D);
  AddEpsToM(M,D,epsC,Nxyz,omega); 

  eigsolver(M,epsC,D);
/*-------------------------------------------------------------------------*/

  ierr = MatDestroy(&C); CHKERRQ(ierr);
  ierr = MatDestroy(&D); CHKERRQ(ierr);
  ierr = MatDestroy(&M); CHKERRQ(ierr);  

  ierr = VecDestroy(&vR); CHKERRQ(ierr);

  ierr = VecDestroy(&munivpml); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml); CHKERRQ(ierr);
  ierr = VecDestroy(&epsNoPML); CHKERRQ(ierr);
  ierr = VecDestroy(&epsC); CHKERRQ(ierr);

  free(muinv);
  free(epsArray);

  /*------------ finalize the program -------------*/

  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Barrier(PETSC_COMM_WORLD);
  }
  
  ierr = SlepcFinalize(); CHKERRQ(ierr);

  return 0;
}
  
PetscErrorCode ComplexVectorProduct(Vec va, Vec vb, Vec vout, Mat D)
{
  PetscErrorCode ierr;

  int N;
  ierr=VecGetSize(va, &N); CHKERRQ(ierr);

  Vec vai,vbi;
  ierr=VecDuplicate(va, &vai); CHKERRQ(ierr);
  ierr=VecDuplicate(va, &vbi); CHKERRQ(ierr);
  
  ierr=MatMult(D,va,vai);CHKERRQ(ierr);
  ierr=MatMult(D,vb,vbi);CHKERRQ(ierr);

  double *a, *b, *ai, *bi, *out;
  ierr=VecGetArray(va,&a);CHKERRQ(ierr);
  ierr=VecGetArray(vb,&b);CHKERRQ(ierr);
  ierr=VecGetArray(vai,&ai);CHKERRQ(ierr);
  ierr=VecGetArray(vbi,&bi);CHKERRQ(ierr);
  ierr=VecGetArray(vout,&out);CHKERRQ(ierr);

  int i, ns, ne, nlocal;
  ierr = VecGetOwnershipRange(vout, &ns, &ne);
  nlocal = ne-ns;

  for (i=0; i<nlocal; i++)
    {  
      if(i<(N/2-ns)) // N is the total length of Vec;
	out[i] = a[i]*b[i] - ai[i]*bi[i];
      else
	out[i] = ai[i]*b[i] + a[i]*bi[i];
    }
  
  ierr=VecRestoreArray(va,&a);CHKERRQ(ierr);
  ierr=VecRestoreArray(vb,&b);CHKERRQ(ierr);
  ierr=VecRestoreArray(vai,&ai);CHKERRQ(ierr);
  ierr=VecRestoreArray(vbi,&bi);CHKERRQ(ierr);
  ierr=VecRestoreArray(vout,&out);CHKERRQ(ierr);

  ierr=VecDestroy(&vai);CHKERRQ(ierr);
  ierr=VecDestroy(&vbi);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

