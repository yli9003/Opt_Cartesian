#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

int maxit=15;
Mat D;

PetscErrorCode setupKSP(MPI_Comm comm, KSP *ksp, PC *pc, int solver, int iteronly);
PetscErrorCode ComplexVectorProduct(Vec va, Vec vb, Vec vout, Mat D);

/*Unused Global Varibles because of library*/
Mat A, B, C;
Vec vR, epsFReal;

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize ------*/
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;


  int myrank, mma_verbose;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0) 
    mma_verbose=1;

/*****************************************************-------Set up the options parameters-------------********************************************************/
  PetscBool flg;

  PetscPrintf(PETSC_COMM_WORLD,"***NOTE for INPUT FORMAT: supply the epsilon input file and the current file in a single-column 6Nxyz format [fXr(z1,y1,x1),f(z2,y1,x1),...,fYr(z1,y1,x1),...,fZr(z1,y1,x1),...,fXi(z1,y1,x1),...]; epsilon should be the most general complex anisotropic (up to biaxial) ******\n");

  int iteronly;
  PetscOptionsGetInt(PETSC_NULL,"-iteronly",&iteronly,&flg);
  if(flg) MyCheckAndOutputInt(flg,iteronly,"iteronly","iteronly");
  if(!flg) iteronly=0;

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

  int Npmlx, Npmly, Npmlz, Nx, Ny, Nz, Nxyz;
  double hx, hy, hz, hxyz;
  PetscOptionsGetInt(PETSC_NULL,"-Npmlx",&Npmlx,&flg);  MyCheckAndOutputInt(flg,Npmlx,"Npmlx","Npmlx");
  PetscOptionsGetInt(PETSC_NULL,"-Npmly",&Npmly,&flg);  MyCheckAndOutputInt(flg,Npmly,"Npmly","Npmly");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlz",&Npmlz,&flg);  MyCheckAndOutputInt(flg,Npmlz,"Npmlz","Npmlz");
  PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg);  MyCheckAndOutputInt(flg,Nx,"Nx","Nx");
  PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg);  MyCheckAndOutputInt(flg,Ny,"Ny","Ny");
  PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg);  MyCheckAndOutputInt(flg,Nz,"Nz","Nz");
  Nxyz=Nx*Ny*Nz;
  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  PetscOptionsGetReal(PETSC_NULL,"-hy",&hy,&flg);  MyCheckAndOutputDouble(flg,hy,"hy","hy");
  PetscOptionsGetReal(PETSC_NULL,"-hz",&hz,&flg);  MyCheckAndOutputDouble(flg,hz,"hz","hz");
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;

  int BCPeriod, LowerPML;
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod");
  PetscOptionsGetInt(PETSC_NULL,"-LowerPML",&LowerPML,&flg);  MyCheckAndOutputInt(flg,LowerPML,"LowerPML","LowerPML");

  int bx[2], by[2], bz[2];
  PetscOptionsGetInt(PETSC_NULL,"-bxl",bx,&flg);    MyCheckAndOutputInt(flg,bx[0],"bxl","BC at x lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-bxu",bx+1,&flg);  MyCheckAndOutputInt(flg,bx[1],"bxu","BC at x upper for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-byl",by,&flg);    MyCheckAndOutputInt(flg,by[0],"byl","BC at y lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-byu",by+1,&flg);  MyCheckAndOutputInt(flg,by[1],"byu","BC at y upper for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-bzl",bz,&flg);    MyCheckAndOutputInt(flg,bz[0],"bzl","BC at z lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-bzu",bz+1,&flg);  MyCheckAndOutputInt(flg,bz[1],"bzu","BC at z upper for mode 1");

  double freq, omega;
  PetscOptionsGetReal(PETSC_NULL,"-freq",&freq,&flg);
  if(!flg) freq=1.0;
  PetscPrintf(PETSC_COMM_WORLD,"-------freq: %g \n",freq);
  omega=2.0*PI*freq;

  double RRT, sigmax, sigmay, sigmaz;
  RRT=1e-25;
  sigmax = pmlsigma(RRT,(double) Npmlx*hx);
  sigmay = pmlsigma(RRT,(double) Npmly*hy);
  sigmaz = pmlsigma(RRT,(double) Npmlz*hz);

  PetscPrintf(PETSC_COMM_WORLD,"sigma, omega, Nxyz: %g, %g, %g, %g %d \n", sigmax, sigmay, sigmaz, omega, Nxyz);

  char filenameComm[PETSC_MAX_PATH_LEN], epsfile[PETSC_MAX_PATH_LEN], inputsrc[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-filenameprefix",filenameComm,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,filenameComm,"filenameprefix","Filename Prefix");
  PetscOptionsGetString(PETSC_NULL,"-epsfile",epsfile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,epsfile,"epsfile","Input epsilon file");
  PetscOptionsGetString(PETSC_NULL,"-inputsrc",inputsrc,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrc,"inputsrc","Input source current");

  int solver;
  PetscOptionsGetInt(PETSC_NULL,"-solver",&solver,&flg);  MyCheckAndOutputInt(flg,solver,"solver","LU Direct solver choice (0 PASTIX, 1 MUMPS, 2 SUPERLU_DIST)");

/**************************************************************************************************************************************************************/
  Vec epsNoPML, epspml, J, b, x;
  Mat M, D;

  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &epsNoPML);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&epspml);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&J);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&x);CHKERRQ(ierr);

  /*----------Imaginary number matrix D---------------*/
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  /*---------Setup the epsArray and epsNoPML----------------*/
  double *epsArray;
  FILE *ptf;
  epsArray = (double *) malloc(6*Nxyz*sizeof(double));
  ptf = fopen(epsfile,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from epsilon file \n");
  int i;
  for (i=0;i<6*Nxyz;i++)
    { 
      fscanf(ptf,"%lf",&epsArray[i]);
    }
  fclose(ptf);
  ierr=ArrayToVec(epsArray, epsNoPML); CHKERRQ(ierr);								
  PetscPrintf(PETSC_COMM_WORLD,"epsNoPML constructed. \n");

  /*--------Setup the epsilon PML------------------*/
  EpsPMLFull(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);

  /*-------Setup J and b-------------*/
  double *JArray;
  FILE *Jptf;
  JArray = (double *) malloc(6*Nxyz*sizeof(double));
  Jptf = fopen(inputsrc,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from current file \n");
  int inJi;
  for (inJi=0;inJi<6*Nxyz;inJi++)
    { 
      fscanf(Jptf,"%lf",&JArray[inJi]);
    }
  fclose(Jptf);
  ArrayToVec(JArray,J);
  free(JArray);
  ierr = MatMult(D,J,b);CHKERRQ(ierr);
  VecScale(b,omega);
  PetscPrintf(PETSC_COMM_WORLD,"J and b constructed. \n");

  /*--------Setup M matrix-----------*/
  Vec muinvpml;
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega,LowerPML);
  double *muinv;
  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  double Qabs=1.0/0.0;
  int add=0;
  AddMuAbsorption(muinv,muinvpml,Qabs,add);

  if(blochcondition){
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M, Nx,Ny,Nz, hx,hy,hz, bx,by,bz, muinv, BCPeriod, beta);
  }else{
    MoperatorGeneral(PETSC_COMM_WORLD, &M, Nx,Ny,Nz, hx,hy,hz, bx,by,bz, muinv, BCPeriod);
  }
  ierr = PetscObjectSetName((PetscObject) M, "M"); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"M matrix constructed. \n"); 

  /*--------Setup the KSP variables ---------------*/
  KSP ksp;
  PC pc;
  int its=100;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the KSP variables.--------\n ");CHKERRQ(ierr);
  setupKSP(PETSC_COMM_WORLD,&ksp,&pc,solver,iteronly);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the KSP variables DONE!--------\n ");CHKERRQ(ierr);

  /*---------Setup Done!---------*/
  PetscPrintf(PETSC_COMM_WORLD,"--------Everything set up! Ready to solve Maxwell's Equations.--------\n ");


/**************************************************************************************************************************************************************/

  int Mrows, Mcols;
  MatGetSize(M,&Mrows,&Mcols);
  PetscPrintf(PETSC_COMM_WORLD,"****Dimensions of M is %d by %d \n",Mrows,Mcols);

  Vec epsC;
  VecDuplicate(epsNoPML,&epsC);
  ComplexVectorProduct(epsNoPML,epspml,epsC,D);

  AddEpsToM(M,D,epsC,Nxyz,omega);
  SolveMatrix(PETSC_COMM_WORLD,ksp,M,b,x,&its);

  OutputVec(PETSC_COMM_WORLD, x, filenameComm,"_Efield.m"); 

  ierr = VecDestroy(&epsNoPML);CHKERRQ(ierr);
  ierr = VecDestroy(&epspml);CHKERRQ(ierr);
  ierr = VecDestroy(&epsC);CHKERRQ(ierr);
  ierr = VecDestroy(&muinvpml);CHKERRQ(ierr);
  ierr = VecDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  free(muinv);
  /*------------ finalize the program -------------*/

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
