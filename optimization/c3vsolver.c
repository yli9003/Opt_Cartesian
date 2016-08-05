#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

int maxit=15;
Mat C, D;
Vec vR;

PetscErrorCode setupKSP(MPI_Comm comm, KSP *ksp, PC *pc, int solver, int iteronly);
PetscErrorCode ComplexVectorProduct(Vec va, Vec vb, Vec vout, Mat D);

/*Unused Global Varibles because of library*/
Mat A, B;
Vec epsFReal;

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

  int solver;
  PetscOptionsGetInt(PETSC_NULL,"-solver",&solver,&flg);  MyCheckAndOutputInt(flg,solver,"solver","LU Direct solver choice (0 PASTIX, 1 MUMPS, 2 SUPERLU_DIST)");

/**************************************************************************************************************************************************************/
  Vec epsNoPML, kappa, epspml, J1, J2, b1, b2, x1, x2, tmp, Jconj, u1, u2, grad1, grad2;
  Mat M, D;

  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &epsNoPML);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&epspml);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&kappa);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&J1);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&b1);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&x1);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&J2);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&b2);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&x2);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&vR);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&tmp);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&Jconj);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&u1);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&u2);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&grad1);CHKERRQ(ierr);
  ierr = VecDuplicate(epsNoPML,&grad2);CHKERRQ(ierr);


  GetRealPartVec(vR,6*Nxyz);

  /*----------Set up matrices C, D---------------*/
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);
  CongMat(PETSC_COMM_WORLD, &C, 6*Nxyz);

  /*---------Setup the epsArray and epsNoPML----------------*/
  double *epsArray, *J1Array, *J2Array, *kappaArray;
  FILE *ptfeps, *ptfJ1, *ptfJ2, *ptfkappa;
  epsArray = (double *) malloc(6*Nxyz*sizeof(double));
  J1Array = (double *) malloc(6*Nxyz*sizeof(double));
  J2Array = (double *) malloc(6*Nxyz*sizeof(double));
  kappaArray = (double *) malloc(6*Nxyz*sizeof(double));
  ptfeps = fopen("eps.txt","r");
  ptfJ1 = fopen("J1.txt","r");
  ptfJ2 = fopen("J2.txt","r");
  ptfkappa = fopen("kappa.txt","r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from epsilon, J1 and J2 files \n");
  int i;
  for (i=0;i<6*Nxyz;i++)
    { 
      fscanf(ptfeps,"%lf",&epsArray[i]);
      fscanf(ptfJ1,"%lf",&J1Array[i]);
      fscanf(ptfJ2,"%lf",&J2Array[i]);
      fscanf(ptfkappa,"%lf",&kappaArray[i]);
    }
  fclose(ptfeps);
  fclose(ptfJ1);
  fclose(ptfJ2);
  fclose(ptfkappa);
  ierr=ArrayToVec(epsArray, epsNoPML); CHKERRQ(ierr);								
  ierr=ArrayToVec(J1Array, J1); CHKERRQ(ierr);								
  ierr=ArrayToVec(J2Array, J2); CHKERRQ(ierr);								
  ierr=ArrayToVec(kappaArray, kappa); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"epsNoPML, kappa, J1 and J2 have been constructed. \n");
  free(epsArray);
  free(J1Array);
  free(J2Array);
  free(kappaArray);

  /*--------Setup the epsilon PML------------------*/
  EpsPMLFull(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);

  /*-------Setup b1 and b2-------------*/
  ierr = MatMult(D,J1,b1);CHKERRQ(ierr);
  ierr = MatMult(D,J2,b2);CHKERRQ(ierr);
  VecScale(b1,omega);
  VecScale(b2,omega);
  PetscPrintf(PETSC_COMM_WORLD,"b1 and b2 constructed. \n");

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

  SolveMatrix(PETSC_COMM_WORLD,ksp,M,b1,x1,&its);
  SolveMatrix(PETSC_COMM_WORLD,ksp,M,b2,x2,&its);

  double JEr;
  MatMult(C,J1,Jconj);
  ComplexVectorProduct(Jconj,x1,tmp,D);
  VecPointwiseMult(tmp,tmp,vR);
  VecSum(tmp,&JEr);
  JEr = -hxyz*JEr;
  PetscPrintf(PETSC_COMM_WORLD,"****ldos1 %g \n",JEr);
  MatMult(C,J2,Jconj);
  ComplexVectorProduct(Jconj,x2,tmp,D);
  VecPointwiseMult(tmp,tmp,vR);
  VecSum(tmp,&JEr);
  JEr = -hxyz*JEr;
  PetscPrintf(PETSC_COMM_WORLD,"****ldos2 %g \n",JEr);

  //Calculate grad1 and grad2
  ComplexVectorProduct(kappa,epspml,tmp,D);
  VecCopy(tmp,kappa);

  KSPSolveTranspose(ksp,J1,u1);
  MatMult(C,u1,grad1);
  ComplexVectorProduct(grad1,kappa,u1,D);
  ComplexVectorProduct(u1,x1,grad1,D);
  VecPointwiseMult(grad1,grad1,vR);
  VecScale(grad1,-hxyz);
  KSPSolveTranspose(ksp,J2,u2);
  MatMult(C,u2,grad2);
  ComplexVectorProduct(grad2,kappa,u2,D);
  ComplexVectorProduct(u2,x2,grad2,D);
  VecPointwiseMult(grad2,grad2,vR);
  VecScale(grad2,-hxyz);

  OutputVec(PETSC_COMM_WORLD, grad1, "grad1",".m"); 
  OutputVec(PETSC_COMM_WORLD, grad2, "grad2",".m"); 

  ierr = VecDestroy(&epsNoPML);CHKERRQ(ierr);
  ierr = VecDestroy(&epspml);CHKERRQ(ierr);
  ierr = VecDestroy(&kappa);CHKERRQ(ierr);
  ierr = VecDestroy(&epsC);CHKERRQ(ierr);
  ierr = VecDestroy(&muinvpml);CHKERRQ(ierr);
  ierr = VecDestroy(&J1);CHKERRQ(ierr);
  ierr = VecDestroy(&b1);CHKERRQ(ierr);
  ierr = VecDestroy(&x1);CHKERRQ(ierr);
  ierr = VecDestroy(&J2);CHKERRQ(ierr);
  ierr = VecDestroy(&b2);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);
  ierr = VecDestroy(&vR);CHKERRQ(ierr);
  ierr = VecDestroy(&u1);CHKERRQ(ierr);
  ierr = VecDestroy(&u2);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp);CHKERRQ(ierr);
  ierr = VecDestroy(&Jconj);CHKERRQ(ierr);
  ierr = VecDestroy(&grad1);CHKERRQ(ierr);
  ierr = VecDestroy(&grad2);CHKERRQ(ierr);

  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
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
