#include <petsc.h>
#include <time.h>
#include "libOPT.h"
#include <complex.h>
#include "petsctime.h"

#define Ptime PetscTime

//define Global variables;
extern int Nxyz, count;
extern double hxyz;
extern Mat B, C, D;
extern Vec vR, weight;
extern Vec vgradlocal;
extern Vec epsSReal;
extern IS from, to;
extern VecScatter scatter;
extern char filenameComm[PETSC_MAX_PATH_LEN];
extern int outputbase;

extern int pSIMP;
extern double bproj, etaproj;
extern Mat Hfilt;
extern KSP kspH;
extern int itsH;

#undef __FUNCT__ 
#define __FUNCT__ "ldoskchiralconstraint"
double ldoskchiralconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the LDOS chiral solver (Full Vectorial Version).********** \n");

  ChiraldataGroup *ptdata = (ChiraldataGroup *) data;

  double omega = ptdata->omega;
  Mat    M = ptdata->M;
  Mat    A = ptdata->A;
  Vec    xL = ptdata->xL;
  Vec 	 bL = ptdata->bL;
  Vec    weightedJL = ptdata->weightedJL;
  Vec    epspmlQ  = ptdata->epspmlQ;
  Vec    epsmedium = ptdata->epsmedium;
  Vec    epsDiff = ptdata->epsDiff;
  int    *its = ptdata->its; 
  Vec    epscoef = ptdata->epscoef;  
  Vec    vgrad = ptdata->vgrad;
  KSP    ksp = ptdata->ksp;
  double chiralweight = ptdata->chiralweight;
  int    fomopt = ptdata->fomopt;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The DegFreeAll is %d \n", DegFreeAll);

  //declare temporary variables
  Vec epsC, epsCi, epsP, tmp, Grad;
  Mat tmpM;

  int DegFree = DegFreeAll - 1;

  VecDuplicate(epsDiff,&epsC);
  VecDuplicate(epsDiff,&epsCi);
  VecDuplicate(epsDiff,&epsP);
  VecDuplicate(epsDiff,&tmp);
  VecDuplicate(epsDiff,&Grad);

  
  Vec epsgrad, gradChiral;
  VecDuplicate(epsSReal,&epsgrad);
  VecDuplicate(epsSReal,&gradChiral);
  RegzProj(DegFree,epsoptAll,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);

  // Update the diagonals of M;
  MatDuplicate(M,MAT_COPY_VALUES,&tmpM);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(tmpM, A, D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz, omega, epsDiff);

  /*-----------------KSP Solving------------------*/   
  SolveMatrix(PETSC_COMM_WORLD,ksp,tmpM,bL,xL,its);

  /*-------------Calculate and print out the LDOS----------*/
  //tmpldos = -Re((wt.*J^*)'*E) 
  double tmpldosr, tmpldosi;

  double ldosChiral;
  MatMult(C,weightedJL,tmp);
  CmpVecDot(xL,tmp,&tmpldosr,&tmpldosi);
  ldosChiral=-1.0*hxyz*tmpldosr;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current ldos for omega %.4e at step %.5d is %.16e \n", omega/(2*PI),count,ldosChiral);

  if (gradAll) {
    KSPSolveTranspose(ksp,weightedJL,tmp);
    MatMult(C,tmp,Grad);
    CmpVecProd(Grad,epscoef,tmp);
    CmpVecProd(tmp,xL,Grad);
    VecScale(Grad,-1.0*hxyz);
    ierr = VecPointwiseMult(Grad,Grad,vR); CHKERRQ(ierr);

    ierr = MatMultTranspose(A,Grad,vgrad);CHKERRQ(ierr);

    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);
    KSPSolveTranspose(kspH,vgrad,gradChiral);

    if (fomopt==0){
      VecScale(gradChiral,-1.0);
    }else{
      VecScale(gradChiral, 1.0);
    }

    ierr = VecToArray(gradChiral,gradAll,scatter,from,to,vgradlocal,DegFree);

    if (fomopt==0){
      gradAll[DegFreeAll-1]=chiralweight;
    }else{
      gradAll[DegFreeAll-1]=chiralweight/epsoptAll[DegFreeAll-1];
    }


  }

  ierr = MatDestroy(&tmpM); CHKERRQ(ierr);

  VecDestroy(&epsC);
  VecDestroy(&epsCi);
  VecDestroy(&epsP);
  VecDestroy(&tmp);
  VecDestroy(&Grad);
  VecDestroy(&epsgrad);
  VecDestroy(&gradChiral);

  double fom=(fomopt==0)*(chiralweight*epsoptAll[DegFreeAll-1]-ldosChiral) + (fomopt>0)*(ldosChiral-chiralweight/epsoptAll[DegFreeAll-1]);

  return fom;
}
