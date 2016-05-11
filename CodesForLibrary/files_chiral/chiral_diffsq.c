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
#define __FUNCT__ "ldoskdiff"
double ldoskdiff(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the LDOS difference solver (Full Vectorial Version). Minimum approach NOT available.********** \n");

  ChiraldataGroup *ptdata = (ChiraldataGroup *) data;

  double omega = ptdata->omega;
  Mat    M = ptdata->M;
  Mat    A = ptdata->A;
  Vec    xL = ptdata->xL;
  Vec    xR = ptdata->xR;
  Vec 	 bL = ptdata->bL;
  Vec    bR = ptdata->bR;
  Vec    weightedJL = ptdata->weightedJL;
  Vec    weightedJR = ptdata->weightedJR;
  Vec    epspmlQ  = ptdata->epspmlQ;
  Vec    epsmedium = ptdata->epsmedium;
  Vec    epsDiff = ptdata->epsDiff;
  int    *its = ptdata->its; 
  Vec    epscoef = ptdata->epscoef;  
  Vec    vgrad = ptdata->vgrad;
  KSP    ksp = ptdata->ksp;
  double chiralweight = ptdata->chiralweight;
  
  //declare temporary variables
  Vec epsC, epsCi, epsP, tmp, Grad;
  Mat tmpM;

  int DegFree = DegFreeAll - (chiralweight>0);

  VecDuplicate(epsDiff,&epsC);
  VecDuplicate(epsDiff,&epsCi);
  VecDuplicate(epsDiff,&epsP);
  VecDuplicate(epsDiff,&tmp);
  VecDuplicate(epsDiff,&Grad);

  
  Vec epsgrad, gradL, gradR;
  VecDuplicate(epsSReal,&epsgrad);
  VecDuplicate(epsSReal,&gradL);
  VecDuplicate(epsSReal,&gradR);
  RegzProj(DegFree,epsoptAll,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);

  // Update the diagonals of M;
  MatDuplicate(M,MAT_COPY_VALUES,&tmpM);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(tmpM, A, D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz, omega, epsDiff);

  /*-----------------KSP Solving------------------*/   
  SolveMatrix(PETSC_COMM_WORLD,ksp,tmpM,bL,xL,its);
  SolveMatrix(PETSC_COMM_WORLD,ksp,tmpM,bR,xR,its);

  /*-------------Calculate and print out the LDOS----------*/
  //tmpldos = -Re((wt.*J^*)'*E) 
  double tmpldosr, tmpldosi;

  double ldosL;
  MatMult(C,weightedJL,tmp);
  CmpVecDot(xL,tmp,&tmpldosr,&tmpldosi);
  ldosL=-1.0*hxyz*tmpldosr;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current ldosL for omega %.4e at step %.5d is %.16e \n", omega/(2*PI),count,ldosL);

  double ldosR;
  MatMult(C,weightedJR,tmp);
  CmpVecDot(xR,tmp,&tmpldosr,&tmpldosi);
  ldosR=-1.0*hxyz*tmpldosr;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current ldosR for omega %.4e at step %.5d is %.16e \n", omega/(2*PI),count,ldosR);

  double ldosdiffsq;
  ldosdiffsq = (ldosL - ldosR)*(ldosL - ldosR);
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current squared ldos difference between left and right at omega %.4e at step %.5d is %.16e \n", omega/(2*PI),count,ldosdiffsq);

  if (gradAll) {
    KSPSolveTranspose(ksp,weightedJL,tmp);
    MatMult(C,tmp,Grad);
    CmpVecProd(Grad,epscoef,tmp);
    CmpVecProd(tmp,xL,Grad);
    VecScale(Grad,-1.0*hxyz);
    ierr = VecPointwiseMult(Grad,Grad,vR); CHKERRQ(ierr);

    ierr = MatMultTranspose(A,Grad,vgrad);CHKERRQ(ierr);

    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);
    KSPSolveTranspose(kspH,vgrad,gradL);

    KSPSolveTranspose(ksp,weightedJR,tmp);
    MatMult(C,tmp,Grad);
    CmpVecProd(Grad,epscoef,tmp);
    CmpVecProd(tmp,xR,Grad);
    VecScale(Grad,-1.0*hxyz);
    ierr = VecPointwiseMult(Grad,Grad,vR); CHKERRQ(ierr);

    ierr = MatMultTranspose(A,Grad,vgrad);CHKERRQ(ierr);

    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);
    KSPSolveTranspose(kspH,vgrad,gradR);

    VecAXPY(gradL,-1.0,gradR);
    VecScale(gradL,2.0*(ldosL-ldosR));
    if (chiralweight) VecScale(gradL,-1.0*chiralweight);

    ierr = VecToArray(gradL,gradAll,scatter,from,to,vgradlocal,DegFree);

    if (chiralweight) gradAll[DegFreeAll-1] = 1.0;

  }

  ierr = MatDestroy(&tmpM); CHKERRQ(ierr);

  VecDestroy(&epsC);
  VecDestroy(&epsCi);
  VecDestroy(&epsP);
  VecDestroy(&tmp);
  VecDestroy(&Grad);
  VecDestroy(&epsgrad);
  VecDestroy(&gradL);
  VecDestroy(&gradR);


  if((chiralweight>0)==0){
    char buffer [100];
    int STORE=1;    
    if(STORE==1 && (count%outputbase==0))
      {
	sprintf(buffer,"%.5depsSReal.m",count);
	OutputVec(PETSC_COMM_WORLD, epsSReal, filenameComm, buffer);
      }
    count++;
  }



  if (chiralweight)
    return epsoptAll[DegFreeAll-1] - chiralweight*ldosdiffsq;
  else
    return ldosdiffsq;
}
