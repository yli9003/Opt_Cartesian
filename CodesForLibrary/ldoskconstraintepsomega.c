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
IS from, to;
VecScatter scatter;
extern char filenameComm[PETSC_MAX_PATH_LEN];
extern int outputbase;

extern int pSIMP;
extern double bproj, etaproj;
extern Mat Hfilt;
extern KSP kspH;
extern int itsH;

#undef __FUNCT__ 
#define __FUNCT__ "ldoskconstraintepsomega"
double ldoskconstraintepsomega(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{

  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the LDOS constraint solver (Full Vectorial Version). Minimum approach NOT available.********** \n");
 
  LDOSdataGroupEpsOmega *ptdata = (LDOSdataGroupEpsOmega *) data;
 
  Mat  M = ptdata->M;
  Mat  A = ptdata->A;
  Vec  x = ptdata->x;
  Vec  b = ptdata->b;
  Vec  weightedJ = ptdata->weightedJ;
  Vec  epspmlQ  = ptdata->epspmlQ;
  Vec  epsmedium = ptdata->epsmedium;
  int  *its = ptdata->its; 
  Vec  vgrad = ptdata->vgrad;
  KSP  ksp = ptdata->ksp;
  int  nfreq = ptdata->nfreq;
  
  //declare temporary variables
  Vec epsDiff, epsbar, epsC, epsCi, epsP, AdjE, tmp, Grad;
  Mat tmpM;
  int DegFree=DegFreeAll-5;

  VecDuplicate(x,&epsDiff);
  VecDuplicate(x,&epsbar);
  VecDuplicate(x,&epsC);
  VecDuplicate(x,&epsCi);
  VecDuplicate(x,&epsP);
  VecDuplicate(x,&AdjE);
  VecDuplicate(x,&tmp);
  VecDuplicate(x,&Grad);

  double dummyt=epsoptAll[DegFreeAll-5];
  double epsdiff=epsoptAll[DegFreeAll-4];
  double omega;
  omega = (nfreq==1)*2*PI*epsoptAll[DegFreeAll-3] + (nfreq==2)*2*PI*epsoptAll[DegFreeAll-2] + (nfreq==3)*2*PI*epsoptAll[DegFreeAll-1];
  VecSet(epsDiff,1.0);
  VecScale(epsDiff,epsdiff);

  PetscPrintf(PETSC_COMM_WORLD,"---*****The current epsilondiff and freq at step %.5d are %.16e and %.16e. \n", count, epsdiff, omega/(2*PI));

  Vec epsgrad;
  VecDuplicate(epsSReal,&epsgrad);
  RegzProj(DegFree,epsoptAll,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);
  MatMult(A,epsSReal,epsbar);

  // Update the diagonals of M;
  MatDuplicate(M,MAT_COPY_VALUES,&tmpM);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(tmpM, A, D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz, omega, epsDiff);

  /*-----------------KSP Solving------------------*/   
  SolveMatrix(PETSC_COMM_WORLD,ksp,tmpM,b,x,its);

  /*-------------Calculate and print out the LDOS----------*/
  //tmpldos = -Re((wt.*J^*)'*E) 
  double tmpldosr, tmpldosi, ldos;
  MatMult(C,weightedJ,tmp);
  CmpVecDot(x,tmp,&tmpldosr,&tmpldosi);
  ldos=-1.0*hxyz*tmpldosr;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current ldos for omega %.4e at step %.5d is %.16e \n", omega/(2*PI),count,ldos);

  if (gradAll) {
    KSPSolveTranspose(ksp,weightedJ,tmp);
    MatMult(C,tmp,AdjE);
    CmpVecProd(AdjE,epspmlQ,tmp);
    CmpVecProd(tmp,x,Grad);
    VecScale(Grad,1.0*hxyz*epsdiff*omega*omega);
    ierr = VecPointwiseMult(Grad,Grad,vR); CHKERRQ(ierr);

    ierr = MatMultTranspose(A,Grad,vgrad);CHKERRQ(ierr);

    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);
    KSPSolveTranspose(kspH,vgrad,epsgrad);

    ierr = VecToArray(epsgrad,gradAll,scatter,from,to,vgradlocal,DegFree);

    double epsdiffgrad;
    CmpVecProd(AdjE,epspmlQ,tmp);
    VecPointwiseMult(tmp,tmp,epsbar);
    CmpVecProd(tmp,x,Grad);
    VecScale(Grad,1.0*hxyz*omega*omega);
    VecPointwiseMult(Grad,Grad,vR);
    VecSum(Grad,&epsdiffgrad);

    double omegagrad;
    VecCopy(epsbar,epsC);
    VecScale(epsC,epsdiff);
    VecAXPY(epsC,1.0,epsmedium);
    VecPointwiseMult(epsC,epsC,epspmlQ);
    VecScale(epsC,2.0*omega);
    CmpVecProd(epsC,AdjE,tmp);
    CmpVecProd(tmp,x,Grad);
    VecScale(Grad,1.0*hxyz);
    VecPointwiseMult(Grad,Grad,vR);
    VecSum(Grad,&omegagrad);

    gradAll[DegFreeAll-5]=1;
    gradAll[DegFreeAll-4]=epsdiffgrad;
    gradAll[DegFreeAll-3]=0;
    gradAll[DegFreeAll-2]=0;
    gradAll[DegFreeAll-1]=0;
    if(nfreq==1) gradAll[DegFreeAll-3]=2*PI*omegagrad;
    if(nfreq==2) gradAll[DegFreeAll-2]=2*PI*omegagrad;
    if(nfreq==3) gradAll[DegFreeAll-1]=2*PI*omegagrad;

}

  ierr = MatDestroy(&tmpM); CHKERRQ(ierr);
  
  VecDestroy(&epsDiff);
  VecDestroy(&epsbar);
  VecDestroy(&AdjE);
  VecDestroy(&epsC);
  VecDestroy(&epsCi);
  VecDestroy(&epsP);
  VecDestroy(&tmp);
  VecDestroy(&Grad);
  VecDestroy(&epsgrad);

  double output=dummyt-ldos;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current dummyt-ldos and gradAll is %.16e, %.16e \n", output, gradAll[DegFreeAll-4]);
  return output;
}

