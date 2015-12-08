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

//options constr 0, 1 or 2
//LDOS: 0 -> bare ldos, 1 -> a*t - ldos, 2 -> a - ldos
//SOF: 0 -> bare sof, 1 -> sof - b/t, 2 -> sof - b*t
//use constr 2 for SOF minimax

#undef __FUNCT__ 
#define __FUNCT__ "EPSOF"
double EPSOF(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the Self-Orthogonality Factor (SOF) calculator with/without constraint condition.********** \n");

  EPdataGroup *ptdata = (EPdataGroup *) data;

  double omega = ptdata->omega;
  Mat 	 M = ptdata->M;
  Mat    A = ptdata->A;
  Vec    x = ptdata->x;
  Vec 	 b = ptdata->b;
  Vec    epspmlQ  = ptdata->epspmlQ;
  Vec    epsmedium = ptdata->epsmedium;
  Vec    epsDiff = ptdata->epsDiff;
  int    *its = ptdata->its; 
  Vec    epscoef = ptdata->epscoef;  
  Vec    vgrad = ptdata->vgrad;
  KSP    ksp = ptdata->ksp;
  int    constr=ptdata->constr;
  double multipurposescalar = ptdata->multipurposescalar;

  //declare temporary variables
  Vec epsC, epsCi, epsP, tmp, xconj, Grad;
  Mat tmpM;
  int DegFree=(constr)*DegFreeAll-1 + (!constr)*DegFreeAll;

  VecDuplicate(x,&epsC);
  VecDuplicate(x,&epsCi);
  VecDuplicate(x,&epsP);
  VecDuplicate(x,&tmp);
  VecDuplicate(x,&xconj);
  VecDuplicate(x,&Grad);

  Vec epsgrad;
  VecDuplicate(epsSReal,&epsgrad);
  RegzProjnoH(DegFree,epsoptAll,epsSReal,epsgrad,pSIMP,bproj,etaproj);

  // Update the diagonals of M;
  MatDuplicate(M,MAT_COPY_VALUES,&tmpM);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(tmpM, A, D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz, omega, epsDiff);

  /*-----------------KSP Solving------------------*/   
  SolveMatrix(PETSC_COMM_WORLD,ksp,tmpM,b,x,its);

  /*-------------Calculate and print out the SOF----------*/
  //tmpSOF = Magnitude[ E . E ]^2; 
  double SOFr, SOFi, SOF0, SOF;
  VecPointwiseMult(tmp,x,weight);
  CmpVecDot(tmp,x,&SOFr,&SOFi);
  SOF0=sqrt(0.5*hxyz*hxyz*(SOFr*SOFr + SOFi*SOFi));
  SOF=hxyz*SOF0;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current SOF for omega %.4e at step %.5d is %.16e \n", omega/(2*PI),count,SOF);

  if (gradAll) {
    KSPSolve(ksp,x,tmp);
    CmpVecProd(tmp,x,Grad);
    CmpVecProd(Grad,epscoef,tmp);
    ierr = VecPointwiseMult(tmp,tmp,weight); CHKERRQ(ierr);
    VecScale(tmp,2.0*hxyz*hxyz);
    
    MatMult(D,vR,xconj);
    VecScale(xconj,-1.0*SOFi);
    VecAXPY(xconj,SOFr,vR);

    CmpVecProd(xconj,tmp,Grad);
    ierr = VecPointwiseMult(Grad,Grad,vR); CHKERRQ(ierr);
    VecScale(Grad,0.5*hxyz/SOF0);

    ierr = MatMultTranspose(A,Grad,vgrad);CHKERRQ(ierr);

    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);
    VecCopy(vgrad,epsgrad);

    ierr = VecToArray(epsgrad,gradAll,scatter,from,to,vgradlocal,DegFree);

    if(constr==1){
      gradAll[DegFree]=multipurposescalar/(epsoptAll[DegFree]*epsoptAll[DegFree]);
    }else if(constr==2){
      gradAll[DegFree]=multipurposescalar;
    }
  }


  ierr = MatDestroy(&tmpM); CHKERRQ(ierr);

  VecDestroy(&epsC);
  VecDestroy(&epsCi);
  VecDestroy(&epsP);
  VecDestroy(&tmp);
  VecDestroy(&Grad);
  VecDestroy(&epsgrad);
  VecDestroy(&xconj);

  double output=0;
  if(constr==0){
    output=SOF;
  }else if(constr==1){
    output=SOF - multipurposescalar/epsoptAll[DegFree];
  }else if(constr==2){
    output=SOF - multipurposescalar*epsoptAll[DegFree];
  }
  return output;
}

#undef __FUNCT__ 
#define __FUNCT__ "EPLDOS"
double EPLDOS(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the EP LDOS solver with/without constraint condition. ********** \n");

  EPdataGroup *ptdata = (EPdataGroup *) data;

  double omega = ptdata->omega;
  Mat 	 M = ptdata->M;
  Mat    A = ptdata->A;
  Vec    x = ptdata->x;
  Vec 	 b = ptdata->b;
  Vec    weightedJ=ptdata->weightedJ;
  Vec    epspmlQ  = ptdata->epspmlQ;
  Vec    epsmedium = ptdata->epsmedium;
  Vec    epsDiff = ptdata->epsDiff;
  int    *its = ptdata->its; 
  Vec    epscoef = ptdata->epscoef;  
  Vec    vgrad = ptdata->vgrad;
  KSP    ksp = ptdata->ksp;
  int    constr=ptdata->constr;
  double multipurposescalar = ptdata->multipurposescalar;

  //declare temporary variables
  Vec epsC, epsCi, epsP, tmp, Grad;
  Mat tmpM;
  int DegFree=(constr)*DegFreeAll-1 + (!constr)*DegFreeAll;

  VecDuplicate(x,&epsC);
  VecDuplicate(x,&epsCi);
  VecDuplicate(x,&epsP);
  VecDuplicate(x,&tmp);
  VecDuplicate(x,&Grad);

  // copy epsoptAll to epsSReal, fills the first DegFree elements;
  //ierr=ArrayToVec(epsoptAll, epsSReal); CHKERRQ(ierr);
  
  Vec epsgrad;
  VecDuplicate(epsSReal,&epsgrad);
  RegzProjnoH(DegFree,epsoptAll,epsSReal,epsgrad,pSIMP,bproj,etaproj);
  
  // Update the diagonals of M;
  MatDuplicate(M,MAT_COPY_VALUES,&tmpM);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(tmpM, A, D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz, omega, epsDiff);

  /*-----------------KSP Solving------------------*/   
  SolveMatrix(PETSC_COMM_WORLD,ksp,tmpM,b,x,its);

  /*-------------Calculate and print out the LDOS----------*/
  //tmpldos = -Re((wt.*J^*)'*E) 
  double tmpldosr, tmpldosi, ldos;
  CmpVecDot(x,weightedJ,&tmpldosr,&tmpldosi);
  ldos=-1.0*hxyz*tmpldosr;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current ldos for omega %.4e at step %.5d is %.16e \n", omega/(2*PI),count,ldos);

  if (gradAll) {
    CmpVecProd(x,x,Grad);
    CmpVecProd(Grad,epscoef,tmp);
    ierr = MatMult(D,tmp,Grad); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Grad,Grad,weight); CHKERRQ(ierr);
    VecScale(Grad,hxyz/omega);
    if(constr) VecScale(Grad,-1.0);
    ierr = VecPointwiseMult(Grad,Grad,vR); CHKERRQ(ierr);

    ierr = MatMultTranspose(A,Grad,vgrad);CHKERRQ(ierr);

    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);
    VecCopy(vgrad,epsgrad);

    ierr = VecToArray(epsgrad,gradAll,scatter,from,to,vgradlocal,DegFree);

    if(constr==1){
      gradAll[DegFree]=multipurposescalar;
    }else if(constr==2){
      gradAll[DegFree]=0;
    }
  }

  ierr = MatDestroy(&tmpM); CHKERRQ(ierr);

  VecDestroy(&epsC);
  VecDestroy(&epsCi);
  VecDestroy(&epsP);
  VecDestroy(&tmp);
  VecDestroy(&Grad);
  VecDestroy(&epsgrad);

  double output=0;
  if(constr==0){
    output=ldos;
  }else if(constr==1){
    output=multipurposescalar*epsoptAll[DegFree]-ldos;
  }else if(constr==2){
    output=multipurposescalar-ldos;
  }
  return output;
}


