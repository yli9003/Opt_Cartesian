#include <petsc.h>
#include <time.h>
#include "libOPT.h"
#include <complex.h>
#include "petsctime.h"

#define Ptime PetscTime

extern int count;
extern Mat C,D;
extern Vec vR, weight, vgradlocal;
extern VecScatter scatter;
extern IS from, to;

extern int pSIMP;
extern double bproj, etaproj;
extern Mat Hfilt;
extern KSP kspH;
extern int itsH;

extern char filenameComm[PETSC_MAX_PATH_LEN];

extern double mintrans;

#undef __FUNCT__ 
#define __FUNCT__ "refphiopt"
double refphiopt(int ndof,double *refphi, double *grad, void *data)
{
  
  PetscErrorCode ierr;

  Meta *ptdata = (Meta *) data;

  double omega = ptdata->omega;
  Vec x = ptdata->x;
  Vec pvec = ptdata->pvec;
  Vec qvec = ptdata->qvec;

  PetscPrintf(PETSC_COMM_WORLD,"----Modifying qvec. ------- \n");

  double phi=*refphi;

  Vec xconj,xmag,xmagsq,xmagrecp,xpq,xpqmagsq,phasesumvec,tmp;
  VecDuplicate(vR,&xconj);
  VecDuplicate(vR,&xmag);
  VecDuplicate(vR,&xmagsq);
  VecDuplicate(vR,&xmagrecp);
  VecDuplicate(vR,&xpq);
  VecDuplicate(vR,&xpqmagsq);
  VecDuplicate(vR,&phasesumvec);
  VecDuplicate(vR,&tmp);

  CmpVecScale(qvec, tmp, cos(phi), sin(phi));
  VecCopy(tmp,qvec);

  VecWAXPY(xpq,1.0,x,qvec);
  MatMult(C,xpq,tmp);
  CmpVecProd(xpq,tmp,xpqmagsq);
  MatMult(C,x,xconj);
  CmpVecProd(x,xconj,xmagsq);
  VecWAXPY(phasesumvec,-1.0,xmagsq,xpqmagsq);
  VecAXPY(phasesumvec,-1.0,pvec);
  VecCopy(xmagsq,xmag);
  VecSqrtAbs(xmag);
  MatMult(D,xmag,tmp);
  VecWAXPY(xmagrecp,1.0,xmag,tmp);
  VecReciprocal(xmagrecp);
  VecPointwiseMult(xmagrecp,xmagrecp,vR);
  CmpVecProd(xmagrecp,phasesumvec,tmp);
  CmpVecProd(tmp,pvec,phasesumvec);
  VecPointwiseMult(phasesumvec,phasesumvec,vR);

  double phasesum;
  VecSum(phasesumvec,&phasesum);
  PetscPrintf(PETSC_COMM_WORLD,"---refphi for freq %g and at step %d is %g.\n", omega/(2*PI),count,phi);
  PetscPrintf(PETSC_COMM_WORLD,"---phase sum for freq %.4e and at step %d is: %.8e\n", omega/(2*PI),count,phasesum);

  count++;

  VecDestroy(&xconj);
  VecDestroy(&xmag);
  VecDestroy(&xmagsq);
  VecDestroy(&xmagrecp);
  VecDestroy(&xpq);
  VecDestroy(&xpqmagsq);
  VecDestroy(&phasesumvec);
  VecDestroy(&tmp);

  if(grad)
      PetscPrintf(PETSC_COMM_WORLD,"---ERROR: you must not use a gradient algorithm for ref optimization.\n");

  return phasesum;
}

