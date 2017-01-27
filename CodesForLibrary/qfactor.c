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

#undef __FUNCT__ 
#define __FUNCT__ "qfactor"
double qfactor(int DegFree,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;

  QGroup *ptdata = (QGroup *) data;

  int Nx = ptdata->Nx;
  int Ny = ptdata->Ny;
  int Nz = ptdata->Nz;
  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  double omega = ptdata->omega;
  Mat M = ptdata->M;
  Mat A = ptdata->A;
  Vec b = ptdata->b;
  Vec J = ptdata->J;
  Vec x = ptdata->x;
  Vec epspmlQ  = ptdata->epspmlQ;
  Vec epsmedium = ptdata->epsmedium;
  Vec epsDiff = ptdata->epsDiff;
  Vec epscoef = ptdata->epscoef;  
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its; 
  Vec VecVol = ptdata->VecVol;
  int outputbase = ptdata->outputbase;

  Vec wtJ, wtJconj, xconj, xmag, Uone, u1, Grad1, Grad2, tmp, vgrad, epsgrad;
  VecDuplicate(vR,&wtJ);
  VecDuplicate(vR,&wtJconj);
  VecDuplicate(vR,&xconj);
  VecDuplicate(vR,&xmag);
  VecDuplicate(vR,&Uone);
  VecDuplicate(vR,&u1);
  VecDuplicate(vR,&Grad1);
  VecDuplicate(vR,&Grad2);
  VecDuplicate(vR,&tmp);
  VecDuplicate(epsSReal,&vgrad);
  VecDuplicate(epsSReal,&epsgrad);

  PetscPrintf(PETSC_COMM_WORLD,"----Calculating Qfactor. ------- \n");
  
  RegzProj(DegFree,epsopt,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);

  MatMult(A,epsSReal,epsFReal);

  Mat Mtmp;
  MatDuplicate(M,MAT_COPY_VALUES,&Mtmp);
  ModifyMatDiag(Mtmp, D, epsFReal, epsDiff, epsmedium, epspmlQ, omega, Nx, Ny, Nz);

  // solve the two fundamental modes and their ldos
  SolveMatrix(PETSC_COMM_WORLD,ksp,Mtmp,b,x,its);

  MatMult(C,x,xconj);
  CmpVecProd(x,xconj,xmag);
  VecPointwiseMult(xmag,xmag,vR);
  VecPointwiseMult(xmag,xmag,VecVol);
  double xmagscalar;
  VecSum(xmag,&xmagscalar);

  VecPointwiseMult(wtJ,J,weight);
  MatMult(C,wtJ,wtJconj);
  double ldosr, ldosi;
  CmpVecDot(wtJconj,x,&ldosr,&ldosi);
  ldosr = -1.0*ldosr;
  
  double qfactor=xmagscalar/ldosr;
  PetscPrintf(PETSC_COMM_WORLD,"---qfactor for freq %.4e at step %d is: %.8e\n", omega/(2*PI),count,qfactor);

  /*------------------------------------------------*/
  /*------------------------------------------------*/

  /*-------------- Now store the epsilon at each step--------------*/

  char buffer [100];

  int STORE=1;    
  if(STORE==1 && (count%outputbase==0))
    {

      sprintf(buffer,"%.5depsSReal.m",count);
      OutputVec(PETSC_COMM_WORLD, epsSReal, filenameComm, buffer);
      
      FILE *tmpfile;
      int i;
      tmpfile = fopen(strcat(buffer,"DOF.txt"),"w");
      for (i=0;i<DegFree;i++){
      fprintf(tmpfile,"%0.16e \n",epsopt[i]);}
      fclose(tmpfile);

    }
  
  /*------------------------------------------------*/

  /*-------take care of the gradient---------*/
  if (grad) {
    VecCopy(x,Uone);
    VecPointwiseMult(Uone,Uone,VecVol);
    KSPSolveTranspose(ksp,Uone,u1);
    MatMult(C,u1,Grad1);
    CmpVecProd(Grad1,epscoef,tmp);
    CmpVecProd(tmp,x,Grad1);
    VecPointwiseMult(Grad1,Grad1,vR);
    VecScale(Grad1,2.0);
    VecScale(Grad1,1/ldosr);

    KSPSolveTranspose(ksp,wtJ,tmp);
    MatMult(C,tmp,Grad2);
    CmpVecProd(Grad2,epscoef,tmp);
    CmpVecProd(tmp,x,Grad2);
    VecPointwiseMult(Grad2,Grad2,vR);
    VecScale(Grad2,-1.0);
    VecScale(Grad2,-1.0*xmagscalar/pow(ldosr,2));

    VecSet(tmp,0.0);
    VecAXPY(tmp,1.0,Grad1);
    VecAXPY(tmp,1.0,Grad2);

    MatMultTranspose(A,tmp,vgrad);
    
    //correction from filters
    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);
    KSPSolveTranspose(kspH,vgrad,epsgrad);
    
    // copy vgrad (distributed vector) to a regular array grad;
    ierr = VecToArray(epsgrad,grad,scatter,from,to,vgradlocal,DegFree);
  
  }

  count++;

  MatDestroy(&Mtmp);

  VecDestroy(&wtJ);
  VecDestroy(&wtJconj);
  VecDestroy(&xconj);
  VecDestroy(&xmag);
  VecDestroy(&Uone);
  VecDestroy(&u1);
  VecDestroy(&Grad1);
  VecDestroy(&Grad2);
  VecDestroy(&tmp);

  VecDestroy(&vgrad);
  VecDestroy(&epsgrad);

  return qfactor;
}


#undef __FUNCT__ 
#define __FUNCT__ "makeBlock"
PetscErrorCode makeBlock(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, int lx, int ux, int ly, int uy, int lz, int uz)
{
  int i, j, ix, iy, iz, N;
  N=Nx*Ny*Nz;

  Vec b;
  PetscErrorCode ierr;
  ierr = VecCreate(comm,&b);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) b, "Source");CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,6*N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b); CHKERRQ(ierr);
  VecSet(b,0.0);

  int ns, ne;
  ierr = VecGetOwnershipRange(b, &ns, &ne); CHKERRQ(ierr);

  for(i=ns; i<ne; i++)
    {
      iz = (j = i) % Nz;
      iy = (j /= Nz) % Ny;
      ix = (j /= Ny) % Nx;

      if((ix>=lx) && (ix<ux) && (iy>=ly) && (iy<uy) && (iz>=lz) && (iz<uz)) 
	VecSetValue(b, i, 1.0, INSERT_VALUES);
    }

  VecAssemblyBegin(b);
  VecAssemblyEnd(b); 
  
  *bout = b;
  PetscFunctionReturn(0);
  
}
