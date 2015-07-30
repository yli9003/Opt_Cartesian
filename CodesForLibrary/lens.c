#include <petsc.h>
#include <time.h>
#include "libOPT.h"
#include <complex.h>
#include "petsctime.h"

#define Ptime PetscTime

extern int count;
extern Mat A,C,D;
extern Vec vR,  weight, vgradlocal;
extern VecScatter scatter;
extern IS from, to;
extern char filenameComm[PETSC_MAX_PATH_LEN];

extern int pSIMP;
extern double bproj, etaproj;
extern Mat Hfilt;
extern KSP kspH;
extern int itsH;

#undef __FUNCT__ 
#define __FUNCT__ "optfocalpt"
double optfocalpt(int DegFree, double *epsopt, double *grad, void *data)
{

  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the focal point solver.********** \n");

  LensGroup *ptdata = (LensGroup *) data;

  int Nx = ptdata->Nx;
  int Ny = ptdata->Ny;
  int Nz = ptdata->Nz;
  double hxyz = ptdata->hxyz;
  double omega = ptdata->omega;
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its;
  Mat M = ptdata->M;
  Vec b = ptdata->b;
  Vec x = ptdata->x;
  Vec VecFocalpt = ptdata->VecFocalpt;
  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  Vec epsDiff = ptdata->epsDiff;
  Vec epsMed = ptdata->epsMed;
  Vec epspmlQ = ptdata->epspmlQ;
  Vec epscoef = ptdata->epscoef;
  Vec Vecgrad = ptdata->Vecgrad;
  int outputbase = ptdata->outputbase;

  Vec xconj, tmp, u;
  VecDuplicate(vR,&xconj);
  VecDuplicate(vR,&tmp);
  VecDuplicate(vR,&u);

  Vec epsgrad;
  VecDuplicate(epsSReal,&epsgrad);
  RegzProj(DegFree,epsopt,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);

  MatMult(A,epsSReal,epsFReal);  
  // Update the diagonals of M;
  Mat Mopr;
  MatDuplicate(M,MAT_COPY_VALUES,&Mopr);
  ModifyMatDiag(Mopr, D, epsFReal, epsDiff, epsMed, epspmlQ, omega, Nx, Ny, Nz);
  
  SolveMatrix(PETSC_COMM_WORLD,ksp,Mopr,b,x,its);
  MatMult(C,x,xconj);
  CmpVecProd(x,xconj,tmp);
  VecPointwiseMult(tmp,tmp,VecFocalpt);
  VecPointwiseMult(tmp,tmp,weight);
  double focalpower;
  VecSum(tmp, &focalpower);
  focalpower = hxyz*focalpower;

  VecPointwiseMult(tmp,VecFocalpt,xconj);
  KSPSolve(ksp,tmp,u);


  CmpVecProd(u,epscoef,tmp);
  CmpVecProd(tmp,x,u);
  VecPointwiseMult(u,u,weight);
  VecPointwiseMult(u,u,vR);
  VecScale(u,2.0);
  VecScale(u,hxyz);

  MatMultTranspose(A,u,Vecgrad);

  ierr=VecPointwiseMult(Vecgrad,Vecgrad,epsgrad); CHKERRQ(ierr);
  KSPSolveTranspose(kspH,Vecgrad,epsgrad);
  ierr = VecToArray(epsgrad,grad,scatter,from,to,vgradlocal,DegFree);

  if(count%outputbase==0)
    {
      char buffer[1000];
      FILE *epsFile,*dofFile;
      int i;
      double *tmpeps;
      tmpeps = (double *) malloc(DegFree*sizeof(double));
      ierr = VecToArray(epsSReal,tmpeps,scatter,from,to,vgradlocal,DegFree);

      sprintf(buffer,"%s_%.5deps.txt",filenameComm,count);
      epsFile = fopen(buffer,"w");
      for (i=0;i<DegFree;i++){
        fprintf(epsFile,"%0.16e \n",tmpeps[i]);}
      fclose(epsFile);

      sprintf(buffer,"%s_%.5ddof.txt",filenameComm,count);
      dofFile = fopen(buffer,"w");
      for (i=0;i<DegFree;i++){
        fprintf(dofFile,"%0.16e \n",epsopt[i]);}
      fclose(dofFile);

      sprintf(buffer,"%s_%.5dEfield",filenameComm,count);
      OutputVec(PETSC_COMM_WORLD, x,buffer,".m");

      Vec epsVec;
      VecDuplicate(vR,&epsVec);
      VecPointwiseMult(epsVec,epsFReal,epsDiff);
      VecAXPY(epsVec,1.0,epsMed);
      sprintf(buffer,"%s_%.5depsF",filenameComm,count);
      OutputVec(PETSC_COMM_WORLD,epsVec,buffer,".m");
      VecDestroy(&epsVec);


    }

  PetscPrintf(PETSC_COMM_WORLD,"**********-----current focal power at step %d is %g \n",count,focalpower);

  MatDestroy(&Mopr);
  VecDestroy(&epsgrad);

  VecDestroy(&xconj);
  VecDestroy(&tmp);
  VecDestroy(&u);
  count++;

  return focalpower;

}


PetscErrorCode MakeVecFocalpt(Vec VecFocalpt, int Nx, int Ny, int Nz, int ix, int iy, int iz, int ic1, int ic2)
{

  VecSet(VecFocalpt,0.0);
  int pos1jreal= 0*3*Nx*Ny*Nz + ic1*Nx*Ny*Nz + ix*Ny*Nz + iy*Nz + iz;
  int pos1jimag= 1*3*Nx*Ny*Nz + ic1*Nx*Ny*Nz + ix*Ny*Nz + iy*Nz + iz;
  int pos2jreal= 0*3*Nx*Ny*Nz + ic2*Nx*Ny*Nz + ix*Ny*Nz + iy*Nz + iz;
  int pos2jimag= 1*3*Nx*Ny*Nz + ic2*Nx*Ny*Nz + ix*Ny*Nz + iy*Nz + iz;
  VecSetValue(VecFocalpt,pos1jreal,1.0,ADD_VALUES);
  VecSetValue(VecFocalpt,pos1jimag,1.0,ADD_VALUES);
  VecSetValue(VecFocalpt,pos2jreal,1.0,ADD_VALUES);
  VecSetValue(VecFocalpt,pos2jimag,1.0,ADD_VALUES);

  return 0;

}
