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

#undef __FUNCT__ 
#define __FUNCT__ "metasurface"
double metasurface(int DegFree,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;

  MetaSurfGroup *ptdata = (MetaSurfGroup *) data;

  int Nx = ptdata->Nx;
  int Ny = ptdata->Ny;
  int Nz = ptdata->Nz;
  //double hxyz = ptdata->hxyz;
  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  double omega = ptdata->omega;
  Mat M = ptdata->M;
  Mat A = ptdata->A;
  Vec b = ptdata->b;
  //Vec J = ptdata->J;
  Vec x = ptdata->x;
  //Vec weightedJ = ptdata->weightedJ;
  Vec epspmlQ  = ptdata->epspmlQ;
  Vec epsmedium = ptdata->epsmedium;
  Vec epsDiff = ptdata->epsDiff;
  Vec epscoef = ptdata->epscoef;  
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its; 
  double metaphase = ptdata->metaphase;
  int trigoption = ptdata->trigoption;
  Vec refField = ptdata->refField;
  Vec refFieldconj = ptdata->refFieldconj;
  Vec VecPt = ptdata->VecPt;
  int outputbase = ptdata->outputbase;
  char filenameComm[PETSC_MAX_PATH_LEN];
  strcpy(filenameComm,ptdata->filenameComm);

  PetscPrintf(PETSC_COMM_WORLD,"----Calculating Metasurface Phase. Make sure that refField has magnitude 1 at all points.------- \n");
  
  Vec xconj, xmag, uvstar, uvstarR, uvstarI, vI, tmp, Uone, u1, Utwo, u2, Grad1, Grad2, vgrad, epsgrad;
  VecDuplicate(x,&xconj);
  VecDuplicate(x,&xmag);
  VecDuplicate(x,&uvstar);
  VecDuplicate(x,&uvstarR);
  VecDuplicate(x,&uvstarI);
  VecDuplicate(x,&vI);
  VecDuplicate(x,&tmp);
  VecDuplicate(x,&Uone);
  VecDuplicate(x,&u1);
  VecDuplicate(x,&Utwo);
  VecDuplicate(x,&u2);
  VecDuplicate(x,&Grad1);
  VecDuplicate(x,&Grad2);

  VecDuplicate(epsSReal,&vgrad);
  VecDuplicate(epsSReal,&epsgrad);
  
  MatMult(D,vR,vI);

  RegzProj(DegFree,epsopt,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);
  
  MatMult(A,epsSReal,epsFReal);

  // Update the diagonals of M1, M2 and M3 Matrices;
  Mat Mtmp;
  MatDuplicate(M,MAT_COPY_VALUES,&Mtmp);
  ModifyMatDiag(Mtmp, D, epsFReal, epsDiff, epsmedium, epspmlQ, omega, Nx, Ny, Nz);

  // solve the two fundamental modes and their ldos
  SolveMatrix(PETSC_COMM_WORLD,ksp,Mtmp,b,x,its);

  CmpVecProd(x,refFieldconj,uvstar);
  VecPointwiseMult(uvstar,uvstar,VecPt);
  VecPointwiseMult(uvstarR,uvstar,vR);
  VecPointwiseMult(uvstarI,uvstar,vI);

  double xamp;
  MatMult(C,x,xconj);
  CmpVecProd(x,xconj,xmag);
  VecPointwiseMult(xmag,xmag,vR);
  VecPointwiseMult(xmag,xmag,VecPt);
  VecSum(xmag,&xamp);
  xamp=sqrt(xamp);
  if(fabs(xamp)<1e-3) PetscPrintf(PETSC_COMM_WORLD,"***WARNING zero amp at ref point.\n");
  
  double quadr, quadi, quadrature, refquad, quaddiff, quaddiffsq;
  VecSum(uvstarR,&quadr);
  VecSum(uvstarI,&quadi);
  quadrature=((trigoption==1)*quadr + (trigoption==2)*quadi)/xamp;
  refquad=(trigoption==1)*cos(metaphase)+(trigoption==2)*sin(metaphase);
  quaddiff=quadrature-refquad;
  quaddiffsq=quaddiff*quaddiff;

  PetscPrintf(PETSC_COMM_WORLD,"---*****step, quadrature choice, value: %d, %d, %.8e, %.8e, %.8e\n", count,trigoption,quadrature,quaddiff,quaddiffsq);


  /*-------------- Now store the epsilon at each step--------------*/

  char buffer [100];

  int STORE=1;    
  if(STORE==1 && (count%outputbase==0))
    {
      sprintf(buffer,"%.5depsSReal.m",count);
      OutputVec(PETSC_COMM_WORLD, epsSReal, "optstep", buffer);
      
      FILE *tmpfile;
      int i;
      tmpfile = fopen(strcat(buffer,"DOF.txt"),"w");
      for (i=0;i<DegFree;i++){
      fprintf(tmpfile,"%0.16e \n",epsopt[i]);}
      fclose(tmpfile);

    }

  /*------------------------------------------------*/
  /*------------------------------------------------*/

  /*-------take care of the gradient---------*/
  if (grad) {
    VecPointwiseMult(Uone,refField,VecPt);
    VecScale(Uone,1/xamp);
    KSPSolveTranspose(ksp,Uone,u1);
    MatMult(C,u1,Grad1);
    CmpVecProd(Grad1,epscoef,tmp);
    CmpVecProd(tmp,x,Grad1);


    VecPointwiseMult(Utwo,x,VecPt);
    KSPSolveTranspose(ksp,Utwo,u2);
    MatMult(C,u2,tmp); 
    CmpVecProd(tmp,x,Grad2);
    CmpVecProd(Grad2,epscoef,tmp); 
    VecPointwiseMult(tmp,tmp,vR);
    VecScale(tmp,1/xamp);

    //VecSet(tmp,0.0);
    //VecAXPY(tmp,1.0,Grad1);
    //VecAXPY(tmp,1.0,Grad2);
    
    //if(trigoption==1) VecPointwiseMult(tmp,tmp,vR);
    //if(trigoption==2) VecPointwiseMult(tmp,tmp,vI);
    //VecScale(tmp,2*quaddiff);
    MatMultTranspose(A,tmp,vgrad);
    
    //correction from filters
    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);
    KSPSolveTranspose(kspH,vgrad,epsgrad);
    
    // copy vgrad (distributed vector) to a regular array grad;
    ierr = VecToArray(epsgrad,grad,scatter,from,to,vgradlocal,DegFree);
  
  }

  count++;

  MatDestroy(&Mtmp);

  VecDestroy(&xconj);
  VecDestroy(&xmag);
  VecDestroy(&uvstar);
  VecDestroy(&uvstarR);
  VecDestroy(&uvstarI);
  VecDestroy(&vI);
  VecDestroy(&tmp);
  VecDestroy(&Uone);
  VecDestroy(&u1);
  VecDestroy(&Utwo);
  VecDestroy(&u2);
  VecDestroy(&Grad1);
  VecDestroy(&Grad2);

  VecDestroy(&vgrad);
  VecDestroy(&epsgrad);

  //return quaddiffsq;
  return xamp;
}

PetscErrorCode MakeVecPt(Vec VecPt, int Nx, int Ny, int Nz, int ix, int iy, int iz, int ic)
{

  VecSet(VecPt,0.0);
  int posjreal= 0*3*Nx*Ny*Nz + ic*Nx*Ny*Nz + ix*Ny*Nz + iy*Nz + iz;
  int posjimag= 1*3*Nx*Ny*Nz + ic*Nx*Ny*Nz + ix*Ny*Nz + iy*Nz + iz;
  VecSetValue(VecPt,posjreal,1.0,ADD_VALUES);
  VecSetValue(VecPt,posjimag,1.0,ADD_VALUES);
  VecAssemblyBegin(VecPt);
  VecAssemblyEnd(VecPt);

  return 0;

}

#undef __FUNCT__
#define __FUNCT__ "metasurfaceminimax"
double metasurfaceminimax(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  int DegFree=DegFreeAll-1;
  double *epsopt, *grad;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  grad = (double *) malloc(DegFree*sizeof(double));
  int i;
  for(i=0;i<DegFree;i++){
    epsopt[i]=epsoptAll[i];
  }
  double obj=metasurface(DegFree,epsopt,grad,data);
  count=count-1;
  for(i=0;i<DegFree;i++){
    gradAll[i]=grad[i];
  }
  gradAll[DegFreeAll-1]=-1.0;

  return obj-epsoptAll[DegFreeAll-1];
  
}

#undef __FUNCT__
#define __FUNCT__ "minimaxobjfun"
double minimaxobjfun(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{

  if(gradAll)
    {
      int i;
      for (i=0;i<DegFreeAll-1;i++)
	{
          gradAll[i]=0;
	}
      gradAll[DegFreeAll-1]=1;
    }

  PetscPrintf(PETSC_COMM_WORLD,"**the current value of dummy objective variable is %.8e**\n",epsoptAll[DegFreeAll-1]);

  count++;

  return epsoptAll[DegFreeAll-1];
}
