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
#define __FUNCT__ "metasurface"
double metasurface(int DegFree,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;

  MetaSurfGroup *ptdata = (MetaSurfGroup *) data;

  int Nx = ptdata->Nx;
  int Ny = ptdata->Ny;
  int Nz = ptdata->Nz;
  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  double omega = ptdata->omega;
  Mat M = ptdata->M;
  Mat A = ptdata->A;
  Vec b = ptdata->b;
  Vec x = ptdata->x;
  Vec epspmlQ  = ptdata->epspmlQ;
  Vec epsmedium = ptdata->epsmedium;
  Vec epsDiff = ptdata->epsDiff;
  Vec epscoef = ptdata->epscoef;  
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its; 
  KSP refksp = ptdata->refksp;
  int *refits = ptdata->refits;
  double metaphase = ptdata->metaphase;
  Vec VecPt = ptdata->VecPt;
  int outputbase = ptdata->outputbase;
  //char filenameComm[PETSC_MAX_PATH_LEN];
  //strcpy(filenameComm,ptdata->filenameComm);

  PetscPrintf(PETSC_COMM_WORLD,"----Calculating Metasurface Phase. ------- \n");
  
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

  //Vec refField = ptdata->refField;
  //Vec refFieldconj = ptdata->refFieldconj;
  //double refmag = ptdata->refmag;

  //********************************************
  //make own refField;
  Vec refField;
  Vec refFieldconj;
  double refmag;
  VecDuplicate(x,&refField);
  VecDuplicate(x,&refFieldconj);
  Vec tmpepsS, tmpepsF, tmpx, tmpxconj, tmpxmag;
  Mat tmpM;
  VecDuplicate(epsSReal,&tmpepsS);
  VecDuplicate(x,&tmpepsF);
  VecDuplicate(x,&tmpx);
  VecDuplicate(x,&tmpxconj);
  VecDuplicate(x,&tmpxmag);
  VecSet(tmpepsS,0);
  MatMult(A,tmpepsS,tmpepsF);
  MatDuplicate(M,MAT_COPY_VALUES,&tmpM);
  ModifyMatDiag(tmpM, D, tmpepsF, epsDiff, epsmedium, epspmlQ, omega, Nx, Ny, Nz);
  PetscPrintf(PETSC_COMM_WORLD,"++++++Note that this is refField calculation++++++\n");
  SolveMatrix(PETSC_COMM_WORLD,refksp,tmpM,b,tmpx,refits);
  PetscPrintf(PETSC_COMM_WORLD,"++++++Note that this is refField calculation DONE++++++\n");
  MatMult(C,tmpx,tmpxconj);
  CmpVecProd(tmpx,tmpxconj,tmpxmag);
  VecPointwiseMult(tmpxmag,tmpxmag,vR);
  VecPointwiseMult(tmpxmag,tmpxmag,VecPt);
  VecSqrtAbs(tmpxmag);
  VecSum(tmpxmag,&refmag);
  VecCopy(tmpx,refField);
  VecCopy(tmpxconj,refFieldconj);
  VecDestroy(&tmpepsS);
  VecDestroy(&tmpepsF);
  VecDestroy(&tmpx);
  VecDestroy(&tmpxconj);
  VecDestroy(&tmpxmag);
  MatDestroy(&tmpM);
  //***********************************************

  MatMult(D,vR,vI);

  PetscPrintf(PETSC_COMM_WORLD,"!+!+!+!+!+!Note that this is FILTER calculation. \n");
  RegzProj(DegFree,epsopt,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);
  PetscPrintf(PETSC_COMM_WORLD,"!+!+!+!+!+!Note that this is FILTER calculation DONE. \n");

  MatMult(A,epsSReal,epsFReal);

  // Update the diagonals of M1, M2 and M3 Matrices;
  Mat Mtmp;
  MatDuplicate(M,MAT_COPY_VALUES,&Mtmp);
  ModifyMatDiag(Mtmp, D, epsFReal, epsDiff, epsmedium, epspmlQ, omega, Nx, Ny, Nz);

  // solve the two fundamental modes and their ldos
  SolveMatrix(PETSC_COMM_WORLD,ksp,Mtmp,b,x,its);

  MatMult(C,x,xconj);
  CmpVecProd(x,xconj,xmag);
  VecPointwiseMult(xmag,xmag,vR);
  VecPointwiseMult(xmag,xmag,VecPt);
  VecSqrtAbs(xmag);
  double xmagscalar;
  VecSum(xmag,&xmagscalar);

  CmpVecProd(x,refFieldconj,uvstar);
  VecPointwiseMult(uvstar,uvstar,VecPt);
  VecPointwiseMult(uvstarR,uvstar,vR);
  VecPointwiseMult(uvstarI,uvstar,vI);

  double fieldr, fieldi;
  VecSum(uvstarR,&fieldr);
  VecSum(uvstarI,&fieldi);
  double complex fieldval=fieldr+I*fieldi;
  double complex metafield=cos(metaphase)+I*sin(metaphase);
  double complex superpose=fieldval+metafield;
  double fieldmag=creal(fieldval*conj(fieldval));
  double superposemag=creal(superpose*conj(superpose));
  double superposephase=superposemag - fieldmag - 1;
  double superposephasenormalized= superposephase/sqrt(fieldmag);

  PetscPrintf(PETSC_COMM_WORLD,"---*****step, superposemag, superposephase, superposephasenormalized at freq %g: %d, %.8e, %.8e, %.8e\n", count,omega/(2*PI),superposemag,superposephase,superposephasenormalized);
  
  double phase_error=180*acos(superposephasenormalized/2)/PI;
  PetscPrintf(PETSC_COMM_WORLD,"---phase error (degree): %.8e\n", phase_error);
  PetscPrintf(PETSC_COMM_WORLD,"---transmission coefficient: %.8e\n", pow(xmagscalar/refmag,2));

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
    VecCopy(uvstar,tmp);
    VecAXPY(tmp,cos(metaphase),vR);
    VecAXPY(tmp,sin(metaphase),vI);
    VecPointwiseMult(tmp,tmp,VecPt);
    CmpVecProd(tmp,refField,Uone);
    KSPSolveTranspose(ksp,Uone,u1);
    MatMult(C,u1,Grad1);
    CmpVecProd(Grad1,epscoef,tmp);
    CmpVecProd(tmp,x,Grad1);
    VecPointwiseMult(Grad1,Grad1,vR);
    VecScale(Grad1,2.0);

    CmpVecProd(uvstar,refField,Utwo);
    VecPointwiseMult(Utwo,Utwo,VecPt);
    KSPSolveTranspose(ksp,Utwo,u2);
    MatMult(C,u2,Grad2);
    CmpVecProd(Grad2,epscoef,tmp);
    CmpVecProd(tmp,x,Grad2);
    VecPointwiseMult(Grad2,Grad2,vR);
    VecScale(Grad2,2.0);

    VecSet(tmp,0.0);
    VecAXPY(tmp,1.0,Grad1);
    VecAXPY(tmp,-1.0,Grad2);
    VecScale(tmp,1.0/sqrt(fieldmag));
    VecAXPY(tmp,-superposephase/(2*sqrt(fieldmag*fieldmag*fieldmag)),Grad2);
    //VecAXPY(tmp,-1.0/(2*sqrt(fieldmag*fieldmag*fieldmag)),Grad2);

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

  VecDestroy(&refField);
  VecDestroy(&refFieldconj);

  return superposephasenormalized;
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
    gradAll[i]=-1.0*grad[i];
  }
  gradAll[DegFreeAll-1]=1.0;

  return epsoptAll[DegFreeAll-1] - obj;
  
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
#undef __FUNCT__ 
#define __FUNCT__ "transmissionmeta"
double transmissionmeta(int DegFree,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;

  MetaSurfGroup *ptdata = (MetaSurfGroup *) data;

  int Nx = ptdata->Nx;
  int Ny = ptdata->Ny;
  int Nz = ptdata->Nz;
  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  double omega = ptdata->omega;
  Mat M = ptdata->M;
  Mat A = ptdata->A;
  Vec b = ptdata->b;
  Vec x = ptdata->x;
  Vec epspmlQ  = ptdata->epspmlQ;
  Vec epsmedium = ptdata->epsmedium;
  Vec epsDiff = ptdata->epsDiff;
  Vec epscoef = ptdata->epscoef;  
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its; 
  KSP refksp = ptdata->refksp;
  int *refits = ptdata->refits;
  double metaphase = ptdata->metaphase;
  Vec VecPt = ptdata->VecPt;
  int outputbase = ptdata->outputbase;
  //char filenameComm[PETSC_MAX_PATH_LEN];
  //strcpy(filenameComm,ptdata->filenameComm);

  PetscPrintf(PETSC_COMM_WORLD,"----Calculating Metasurface Transmission for constraint. ------- \n");
  
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

  //Vec refField = ptdata->refField;
  //Vec refFieldconj = ptdata->refFieldconj;
  //double refmag = ptdata->refmag;

  //********************************************
  //make own refField;
  Vec refField;
  Vec refFieldconj;
  double refmag;
  VecDuplicate(x,&refField);
  VecDuplicate(x,&refFieldconj);
  Vec tmpepsS, tmpepsF, tmpx, tmpxconj, tmpxmag;
  Mat tmpM;
  VecDuplicate(epsSReal,&tmpepsS);
  VecDuplicate(x,&tmpepsF);
  VecDuplicate(x,&tmpx);
  VecDuplicate(x,&tmpxconj);
  VecDuplicate(x,&tmpxmag);
  VecSet(tmpepsS,0);
  MatMult(A,tmpepsS,tmpepsF);
  MatDuplicate(M,MAT_COPY_VALUES,&tmpM);
  ModifyMatDiag(tmpM, D, tmpepsF, epsDiff, epsmedium, epspmlQ, omega, Nx, Ny, Nz);
  PetscPrintf(PETSC_COMM_WORLD,"++++++Note that this is refField calculation++++++\n");
  SolveMatrix(PETSC_COMM_WORLD,refksp,tmpM,b,tmpx,refits);
  PetscPrintf(PETSC_COMM_WORLD,"++++++Note that this is refField calculation DONE++++++\n");
  MatMult(C,tmpx,tmpxconj);
  CmpVecProd(tmpx,tmpxconj,tmpxmag);
  VecPointwiseMult(tmpxmag,tmpxmag,vR);
  VecPointwiseMult(tmpxmag,tmpxmag,VecPt);
  VecSqrtAbs(tmpxmag);
  VecSum(tmpxmag,&refmag);
  VecCopy(tmpx,refField);
  VecCopy(tmpxconj,refFieldconj);
  VecDestroy(&tmpepsS);
  VecDestroy(&tmpepsF);
  VecDestroy(&tmpx);
  VecDestroy(&tmpxconj);
  VecDestroy(&tmpxmag);
  MatDestroy(&tmpM);
  //***********************************************

  MatMult(D,vR,vI);

  PetscPrintf(PETSC_COMM_WORLD,"!+!+!+!+!+!Note that this is FILTER calculation. \n");
  RegzProj(DegFree,epsopt,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);
  PetscPrintf(PETSC_COMM_WORLD,"!+!+!+!+!+!Note that this is FILTER calculation DONE. \n");

  MatMult(A,epsSReal,epsFReal);

  // Update the diagonals of M1, M2 and M3 Matrices;
  Mat Mtmp;
  MatDuplicate(M,MAT_COPY_VALUES,&Mtmp);
  ModifyMatDiag(Mtmp, D, epsFReal, epsDiff, epsmedium, epspmlQ, omega, Nx, Ny, Nz);

  // solve the two fundamental modes and their ldos
  SolveMatrix(PETSC_COMM_WORLD,ksp,Mtmp,b,x,its);

  MatMult(C,x,xconj);
  CmpVecProd(x,xconj,xmag);
  VecPointwiseMult(xmag,xmag,vR);
  VecPointwiseMult(xmag,xmag,VecPt);
  VecSqrtAbs(xmag);
  double xmagscalar;
  VecSum(xmag,&xmagscalar);

  double trans=pow(xmagscalar/refmag,2);
  PetscPrintf(PETSC_COMM_WORLD,"---transmission coefficient for constraint: %.8e\n", trans);

  /*------------------------------------------------*/
  /*------------------------------------------------*/

  /*-------take care of the gradient---------*/
  if (grad) {
    VecCopy(x,Uone);
    VecPointwiseMult(Uone,Uone,VecPt);
    KSPSolveTranspose(ksp,Uone,u1);
    MatMult(C,u1,Grad1);
    CmpVecProd(Grad1,epscoef,tmp);
    CmpVecProd(tmp,x,Grad1);
    VecPointwiseMult(Grad1,Grad1,vR);
    VecScale(Grad1,2.0/pow(refmag,2));
    
    VecSet(tmp,0.0);
    VecAXPY(tmp,1.0,Grad1);

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

  VecDestroy(&refField);
  VecDestroy(&refFieldconj);

  return trans;
}

#undef __FUNCT__
#define __FUNCT__ "transmissionmetaconstr"
double transmissionmetaconstr(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  //double mintrans = 0.1; //Hard coded; to be fixed later

  int DegFree=DegFreeAll-1;
  double *epsopt, *grad;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  grad = (double *) malloc(DegFree*sizeof(double));
  int i;
  for(i=0;i<DegFree;i++){
    epsopt[i]=epsoptAll[i];
  }
  double obj=transmissionmeta(DegFree,epsopt,grad,data);
  count=count-1;
  for(i=0;i<DegFree;i++){
    gradAll[i]=-1.0*grad[i];
  }
  gradAll[DegFreeAll-1]=0;

  return mintrans - obj;
  
}

#undef __FUNCT__
#define __FUNCT__ "transmissionminimax"
double transmissionminimax(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{

  int DegFree=DegFreeAll-1;
  double *epsopt, *grad;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  grad = (double *) malloc(DegFree*sizeof(double));
  int i;
  for(i=0;i<DegFree;i++){
    epsopt[i]=epsoptAll[i];
  }
  double obj=transmissionmeta(DegFree,epsopt,grad,data);
  count=count-1;
  for(i=0;i<DegFree;i++){
    gradAll[i]= grad[i];
  }
  gradAll[DegFreeAll-1]=-1;

  return obj - epsoptAll[DegFreeAll-1];
  
}
