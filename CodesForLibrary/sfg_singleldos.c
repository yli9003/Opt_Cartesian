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
#define __FUNCT__ "sfg_singleldos"
double sfg_singleldos(int DegFree,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;

  SFGdataGroup *ptdata = (SFGdataGroup *) data;

  int Nx = ptdata->Nx;
  int Ny = ptdata->Ny;
  int Nz = ptdata->Nz;
  double hxyz = ptdata->hxyz;
  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  double omega1 = ptdata->omega1;
  double omega2 = ptdata->omega2;
  double omega3 = ptdata->omega3;
  Mat M1 = ptdata->M1;
  Mat M2 = ptdata->M2;
  Mat M3 = ptdata->M3;
  Mat A = ptdata->A;
  Vec b1 = ptdata->b1;
  Vec b2 = ptdata->b2;
  Vec x1 = ptdata->x1;
  Vec x2 = ptdata->x2;
  Vec weightedJ1 = ptdata->weightedJ1;
  Vec weightedJ2 = ptdata->weightedJ2;
  Vec epspmlQ1  = ptdata->epspmlQ1;
  Vec epspmlQ2  = ptdata->epspmlQ2;
  Vec epspmlQ3  = ptdata->epspmlQ3;
  Vec epsmedium1 = ptdata->epsmedium1;
  Vec epsmedium2 = ptdata->epsmedium2;
  Vec epsmedium3 = ptdata->epsmedium3;
  Vec epsDiff1 = ptdata->epsDiff1;
  Vec epsDiff2 = ptdata->epsDiff2;
  Vec epsDiff3 = ptdata->epsDiff3;
  Vec epscoef1 = ptdata->epscoef1;  
  Vec epscoef2 = ptdata->epscoef2;  
  Vec epscoef3 = ptdata->epscoef3;  
  Mat B1 = ptdata->B1;
  Mat B2 = ptdata->B2;
  KSP ksp1 = ptdata->ksp1;
  KSP ksp2 = ptdata->ksp2;
  KSP ksp3 = ptdata->ksp3;
  int *its1 = ptdata->its1; 
  int *its2 = ptdata->its2; 
  int *its3 = ptdata->its3; 
  double p1 = ptdata->p1;
  double p2 = ptdata->p2;
  int outputbase = ptdata->outputbase;

  PetscPrintf(PETSC_COMM_WORLD,"----Sum Frequency Generation. Minapporoach NOT Available. ------- \n");
  
  Vec x3,J3,wtJ3,wtJ3c,b3;
  Vec Bx1,Bx2,Bx1c,Bx2c,tmp,tmp1,tmp2;
  Vec u1,u2,u3,u4,u5,Uone,Utwo,Ufour,Ufive;
  Vec Grad0,Grad1,Grad2,Grad3,Grad4,Grad5,Grad6;
  Vec vgrad,epsgrad,betagrad,ldos1grad,ldos2grad;

  VecDuplicate(vR,&x3);
  VecDuplicate(vR,&J3);
  VecDuplicate(vR,&wtJ3);
  VecDuplicate(vR,&wtJ3c);
  VecDuplicate(vR,&b3);
  VecDuplicate(vR,&Bx1);
  VecDuplicate(vR,&Bx2);
  VecDuplicate(vR,&Bx1c);
  VecDuplicate(vR,&Bx2c);
  VecDuplicate(vR,&tmp);
  VecDuplicate(vR,&tmp1);
  VecDuplicate(vR,&tmp2);
  VecDuplicate(vR,&u1);
  VecDuplicate(vR,&u2);
  VecDuplicate(vR,&u3);
  VecDuplicate(vR,&u4);
  VecDuplicate(vR,&u5);
  VecDuplicate(vR,&Uone);
  VecDuplicate(vR,&Utwo);
  VecDuplicate(vR,&Ufour);
  VecDuplicate(vR,&Ufive);
  VecDuplicate(vR,&Grad0);
  VecDuplicate(vR,&Grad1);
  VecDuplicate(vR,&Grad2);
  VecDuplicate(vR,&Grad3);
  VecDuplicate(vR,&Grad4);
  VecDuplicate(vR,&Grad5);
  VecDuplicate(vR,&Grad6);

  ierr=VecDuplicate(epsSReal,&vgrad); CHKERRQ(ierr);
  ierr=VecDuplicate(epsSReal,&epsgrad); CHKERRQ(ierr);
  ierr=VecDuplicate(epsSReal,&betagrad); CHKERRQ(ierr);
  ierr=VecDuplicate(epsSReal,&ldos1grad); CHKERRQ(ierr);
  ierr=VecDuplicate(epsSReal,&ldos2grad); CHKERRQ(ierr);
  

  RegzProj(DegFree,epsopt,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);
  
  MatMult(A,epsSReal,epsFReal);

  // Update the diagonals of M1, M2 and M3 Matrices;
  Mat Mone;
  MatDuplicate(M1,MAT_COPY_VALUES,&Mone);
  ModifyMatDiag(Mone, D, epsFReal, epsDiff1, epsmedium1, epspmlQ1, omega1, Nx, Ny, Nz);

  // solve the two fundamental modes and their ldos
  SolveMatrix(PETSC_COMM_WORLD,ksp1,Mone,b1,x1,its1);
  double ldos1,ldos2,tmpldosr,tmpldosi;
  CmpVecDot(weightedJ1,x1,&tmpldosr,&tmpldosi);
  ldos1=-hxyz*tmpldosr;

  PetscPrintf(PETSC_COMM_WORLD,"---*****The current ldos1 for omega %.4e at step %.5d is %.16e \n", omega1/(2*PI),count,ldos1);

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
/*------------------------------------------------*/

 /*-------take care of the gradient---------*/
  if (grad) {
    //ldos1grad calculation
    CmpVecProd(x1,x1,tmp);
    CmpVecProd(tmp,epscoef1,tmp1);
    MatMult(D,tmp1,tmp);
    VecScale(tmp,hxyz/omega1);
    VecPointwiseMult(tmp,tmp,weight);
    VecPointwiseMult(tmp,tmp,vR);
    MatMultTranspose(A,tmp,ldos1grad);

    //combine grad for beta/(ldos1^p1*ldos2^p2)
    VecSet(vgrad,0.0);
    VecAXPY(vgrad,1.0,ldos1grad);
    
    //DEBUG
    PetscPrintf(PETSC_COMM_WORLD,"ldos grad calculated\n");

    //correction from filters
    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);
    KSPSolveTranspose(kspH,vgrad,epsgrad);
    
    // copy vgrad (distributed vector) to a regular array grad;
    ierr = VecToArray(epsgrad,grad,scatter,from,to,vgradlocal,DegFree);
  
  }  

  VecDestroy(&x3);
  VecDestroy(&J3);
  VecDestroy(&wtJ3);
  VecDestroy(&wtJ3c);
  VecDestroy(&b3);
  VecDestroy(&Bx1);
  VecDestroy(&Bx2);
  VecDestroy(&Bx1c);
  VecDestroy(&Bx2c);
  VecDestroy(&tmp);
  VecDestroy(&tmp1);
  VecDestroy(&tmp2);
  VecDestroy(&u1);
  VecDestroy(&u2);
  VecDestroy(&u3);
  VecDestroy(&u4);
  VecDestroy(&u5);
  VecDestroy(&Uone);
  VecDestroy(&Utwo);
  VecDestroy(&Ufour);
  VecDestroy(&Ufive);
  VecDestroy(&Grad0);
  VecDestroy(&Grad1);
  VecDestroy(&Grad2);
  VecDestroy(&Grad3);
  VecDestroy(&Grad4);
  VecDestroy(&Grad5);
  VecDestroy(&Grad6);

  VecDestroy(&vgrad);
  VecDestroy(&epsgrad);
  VecDestroy(&betagrad);
  VecDestroy(&ldos1grad);
  VecDestroy(&ldos2grad);

  MatDestroy(&Mone);

  count++;

  return ldos1;
}



