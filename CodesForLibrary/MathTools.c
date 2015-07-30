#include <stdio.h>
#include <math.h>
#include <petsc.h>
#include "libOPT.h"

//define Global variables;
#define Ptime PetscTime

extern int maxit;
extern Mat A, B, C, D;
extern Vec vR;

#undef __FUNCT__ 
#define __FUNCT__ "CmpVecScale"
PetscErrorCode CmpVecScale(Vec vin, Vec vout, double a, double b)
{
  Vec vini;
  PetscErrorCode ierr;
  ierr=VecDuplicate(vin,&vini);CHKERRQ(ierr); 
  ierr=MatMult(D,vin,vini);CHKERRQ(ierr);
  VecAXPBYPCZ(vout,a,b,0.0, vin,vini); 
  ierr=VecDestroy(&vini);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "CmpVecProd"
PetscErrorCode CmpVecProd(Vec va, Vec vb, Vec vout)
{
  PetscErrorCode ierr;

  int N;
  ierr=VecGetSize(va, &N); CHKERRQ(ierr);

  Vec vai,vbi;
  ierr=VecDuplicate(va, &vai); CHKERRQ(ierr);
  ierr=VecDuplicate(va, &vbi); CHKERRQ(ierr);
  
  ierr=MatMult(D,va,vai);CHKERRQ(ierr);
  ierr=MatMult(D,vb,vbi);CHKERRQ(ierr);

  double *a, *b, *ai, *bi, *out;
  ierr=VecGetArray(va,&a);CHKERRQ(ierr);
  ierr=VecGetArray(vb,&b);CHKERRQ(ierr);
  ierr=VecGetArray(vai,&ai);CHKERRQ(ierr);
  ierr=VecGetArray(vbi,&bi);CHKERRQ(ierr);
  ierr=VecGetArray(vout,&out);CHKERRQ(ierr);

  int i, ns, ne, nlocal;
  ierr = VecGetOwnershipRange(vout, &ns, &ne);
  nlocal = ne-ns;

  for (i=0; i<nlocal; i++)
    {  
      if(i<(N/2-ns)) // N is the total length of Vec;
	out[i] = a[i]*b[i] - ai[i]*bi[i];
      else
	out[i] = ai[i]*b[i] + a[i]*bi[i];
    }
  
  ierr=VecRestoreArray(va,&a);CHKERRQ(ierr);
  ierr=VecRestoreArray(vb,&b);CHKERRQ(ierr);
  ierr=VecRestoreArray(vai,&ai);CHKERRQ(ierr);
  ierr=VecRestoreArray(vbi,&bi);CHKERRQ(ierr);
  ierr=VecRestoreArray(vout,&out);CHKERRQ(ierr);

  ierr=VecDestroy(&vai);CHKERRQ(ierr);
  ierr=VecDestroy(&vbi);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "CmpVecProdScaF"
PetscErrorCode CmpVecProdScaF(Vec v1, Vec v2, Vec v)
{
  Vec tmp;
  PetscErrorCode ierr;
  ierr=VecDuplicate(v1,&tmp);CHKERRQ(ierr);
  CmpVecProd(v1,v2,tmp);
  ierr=MatMult(B,tmp,v);CHKERRQ(ierr);
  ierr=VecDestroy(&tmp);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "CmpVecDot"
PetscErrorCode CmpVecDot(Vec v1, Vec v2,  double *preal, double *pimag)
{
  Vec tmpr,tmpi,tmp;
  PetscErrorCode ierr;
  ierr=VecDuplicate(v1,&tmpr);CHKERRQ(ierr);  
  ierr=VecDuplicate(v1,&tmpi);CHKERRQ(ierr);
  ierr=VecDuplicate(v1,&tmp);CHKERRQ(ierr);

  CmpVecProd(v1,v2,tmp);
  
  VecPointwiseMult(tmpr,tmp,vR);
  VecSum(tmpr,preal);
  
  MatMult(D,tmp,tmpi);
  VecPointwiseMult(tmpi,tmpi,vR);
  VecSum(tmpi,pimag);
  *pimag = -1.0*(*pimag);

  ierr=VecDestroy(&tmpr); CHKERRQ(ierr);
  ierr=VecDestroy(&tmpi); CHKERRQ(ierr);
  ierr=VecDestroy(&tmp); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode  ArrayToVec(double *pt, Vec V)
{
  PetscErrorCode ierr;
  int j, ns, ne;

  ierr = VecGetOwnershipRange(V,&ns,&ne);
   for(j=ns;j<ne;j++)
    { ierr=VecSetValue(V,j,pt[j],INSERT_VALUES); 
      CHKERRQ(ierr);
    }

  ierr = VecAssemblyBegin(V); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(V);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode VecToArray(Vec V, double *pt, VecScatter scatter, IS from, IS to, Vec Vlocal, int DegFree)
{
  PetscErrorCode ierr;

 // scatter V to Vlocal;
    ierr =VecScatterCreate(V,from,Vlocal,to,&scatter); CHKERRQ(ierr);
    VecScatterBegin(scatter,V,Vlocal,INSERT_VALUES,SCATTER_FORWARD);
   VecScatterEnd(scatter,V,Vlocal,INSERT_VALUES,SCATTER_FORWARD);
   ierr =VecScatterDestroy(&scatter); CHKERRQ(ierr);

   // copy from vgradlocal to grad;
   double *ptVlocal;
   ierr =VecGetArray(Vlocal,&ptVlocal);CHKERRQ(ierr);

   int i;
   for(i=0;i<DegFree;i++)
     pt[i] = ptVlocal[i];
   ierr =VecRestoreArray(Vlocal,&ptVlocal);CHKERRQ(ierr);   
  PetscFunctionReturn(0);
}

#undef _FUNCT_
#define _FUNCT_ "MatSetTwoDiagonals"
/* [M(1,1) + sign*epsC(1), M(1,2) - sign*epsC(2);
   M(2,1) + sign*epsC(2), M(2,2) + sign*epsC(1);] */
PetscErrorCode MatSetTwoDiagonals(Mat M, Vec epsC, Mat D, double sign)
{
  PetscErrorCode ierr;
  
  Vec epsCi;
  ierr=VecDuplicate(epsC, &epsCi); CHKERRQ(ierr);
  ierr=MatMult(D,epsC,epsCi); CHKERRQ(ierr);
  
  int N;
  ierr=VecGetSize(epsC,&N); CHKERRQ(ierr);
    
  int i, ns, ne;
  MatGetOwnershipRange(M, &ns, &ne); CHKERRQ(ierr);

  double *c, *ci;
  ierr = VecGetArray(epsC, &c); CHKERRQ(ierr);  
  ierr = VecGetArray(epsCi, &ci); CHKERRQ(ierr);

  double vr, vi;
  
  for (i = ns; i < ne; ++i) 
    {
      if(i<N/2)
	{ vr = c[i-ns];  // here vr is real;
	  vi = ci[i-ns]; // here vi is -imag; happen to be correct combination;
	}
      else
	{ vr = ci[i-ns]; // here vr is real;
	  vi = c[i-ns];  // here vr is imag;
	}
      
      //M = M + sign*epsC
      ierr = MatSetValue(M,i,i,sign*vr,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(M,i,(i+N/2)%N,sign*vi,ADD_VALUES);CHKERRQ(ierr);
    }
  
  ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  
  ierr = VecRestoreArray(epsC, &c); CHKERRQ(ierr);
  ierr = VecRestoreArray(epsCi, &ci); CHKERRQ(ierr);
  
 /* Destroy Vectors */
  ierr=VecDestroy(&epsCi); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef _FUNCT_
#define _FUNCT_ "SolveMatrix"
PetscErrorCode SolveMatrix(MPI_Comm comm, KSP ksp, Mat M, Vec b, Vec x, int *its)
{
  /*-----------------KSP Solving------------------*/   
  PetscErrorCode ierr;
  PetscLogDouble t1,t2,tpast;
  ierr = Ptime(&t1);CHKERRQ(ierr);

  if (*its>(maxit-5)){
    ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);}
  else{
    ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);}

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,its);CHKERRQ(ierr);

  // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
  if(*its>(maxit-2))
    {
      ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
      ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(ksp,its);CHKERRQ(ierr);
    }

  //Print kspsolving information
  double norm;
  Vec xdiff;
  ierr=VecDuplicate(x,&xdiff);CHKERRQ(ierr);
  ierr = MatMult(M,x,xdiff);CHKERRQ(ierr);
  ierr = VecAXPY(xdiff,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(xdiff,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"==> Matrix solution: norm of error %g, Kryolv Iterations %d----\n ",norm,*its);CHKERRQ(ierr);    

  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0) PetscPrintf(PETSC_COMM_SELF,"==> Matrix solution: the runing time is %f s \n",tpast);
  /*--------------Finish KSP Solving---------------*/ 

  VecDestroy(&xdiff);
  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "ModifyMatDiag"
PetscErrorCode ModifyMatDiag(Mat Mopr, Mat D, Vec epsF, Vec epsDiff, Vec epsMedium, Vec epspmlQ, double omega, int Nx, int Ny, int Nz)
{
  PetscErrorCode ierr;
  Vec epsC, epsCi;
  ierr=VecDuplicate(epsF,&epsC); CHKERRQ(ierr);
  ierr=VecDuplicate(epsF,&epsCi); CHKERRQ(ierr);

  VecPointwiseMult(epsC,epsF,epsDiff);
  VecAXPY(epsC,1.0,epsMedium);
  VecPointwiseMult(epsC,epsC,epspmlQ);

  MatMult(D,epsC,epsCi);

  /*---------Modify diagonals of Mopr (more than main diagonals)------*/

  int ns, ne;
  ierr = MatGetOwnershipRange(Mopr, &ns, &ne); CHKERRQ(ierr);

  double *c, *ci;
  ierr = VecGetArray(epsC, &c); CHKERRQ(ierr);
  ierr = VecGetArray(epsCi, &ci); CHKERRQ(ierr);

  int i;
  int Nxyz=Nz*Ny*Nx;
  double omegasqr=omega*omega;

  double vr, vi;

  for (i = ns; i < ne; ++i) 
    {
      if(i<3*Nxyz)
	{ vr = c[i-ns];
	  vi = -ci[i-ns];
	}
      else
	{ vr = ci[i-ns];
	  vi = c[i-ns];
	}

      //Mopr = Mopr - omega^2*eps
      ierr = MatSetValue(Mopr,i,i,-omegasqr*vr,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(Mopr,i,(i+3*Nxyz)%(6*Nxyz), pow(-1,i/(3*Nxyz))*omegasqr*vi,ADD_VALUES);CHKERRQ(ierr);
    }
  ierr = MatAssemblyBegin(Mopr, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Mopr, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = VecRestoreArray(epsC, &c); CHKERRQ(ierr);
  ierr = VecRestoreArray(epsCi, &ci); CHKERRQ(ierr);

  /*-------------*/

  VecDestroy(&epsC);
  VecDestroy(&epsCi);

  PetscFunctionReturn(0);
}
