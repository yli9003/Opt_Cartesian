#include <petsc.h>
#include <slepceps.h>
#include "libOPT.h"

/*---this subrountie compute the eigenvalues lambda of the generalized eigenvalue problem (M-eps*omega_0^2) V = eps*lambda * V. In order to compute with mpb, user need to convert like this sqrt(lambda+omega_0^2)/(2*pi). */

#undef _FUNCT_
#define _FUNCT_ "eigsolver"
int eigsolver(Mat M, Vec epsC, Mat D)
{

  PetscErrorCode ierr;
  EPS eps;
  PetscInt nconv;

  Mat B;
  int nrow, ncol;

  ierr=MatGetSize(M,&nrow, &ncol); CHKERRQ(ierr);

  ierr=MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nrow, ncol, 2, NULL, 2, NULL, &B); CHKERRQ(ierr);
  ierr=PetscObjectSetName((PetscObject)B, "epsmatrix"); CHKERRQ(ierr);
  
  MatSetTwoDiagonals(B, epsC, D, 1.0);
   
  PetscPrintf(PETSC_COMM_WORLD,"!!!---computing eigenvalues---!!! \n");
  ierr=EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
  ierr=EPSSetOperators(eps, M, B); CHKERRQ(ierr);
  EPSSetFromOptions(eps);

  PetscLogDouble t1, t2, tpast;
  ierr = PetscTime(&t1);CHKERRQ(ierr);

  ierr=EPSSolve(eps); CHKERRQ(ierr);
  EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
  
  {
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    tpast = t2 - t1;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
      PetscPrintf(PETSC_COMM_SELF,"---The eigensolver time is %f s \n",tpast);
  }  

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of converged eigenpairs: %d\n\n",nconv);CHKERRQ(ierr);


  double *krarray, *kiarray, *errorarray;
  krarray = (double *) malloc(sizeof(double)*nconv);
  kiarray = (double *) malloc(sizeof(double)*nconv);
  errorarray =(double *) malloc(sizeof(double)*nconv);

  Vec xr, xi;
  ierr=MatGetVecs(M,PETSC_NULL,&xr); CHKERRQ(ierr);
  ierr=MatGetVecs(M,PETSC_NULL,&xi); CHKERRQ(ierr);
  ierr=PetscObjectSetName((PetscObject) xr, "xr"); CHKERRQ(ierr);
  ierr=PetscObjectSetName((PetscObject) xi, "xi"); CHKERRQ(ierr);
  int ni;
  for(ni=0; ni<nconv; ni++)
    {
      ierr=EPSGetEigenpair(eps, ni, krarray+ni, kiarray+ni, xr, xi);CHKERRQ(ierr);
      ierr = EPSComputeRelativeError(eps,ni,errorarray+ni);CHKERRQ(ierr);
      
      char bufferr[100], bufferi[100];
      sprintf(bufferr,"%.2dxr.m",ni+1);
      sprintf(bufferi,"%.2dxi.m",ni+1);

      OutputVec(PETSC_COMM_WORLD,xr,"eigenmode.",bufferr);
      OutputVec(PETSC_COMM_WORLD,xi,"eigenmode.",bufferi);
    }

  PetscPrintf(PETSC_COMM_WORLD, "Now print the eigenvalues: \n");
  for(ni=0; ni<nconv; ni++)
    PetscPrintf(PETSC_COMM_WORLD," %.12e%+.12ei,", krarray[ni], kiarray[ni]);

  PetscPrintf(PETSC_COMM_WORLD, "\n\nstart printing erros");

  for(ni=0; ni<nconv; ni++)
    PetscPrintf(PETSC_COMM_WORLD," %g,", errorarray[ni]);      

  PetscPrintf(PETSC_COMM_WORLD,"\n\n Finish EPS Solving !!! \n\n");

  /*-- destroy vectors and free space --*/
  MatDestroy(&B);
  EPSDestroy(&eps);
  VecDestroy(&xr);
  VecDestroy(&xi);

  free(krarray);
  free(kiarray);
  free(errorarray);

  PetscFunctionReturn(0);
}
