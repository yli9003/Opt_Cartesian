#include <stdio.h>
#include <math.h>
#include <petsc.h>
#include "libOPT.h"

/* return a sparse matrix A that expands the triangular region of DOFs into square region with C4v symmetry. 
*/


#undef __FUNCT__ 
#define __FUNCT__ "c4v"
PetscErrorCode c4v(MPI_Comm comm, Mat *Aout, int M)
{
  Mat A;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;
  int i, j;

  int Mxy =  M*M;
  int nDOF = M*(M+1)/2;

  int ix, iy;
  double ix0,iy0; 
  int id;

  MatCreate(comm, &A);
  MatSetType(A,MATMPIAIJ);
  MatSetSizes(A,PETSC_DECIDE, PETSC_DECIDE, Mxy, nDOF);
  MatMPIAIJSetPreallocation(A, nz, PETSC_NULL, nz, PETSC_NULL);

  ierr = MatGetOwnershipRange(A, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iy = (j = i) % M;
    ix = (j /= M) % M;

    if(iy >= ix){
      id=iy*(iy+1)/2 + ix;
    }else{
      ix0=iy;
      iy0=ix;
      id=iy0*(iy0+1)/2 + ix0;
    }

    ierr = MatSetValue(A, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) A,"C4vInterpMatrix"); CHKERRQ(ierr);
  *Aout = A;
  PetscFunctionReturn(0);
}
