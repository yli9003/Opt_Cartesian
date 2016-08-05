#include <stdio.h>
#include <math.h>
#include <petsc.h>
#include "libOPT.h"

#undef __FUNCT__ 
#define __FUNCT__ "c3vinterp"
PetscErrorCode c3vinterp(MPI_Comm comm, Mat *Aout, int Mx, int My, int Nx, int Ny)
{
  PetscErrorCode ierr;

  int nz = 1;
  int ns, ne;
  int ix, iy, iz;
  int ix0, iy0, iz0;
  int i, j, id;

  int Nz=1;
  int Mz=1;
  int Mxy =  Mx*My;
  int Nxy =  Nx*Ny;

  Mat A0;
  MatCreate(comm, &A0);
  MatSetType(A0,MATMPIAIJ);
  MatSetSizes(A0,PETSC_DECIDE, PETSC_DECIDE, Nxy, Mxy);
  MatMPIAIJSetPreallocation(A0, nz, PETSC_NULL, nz, PETSC_NULL);

  ierr = MatGetOwnershipRange(A0, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;

    if(iy<Ny/2) iy=Ny-iy-1;

    iz0 = iz;
    ix0 = ix;
    iy0 = iy - Ny/2;
    
    if(ix0 < Mx && iy0 >= 0 && iy0 < ix0/sqrt(3) + Mx/sqrt(3) - 0.5 && iy0 < -sqrt(3)*ix0 + sqrt(3)*(Mx-0.5)) {
      id = iz0 + Mz*iy0 + Mz*My*ix0;
      ierr = MatSetValue(A0, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
    }

  }

  ierr = MatAssemblyBegin(A0, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A0, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) A0,"C3vStage0"); CHKERRQ(ierr);

  double x1, x2, x3, y1, y2, y3;
  Mat Arot1;
  MatCreate(comm,&Arot1);
  MatSetType(Arot1,MATMPIAIJ);
  MatSetSizes(Arot1,PETSC_DECIDE, PETSC_DECIDE, Nxy, Nxy);
  MatMPIAIJSetPreallocation(Arot1, 4, PETSC_NULL, 4, PETSC_NULL);

  ierr = MatGetOwnershipRange(Arot1, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;

    x1=ix-Nx/2 + 1/2;
    y1=iy-Ny/2 + 1/2;
    x2=       -x1/2  -sqrt(3)*y1/2;
    y2=sqrt(3)*x1/2          -y1/2;
    x3=x2+Nx/2 - 1/2;
    y3=y2+Ny/2 - 1/2;
    
    ix0 = round(x3);
    iy0 = round(y3);

    if(ix0>=0 && ix0<Nx && iy0>=0 && iy0<Ny){ 
      id = iz0 + Nz*iy0 + Nz*Ny*ix0;
      ierr = MatSetValue(Arot1, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  ierr = MatAssemblyBegin(Arot1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Arot1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Arot1,"C3vStageRotation1"); CHKERRQ(ierr);

  Mat Arot2;
  MatCreate(comm,&Arot2);
  MatSetType(Arot2,MATMPIAIJ);
  MatSetSizes(Arot2,PETSC_DECIDE, PETSC_DECIDE, Nxy, Nxy);
  MatMPIAIJSetPreallocation(Arot2, 4, PETSC_NULL, 4, PETSC_NULL);

  ierr = MatGetOwnershipRange(Arot2, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;

    x1=ix-Nx/2 + 1/2;
    y1=iy-Ny/2 + 1/2;
    x2=       -x1/2   +sqrt(3)*y1/2;
    y2= -sqrt(3)*x1/2         -y1/2;
    x3=x2+Nx/2 - 1/2;
    y3=y2+Ny/2 - 1/2;
    
    ix0 = round(x3);
    iy0 = round(y3);

    if(ix0>=0 && ix0<Nx && iy0>=0 && iy0<Ny){ 
      id = iz0 + Nz*iy0 + Nz*Ny*ix0;
      ierr = MatSetValue(Arot2, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  ierr = MatAssemblyBegin(Arot2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Arot2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Arot2,"C3vStageRotation2"); CHKERRQ(ierr);

  Mat Atrans1;
  MatCreate(comm,&Atrans1);
  MatSetType(Atrans1,MATMPIAIJ);
  MatSetSizes(Atrans1,PETSC_DECIDE, PETSC_DECIDE, Nxy, Nxy);
  MatMPIAIJSetPreallocation(Atrans1, 4, PETSC_NULL, 4, PETSC_NULL);

  ierr = MatGetOwnershipRange(Atrans1, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;

    x1=ix + (Nx+1.5)/2;
    y1=iy + sqrt(3)*(Nx+1.5)/2;
    
    ix0 = round(x1);
    iy0 = round(y1);

    if(ix0>=0 && ix0<Nx && iy0>=0 && iy0<Ny){ 
      id = iz0 + Nz*iy0 + Nz*Ny*ix0;
      ierr = MatSetValue(Atrans1, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  ierr = MatAssemblyBegin(Atrans1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Atrans1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Atrans1,"C3vStageTrans1"); CHKERRQ(ierr);

  Mat Atrans2;
  MatCreate(comm,&Atrans2);
  MatSetType(Atrans2,MATMPIAIJ);
  MatSetSizes(Atrans2,PETSC_DECIDE, PETSC_DECIDE, Nxy, Nxy);
  MatMPIAIJSetPreallocation(Atrans2, 4, PETSC_NULL, 4, PETSC_NULL);

  ierr = MatGetOwnershipRange(Atrans2, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;

    x1=ix + (Nx+1.5)/2;
    y1=iy - sqrt(3)*(Nx+1.5)/2;
    
    ix0 = round(x1);
    iy0 = round(y1);

    if(ix0>=0 && ix0<Nx && iy0>=0 && iy0<Ny){ 
      id = iz0 + Nz*iy0 + Nz*Ny*ix0;
      ierr = MatSetValue(Atrans2, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  ierr = MatAssemblyBegin(Atrans2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Atrans2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Atrans2,"C3vStageTrans2"); CHKERRQ(ierr);

  Mat Atrans3;
  MatCreate(comm,&Atrans3);
  MatSetType(Atrans3,MATMPIAIJ);
  MatSetSizes(Atrans3,PETSC_DECIDE, PETSC_DECIDE, Nxy, Nxy);
  MatMPIAIJSetPreallocation(Atrans3, 4, PETSC_NULL, 4, PETSC_NULL);

  ierr = MatGetOwnershipRange(Atrans3, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;

    x1=ix - (Nx+1.5)/2;
    y1=iy + sqrt(3)*(Nx+1.5)/2;
    
    ix0 = round(x1);
    iy0 = round(y1);

    if(ix0>=0 && ix0<Nx && iy0>=0 && iy0<Ny){ 
      id = iz0 + Nz*iy0 + Nz*Ny*ix0;
      ierr = MatSetValue(Atrans3, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  ierr = MatAssemblyBegin(Atrans3, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Atrans3, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Atrans3,"C3vStageTrans2"); CHKERRQ(ierr);

  Mat Atrans4;
  MatCreate(comm,&Atrans4);
  MatSetType(Atrans4,MATMPIAIJ);
  MatSetSizes(Atrans4,PETSC_DECIDE, PETSC_DECIDE, Nxy, Nxy);
  MatMPIAIJSetPreallocation(Atrans4, 4, PETSC_NULL, 4, PETSC_NULL);

  ierr = MatGetOwnershipRange(Atrans4, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;

    x1=ix - (Nx+1.5)/2;
    y1=iy - sqrt(3)*(Nx+1.5)/2;
    
    ix0 = round(x1);
    iy0 = round(y1);

    if(ix0>=0 && ix0<Nx && iy0>=0 && iy0<Ny){ 
      id = iz0 + Nz*iy0 + Nz*Ny*ix0;
      ierr = MatSetValue(Atrans4, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  ierr = MatAssemblyBegin(Atrans4, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Atrans4, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Atrans4,"C3vStageTrans2"); CHKERRQ(ierr);

  Mat tmp,A;
  Mat tmp1,tmp2,tmp3,tmp4;
  MatMatMult(Arot1,A0,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A);
  MatAXPY(A,1.0,A0,DIFFERENT_NONZERO_PATTERN);
  MatMatMult(Arot2,A0,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tmp);
  MatAXPY(A,1.0,tmp,DIFFERENT_NONZERO_PATTERN);
  MatMatMult(Atrans1,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tmp1);
  MatMatMult(Atrans2,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tmp2);
  MatMatMult(Atrans3,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tmp3);
  MatMatMult(Atrans4,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tmp4);
  MatAXPY(A,1.0,tmp1,DIFFERENT_NONZERO_PATTERN);
  MatAXPY(A,1.0,tmp2,DIFFERENT_NONZERO_PATTERN);
  MatAXPY(A,1.0,tmp3,DIFFERENT_NONZERO_PATTERN);
  MatAXPY(A,1.0,tmp4,DIFFERENT_NONZERO_PATTERN);
  *Aout = A;

  MatDestroy(&A0);
  MatDestroy(&Arot1);
  MatDestroy(&Arot2);
  MatDestroy(&Atrans1);
  MatDestroy(&Atrans2);
  MatDestroy(&Atrans3);
  MatDestroy(&Atrans4);
  MatDestroy(&tmp);
  MatDestroy(&tmp1);
  MatDestroy(&tmp2);
  MatDestroy(&tmp3);
  MatDestroy(&tmp4);

  PetscFunctionReturn(0);
}
