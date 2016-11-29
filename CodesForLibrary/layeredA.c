#include <stdio.h>
#include <math.h>
#include <petsc.h>
#include "libOPT.h"

/* return a sparse matrix A that performs nearest-neighbor interpolation
   from data d an (Mx,My,Mz) centered grid to a 3x(Nx,Ny,Nz) Yee (E) grid,
   such that A*d computes the interpolated data.  Values outside
   the d box are taken to be zero (if you want a nonzero value,
   just use A*(d - const) + const).

   The data d is taken to reside in the box from (x0,y0,z0) to
   (x1,y1,z1), where these coordinates lie in [0,1]^3, with (0,0,0)
   the index (0,0,0), and (1,1,1) indicating the index (Nx,Ny,Nz).

   All data is assumed to be stored in row-major order
   (Mx-by-My-by-Mz and 3-by-Nx-by-Ny-by-Nz for input and output, respectively).
   
   That is, the coordinate (i,j,k) in the d array refers to:
   1) the index (i*My + j)*Mz + k in d.
   2) the point (Nx * [x0 + (i+0.5) * (x1-x0)/Mx],
   Ny * [y0 + (j+0.5) * (y1-y0)/My],
   Nz * [z0 + (k+0.5) * (z1-z0)/Mz]) in Yee space.
   and the coordinate (c,i,j,k) in the Yee array, where 0 <= c < 3,
   corresponds to:
   1) the index ((c*Nx +  i)*Ny + j)*Nz + k in the output array
   2) the point (i,j,k) + 0.5 * e_c in Yee space,
   where e_c is the unit vector in the c direction.

   The input and output vectors are distributed in the default manner
   for the given communicator comm, and hence A is distributed.
*/

/* 
      epsair
===== eps[2]    ____ Nzo[2]
      epsmid
===== eps[1]    ____ Nzo[1]
      epsmid
===== eps[0]    ____ Nzo[0]
      epssub
*/

#undef __FUNCT__ 
#define __FUNCT__ "layeredA"
PetscErrorCode layeredA(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, int nlayers, int Nxo, int Nyo, int* Nzo, int Mx, int My, int* Mz, int Mzslab)
{
  Mat A;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;
  double shift =  0.5;
  int i;
  int Nc = 3; //modified;

  int DegFree=0;
  for (i=0;i<nlayers;i++) {
    DegFree=DegFree+Mx*My*((Mzslab==0)?Mz[i]:1);
  }

  MatCreate(comm, &A);
  MatSetType(A,MATMPIAIJ);
  MatSetSizes(A,PETSC_DECIDE, PETSC_DECIDE, 6*Nx*Ny*Nz, DegFree);
  MatMPIAIJSetPreallocation(A, nz, PETSC_NULL, nz, PETSC_NULL);

  ierr = MatGetOwnershipRange(A, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    int ix, iy, iz, ic;
    double xd,yd,zd; /* (ix,iy,iz) location in d coordinates */
    int ixd,iyd,izd; /* rounded (xd,yd,zd) */
    int j, id;
    int ik,iik;

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;
    ic = (j /= Nx) % Nc; // modified, Nc = 3;

    xd = (ix-Nxo) + (ic!= 0)*shift;
    ixd = ceil(xd-0.5);
    if (ixd < 0 || ixd >= Mx) continue;
   
    yd = (iy-Nyo) + (ic!= 1)*shift;
    iyd = ceil(yd - 0.5);
    if (iyd < 0 || iyd >= My) continue;

    for (ik=0;ik<nlayers;ik++) {
      zd = (iz-Nzo[ik]) + (ic!= 2)*shift;
      izd = ceil(zd - 0.5);
      if (izd < 0 || izd >= Mz[ik] ) continue;
      
      if(Mzslab!=0) {
	id = (ixd*My + iyd) + ik*Mx*My;
      }else{
	id = (ixd*My + iyd)*Mz[ik] + izd;
	for (iik=0;iik<ik;iik++) {
	  id = id + Mx*My*Mz[iik];
	}
      }

      ierr = MatSetValue(A, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) A,"InterpMatrix"); CHKERRQ(ierr);
  *Aout = A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "layeredepsbkg"
PetscErrorCode layeredepsbkg(Vec epsBkg, int Nx, int Ny, int Nz, int nlayers, int* Nzo, int* Mz, double* epsbkg, double epssub, double epsair, double epsmid)
{
  int i, j, ns, ne;
  int iz, iy, ix, ic, Nc=3;
  double shift=0.5;
  double zd;
  int izd;
  int ik;
  double value;
  PetscErrorCode ierr;

  ierr = VecGetOwnershipRange(epsBkg, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;
    ic = (j /= Nx) % Nc; // modified, Nc = 3;

    for (ik=0;ik<nlayers;ik++) {

      zd = (iz-Nzo[ik]) + (ic!= 2)*shift;
      izd = ceil(zd - 0.5);

      if(izd >= 0 && izd < Mz[ik]) {
	value=epsbkg[ik];
	break;
      }

      if (izd < Nzo[0] - Nzo[ik]) {
	value=epssub;
      }else if(izd >= Nzo[nlayers-1] - Nzo[ik] + Mz[nlayers-1]) {
	value=epsair;
      }else{ 
	value=epsmid;
      }
      
    }

    ierr = VecSetValue(epsBkg, i, value, INSERT_VALUES); CHKERRQ(ierr);
  
  }

  ierr = VecAssemblyBegin(epsBkg); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsBkg); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "layeredepsdiff"
PetscErrorCode layeredepsdiff(Vec epsDiff, int Nx, int Ny, int Nz, int nlayers, int* Nzo, int* Mz, double* epsdiff, double epssubdiff, double epsairdiff, double epsmiddiff)
{
  int i, j, ns, ne;
  int iz, iy, ix, ic, Nc=3;
  double shift=0.5;
  double zd;
  int izd;
  int ik;
  double value;
  PetscErrorCode ierr;

  ierr = VecGetOwnershipRange(epsDiff, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;
    ic = (j /= Nx) % Nc; // modified, Nc = 3;

    for (ik=0;ik<nlayers;ik++) {

      zd = (iz-Nzo[ik]) + (ic!= 2)*shift;
      izd = ceil(zd - 0.5);

      if(izd >= 0 && izd < Mz[ik]) {
	value=epsdiff[ik];
	break;
      }

      if (izd < Nzo[0] - Nzo[ik]) {
	value=epssubdiff;
      }else if(izd >= Nzo[nlayers-1] - Nzo[ik] + Mz[nlayers-1]) {
	value=epsairdiff;
      }else{ 
	value=epsmiddiff;
      }

    }

    ierr = VecSetValue(epsDiff, i, value, INSERT_VALUES); CHKERRQ(ierr);
  
  }

  ierr = VecAssemblyBegin(epsDiff); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsDiff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
