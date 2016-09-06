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

/* MAKE SURE THAT Nz = Lz + Mz + Lz  OR CHOOSE THE UPPER LIMIT */

#undef __FUNCT__ 
#define __FUNCT__ "boosterinterp"
PetscErrorCode boosterinterp(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, int Nxo, int Nyo, int Nzo, int Mx, int My, int Mz, int Mzslab, int Lz, int anisotropic)
{
  Mat A;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;
  double shift =  0.5;
  int i;
  int Nc = 3; //modified;

  int Mxyz =  Mx*My*((Mzslab==0)?Mz:1);  
  int DegFree1 = (anisotropic ? 3 : 1)* Mxyz; 
  int DegFree2 = Lz;
  int DegFree = DegFree1 + DegFree2;

  //ierr = MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, Nx*Ny*Nz*6, Mxyz, nz, NULL, nz, NULL, &A); CHKERRQ(ierr);
  
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

    zd = (iz-Nzo) + (ic!= 2)*shift;
    izd = ceil(zd - 0.5);
    if (izd < 0 || ( izd >= Mz && Lz ==0 )) continue;
    
    if (izd < Mz){

      if(Mzslab!=0)
	id = (ixd*My + iyd) + (anisotropic!=0)*ic*Mxyz;
      else
	id = (ixd*My + iyd)*Mz + izd + (anisotropic!=0)*ic*Mxyz;

    }else if(izd >= Mz && (izd-Mz)+DegFree1 < DegFree){
      id = (izd-Mz) + DegFree1;
    }else continue;

    ierr = MatSetValue(A, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) A,"InterpMatrix"); CHKERRQ(ierr);
  *Aout = A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "makethreelayeredepsbkg"
PetscErrorCode makethreelayeredepsbkg(Vec epsBkg, int Nx, int Ny, int Nz, int Nzo, int Mz, double epsbkg1, double epsbkg2, double epsbkg3)
{
  int i, j, ns, ne;
  int iz, iy, ix, ic, Nc=3;
  double shift=0.5;
  double zd;
  int izd;
  double value;
  PetscErrorCode ierr;

  ierr = VecGetOwnershipRange(epsBkg, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;
    ic = (j /= Nx) % Nc; // modified, Nc = 3;

    zd = (iz-Nzo) + (ic!= 2)*shift;
    izd = ceil(zd - 0.5);
    if(izd < 0){
      value=epsbkg1;
    }else if(izd >= 0 && izd < Mz){
      value=epsbkg2;
    }else{
      value=epsbkg3;
    }
    
    ierr = VecSetValue(epsBkg, i, value, INSERT_VALUES); CHKERRQ(ierr);
  
  }

  ierr = VecAssemblyBegin(epsBkg); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsBkg); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "makethreelayeredepsdiff"
PetscErrorCode makethreelayeredepsdiff(Vec epsDiff, int Nx, int Ny, int Nz, int Nzo, int Mz, double epsdiff1, double epsdiff2, double epsdiff3)
{
  int i, j, ns, ne;
  int iz, iy, ix, ic, Nc=3;
  double shift=0.5;
  double zd;
  int izd;
  double value;
  PetscErrorCode ierr;

  ierr = VecGetOwnershipRange(epsDiff, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;
    ic = (j /= Nx) % Nc; // modified, Nc = 3;

    zd = (iz-Nzo) + (ic!= 2)*shift;
    izd = ceil(zd - 0.5);
    if (izd < 0){
      value=epsdiff1;
    }else if(izd >= 0 && izd < Mz){
      value=epsdiff2;
    }else{
      value=epsdiff3;
    }
    
    ierr = VecSetValue(epsDiff, i, value, INSERT_VALUES); CHKERRQ(ierr);
  
  }

  ierr = VecAssemblyBegin(epsDiff); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsDiff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetWeightVecGeneralSym"
PetscErrorCode GetWeightVecGeneralSym(Vec weight,int Nx, int Ny, int Nz, int lx, int ly, int lz)
{
  PetscErrorCode ierr;
  int i, j, ns, ne, ix, iy, iz, ic;
  double value, tmp;

  int Nc = 3;
  ierr = VecGetOwnershipRange(weight,&ns,&ne); CHKERRQ(ierr);

  /*
  if (Nx == 1 || Ny == 1 || Nz == 1)
    {
      tmp = 4.0;
      PetscPrintf(PETSC_COMM_WORLD,"---Caution! Treat as a 2D problem and Weight is divieded by 2 \n");
    }
  else
    tmp = 8.0;
  */

  tmp=8.0;
  if(Nx==1 && lx!=0) PetscPrintf(PETSC_COMM_WORLD, "***WARNING: CONTRADICTION IN SYM WEIGHT Nx=1 but lowerpmlx=/=0***\n");
  if(Ny==1 && ly!=0) PetscPrintf(PETSC_COMM_WORLD, "***WARNING: CONTRADICTION IN SYM WEIGHT Ny=1 but lowerpmly=/=0***\n");
  if(Nz==1 && lz!=0) PetscPrintf(PETSC_COMM_WORLD, "***WARNING: CONTRADICTION IN SYM WEIGHT Nz=1 but lowerpmlz=/=0***\n");
  if(Nx==1) tmp=tmp/2;
  if(Ny==1) tmp=tmp/2;
  if(Nz==1) tmp=tmp/2;
  if(lx!=0) tmp=tmp/2;
  if(ly!=0) tmp=tmp/2;
  if(lz!=0) tmp=tmp/2;

  for(i=ns; i<ne; i++)
    {
      iz = (j = i) % Nz;
      iy = (j /= Nz) % Ny;
      ix = (j /= Ny) % Nx;
      ic = (j /= Nx) % Nc;

      value = tmp;

      if(ic==0)
	value = value/( ((ly==0)*(iy==0)+1.0) * ((lz==0)*(iz==0)+1.0) );
      if(ic==1)
	value = value/( ((lx==0)*(ix==0)+1.0) * ((lz==0)*(iz==0)+1.0) );
      if(ic==2)
	value = value/( ((lx==0)*(ix==0)+1.0) * ((ly==0)*(iy==0)+1.0) );

      VecSetValue(weight, i, value, INSERT_VALUES);
    }
  ierr = VecAssemblyBegin(weight); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(weight); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
