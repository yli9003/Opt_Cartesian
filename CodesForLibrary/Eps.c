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


/* On Dec 3rd 2011, I combine myinterpSlab into the case myinterp so that the new my interp handle both slab and nonslab case; using Mzslab to indicate whether it's the slab case while Mz is for the thickness */

extern Mat A, B, C, D;
extern Vec vR, weight, ej, ek, epsSReal, epsFReal, vgrad, vgradlocal;

#undef __FUNCT__ 
#define __FUNCT__ "myinterp"
PetscErrorCode myinterp(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, int Nxo, int Nyo, int Nzo, int Mx, int My, int Mz, int Mzslab, int anisotropic)
{
  Mat A;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;
  double shift =  0.5;
  int i;
  int Nc = 3; //modified;

  int Mxyz =  Mx*My*((Mzslab==0)?Mz:1);  
  int DegFree= (anisotropic ? 3 : 1)* Mxyz; 
 
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
    ic = (j /= Nx) % Nc; // modifed, Nc = 3;

    xd = (ix-Nxo) + (ic!= 0)*shift;
    ixd = ceil(xd-0.5);
    if (ixd < 0 || ixd >= Mx) continue;
   
    yd = (iy-Nyo) + (ic!= 1)*shift;
    iyd = ceil(yd - 0.5);
    if (iyd < 0 || iyd >= My) continue;

    zd = (iz-Nzo) + (ic!= 2)*shift;
    izd = ceil(zd - 0.5);
    if (izd < 0 || izd >= Mz) continue;

    if(Mzslab!=0)
      id = (ixd*My + iyd);
    else
      id = (ixd*My + iyd)*Mz + izd;
	  
   
    ierr = MatSetValue(A, i, id + (anisotropic!=0)*ic*Mxyz, 1.0, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) A,
			    "InterpMatrix"); CHKERRQ(ierr);
  *Aout = A;
  PetscFunctionReturn(0);
}


// input  weight, epspml, Qabs, omega;| output: epspmlQ, epscoef;
PetscErrorCode EpsCombine(Mat D, Vec weight, Vec epspml, Vec epspmlQ, Vec epscoef, double Qabs, double omega, Vec epsilon)
{

  PetscErrorCode ierr;
  // compute epspmlQ = epspml*(1+i/Qabs);
  ierr =MatMult(D,epspml,epspmlQ); CHKERRQ(ierr);
  ierr =VecScale(epspmlQ, 1.0/Qabs); CHKERRQ(ierr);
  ierr =VecAXPY(epspmlQ, 1.0, epspml);CHKERRQ(ierr);
  
  
  // compute epscoef = omega^2*epsilon*epspmlQ;
  VecCopy(epspmlQ,epscoef);
  ierr = VecPointwiseMult(epscoef,epscoef,epsilon); CHKERRQ(ierr);
  
  double omegasqr=omega*omega;
  ierr = VecScale(epscoef,omegasqr);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "ModifyMatDiagonals"
PetscErrorCode ModifyMatDiagonals( Mat M, Mat A, Mat D, Vec epsSReal, Vec epspmlQ, Vec epsmedium, Vec epsC, Vec epsCi, Vec epsP, int Nxyz, double omega, Vec epsilon)
{

  PetscErrorCode ierr;
  /*-------Compute the epsdiff -----------------------------*/

  //compute current epsilon epsC;
  ierr =MatMult(A, epsSReal,epsC); CHKERRQ(ierr);
  VecCopy(epsC,epsFReal); 

  ierr = VecPointwiseMult(epsC,epsC,epsilon);
  ierr = VecAXPY(epsC,1.0,epsmedium); CHKERRQ(ierr);

  ierr = VecPointwiseMult(epsC,epsC,epspmlQ); CHKERRQ(ierr);
 
  //store the difference = current-previous into current epsilon;
  ierr =VecAXPY(epsC,-1.0,epsP);CHKERRQ(ierr);   // epsC is the epsDiff now;

  //compute epsCi = i*epsC;
  ierr =MatMult(D,epsC,epsCi);CHKERRQ(ierr); // make all the needed local;


  /*---------Modify diagonals of M (more than main diagonals)------*/

  int ns, ne;
  ierr = MatGetOwnershipRange(M, &ns, &ne); CHKERRQ(ierr);

  double *c, *ci;
  ierr = VecGetArray(epsC, &c); CHKERRQ(ierr);
  ierr = VecGetArray(epsCi, &ci); CHKERRQ(ierr);

  int i;
  double omegasqr=pow(omega,2.0);

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

      //M = M - omega^2*eps
      ierr = MatSetValue(M,i,i,-omegasqr*vr,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(M,i,(i+3*Nxyz)%(6*Nxyz), pow(-1,i/(3*Nxyz))*omegasqr*vi,ADD_VALUES);CHKERRQ(ierr);
    }
  ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = VecRestoreArray(epsC, &c); CHKERRQ(ierr);
  ierr = VecRestoreArray(epsCi, &ci); CHKERRQ(ierr);

  /*-----------update epsP to store current epsilons-------------*/
    
  ierr= VecAXPY(epsP,1.0,epsC); CHKERRQ(ierr);

 PetscFunctionReturn(0);
}

