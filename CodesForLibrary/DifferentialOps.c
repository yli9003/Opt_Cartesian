#include <stdio.h>
#include <math.h>
#include <petsc.h>

/*First order derivative operator D(c2,p,c1) takes the center difference of component c1 with respect to p coordinate and put it in the position of c2 component */
/*Choose c2 = 0/1/2, p = 0/1/2, c1 = 0/1/2 corresponding to x/y/z*/
#undef __FUNCT__
#define __FUNCT__ "firstorderDeriv"
PetscErrorCode firstorderDeriv(MPI_Comm comm, Mat *Dout, int Nx, int Ny, int Nz, double dh, int c2, int p, int c1)
{
  int Nc=3, N=2*Nc*Nx*Ny*Nz;
  Mat Dtmp;
  int ns,ne,i,j,ix,iy,iz,ic,iq;
  int crow=c2, ccol=c1;
  int colx1, colx2, coly1, coly2, colz1, colz2;
  int col1, col2;
  double val1=-1/dh, val2=1/dh;
  PetscErrorCode ierr;
  int pos[3], Nxyz[3]={Nx,Ny,Nz};

  MatCreate(comm, &Dtmp);
  MatSetType(Dtmp,MATMPIAIJ);
  MatSetSizes(Dtmp,PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatMPIAIJSetPreallocation(Dtmp, 2, PETSC_NULL, 2, PETSC_NULL);

  ierr = MatGetOwnershipRange(Dtmp, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;
    ic = (j /= Nx) % Nc;
    iq =  j /= Nc;
    pos[0]=ix;
    pos[1]=iy;
    pos[2]=iz;

    if(ic==crow){
      if(pos[p]>0){
        colx1 = (p==0) ? ix-1 : ix;
	coly1 = (p==1) ? iy-1 : iy;
	colz1 = (p==2) ? iz-1 : iz;
	col1  = iq*Nc*Nx*Ny*Nz + ccol*Nx*Ny*Nz + colx1*Ny*Nz + coly1*Nz + colz1;
        ierr  = MatSetValue(Dtmp,i,col1,val1,INSERT_VALUES); CHKERRQ(ierr);
      }
      if(pos[p]<Nxyz[p]-1){
	colx2 = (p==0) ? ix+1 : ix;
        coly2 = (p==1) ? iy+1 : iy;
	colz2 = (p==2) ? iz+1 : iz;
	col2  = iq*Nc*Nx*Ny*Nz + ccol*Nx*Ny*Nz + colx2*Ny*Nz + coly2*Nz + colz2;
        ierr  = MatSetValue(Dtmp,i,col2,val2,INSERT_VALUES); CHKERRQ(ierr);
      }
    }


  }

  ierr = MatAssemblyBegin(Dtmp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Dtmp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) Dtmp,"Dc2pc1"); CHKERRQ(ierr);

  *Dout = Dtmp;
  PetscFunctionReturn(0);
}
