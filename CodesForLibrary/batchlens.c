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

#undef __FUNCT__
#define __FUNCT__ "SourceAngled"
PetscErrorCode SourceAngled(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, double hx, double hy, double hz, double lx, double ux, double ly, double uy, double lz, double uz, double amp, int Jdir, double kx, double ky, double kz)
{
  int i, j, k, pos, N;
  N = Nx*Ny*Nz;

  Vec b;
  PetscErrorCode ierr;
  ierr = VecCreate(comm,&b);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) b, "Source");CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,6*N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b); CHKERRQ(ierr);
  VecSet(b,0.0);

  int ns, ne;
  ierr = VecGetOwnershipRange(b, &ns, &ne); CHKERRQ(ierr);


  for (i=0;i<Nx;i++)
    if ((i*hx>=lx) && (i*hx<ux))
      {for (j=0; j<Ny;j++)
          if ((j*hy>=ly) && (j*hy<uy))
            {
              for (k=0; k<Nz; k++)
                if ((k*hz>=lz) && (k*hz<uz)) 
		  { pos = i*Ny*Nz + j*Nz + k;
		    if ( ns < pos+Jdir*N+1 && ne > pos + Jdir*N){
		      preal=amp*cos(kx*i*hx+ky*j*hy+kz*k*hz);
		      pimag=amp*sin(kx*i*hx+ky*j*hy+kz*k*hz);
		      VecSetValue(b,pos+Jdir*N,    preal,INSERT_VALUES);
		      VecSetValue(b,pos+Jdir*N+3*N,pimag,INSERT_VALUES);
		    }
		  }
            }

      }
  VecAssemblyBegin(b);
  VecAssemblyEnd(b);

  *bout = b;
  PetscFunctionReturn(0);

}



#undef __FUNCT__ 
#define __FUNCT__ "makepq_lens_inc"
PetscErrorCode makepq_lens_inc(MPI_Comm comm, Vec *pout, Vec *qout, int Nx, int Ny, int Nz, int lx, int ux, int ly, int uy, int lz, int uz, int dir, double focallength, double lambda, double refphi)
{

  int i, j, k, pos, N;
  N = Nx*Ny*Nz;

  double dl, phi, ampr, ampi;
  Vec pvec, qvec, qvecconj;
  PetscErrorCode ierr;
  ierr = VecCreate(comm,&qvec);CHKERRQ(ierr);
  ierr = VecSetSizes(qvec,PETSC_DECIDE,6*N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(qvec); CHKERRQ(ierr);

  VecSet(qvec,0.0);
  int ns, ne;
  ierr = VecGetOwnershipRange(qvec, &ns, &ne); CHKERRQ(ierr);
  for (i=0;i<Nx;i++)
    {if ((i>=lx) && (i<ux))
	{for (j=0; j<Ny;j++)
	    {if ((j>=ly) && (j<uy))
		{for (k=0; k<Nz; k++)
		    {if ((k>=lz) && (k<uz)) 
			{pos = i*Ny*Nz + j*Nz + k;
			  if ( ns < pos+dir*N+1 && ne > pos+dir*N){
			    PetscPrintf(comm,"DEBUG: I AM HERE IN makepq_lens.\n");
 			    dl=((ux-lx>1)*(i-lx) + (uy-ly>1)*(j-ly) + (uz-lz>1)*(k-lz));
			    phi = refphi - (2*PI/lambda) * (sqrt(dl*dl + focallength*focallength)-focallength);
			    ampr=cos(phi);
			    ampi=sin(phi);
			    PetscPrintf(comm,"DEBUG: ampr %g, ampi %g \n",ampr,ampi);
			    if(dir==0) {
			      VecSetValue(qvec,pos+0*N,ampr,INSERT_VALUES);
			      VecSetValue(qvec,pos+3*N,ampi,INSERT_VALUES);
			    }else if(dir==1){
			      VecSetValue(qvec,pos+1*N,ampr,INSERT_VALUES);
			      VecSetValue(qvec,pos+4*N,ampi,INSERT_VALUES);
			    }else{
			      VecSetValue(qvec,pos+2*N,ampr,INSERT_VALUES);
			      VecSetValue(qvec,pos+5*N,ampi,INSERT_VALUES);
			    }
			  }
			}
		    }
		}
	    }
	}
    }
  VecAssemblyBegin(qvec);
  VecAssemblyEnd(qvec); 

  VecDuplicate(qvec,&pvec);
  VecDuplicate(qvec,&qvecconj);
  MatMult(C,qvec,qvecconj);
  CmpVecProd(qvec,qvecconj,pvec);
  VecPointwiseMult(pvec,pvec,vR);

  VecDestroy(&qvecconj);

  *qout = qvec;
  *pout = pvec;
  PetscFunctionReturn(0);
  
}
