#include <stdio.h>
#include <math.h>
#include <petsc.h>

#undef __FUNCT__
#define __FUNCT__ "GetDotMat"
PetscErrorCode GetDotMat(MPI_Comm comm, Mat *Bout, int Nx, int Ny, int Nz)
{
  int Nxyz=Nx*Ny*Nz, N=6*Nx*Ny*Nz, Nc=3;
  Mat B;
  int ns,ne,i,j,ic;
  double value[3]={1,1,1};
  int col[3];
  PetscErrorCode ierr;
  
  MatCreate(comm, &B);
  MatSetType(B,MATMPIAIJ);
  MatSetSizes(B,PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatMPIAIJSetPreallocation(B, 3, PETSC_NULL, 3, PETSC_NULL);

  ierr = MatGetOwnershipRange(B, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    j=i;
    ic = (j /= Nxyz) % Nc;
    
    if(ic==0){col[0]=i,        col[1]=i+Nxyz, col[2]=i+2*Nxyz;}
    if(ic==1){col[0]=i-Nxyz,   col[1]=i,      col[2]=i+Nxyz;  }
    if(ic==2){col[0]=i-2*Nxyz, col[1]=i-Nxyz, col[2]=i;       }

    ierr = MatSetValues(B,1,&i,3,col,value,INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) B,"DotMatrix"); CHKERRQ(ierr);
  
  *Bout = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "ImagIMat"
PetscErrorCode ImagIMat(MPI_Comm comm, Mat *Dout, int N)
{
  Mat D;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;  
  int i;
     
  //ierr = MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, N,N,nz, NULL, nz, NULL, &D); CHKERRQ(ierr); // here N is total length;
  
  MatCreate(comm, &D);
  MatSetType(D,MATMPIAIJ);
  MatSetSizes(D,PETSC_DECIDE, PETSC_DECIDE,N,N);
  MatMPIAIJSetPreallocation(D, nz, PETSC_NULL, nz, PETSC_NULL);

  ierr = MatGetOwnershipRange(D, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    int id = (i+N/2)%(N);
    double sign = pow(-1.0, (i<N/2));
    ierr = MatSetValue(D, i, id, sign, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) D, "ImaginaryIMatrix"); CHKERRQ(ierr);
  
  *Dout = D;
  PetscFunctionReturn(0);
}



#undef __FUNCT__ 
#define __FUNCT__ "CongMat"
PetscErrorCode CongMat(MPI_Comm comm, Mat *Cout, int N)
{
  Mat C;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;  
  int i;
     
  //ierr = MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, N,N,nz, NULL, nz, NULL, &C); CHKERRQ(ierr);
  
  MatCreate(comm, &C);
  MatSetType(C,MATMPIAIJ);
  MatSetSizes(C,PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatMPIAIJSetPreallocation(C, nz, PETSC_NULL, nz, PETSC_NULL);

  ierr = MatGetOwnershipRange(C, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    double sign = pow(-1.0, (i>(N/2-1)));
    ierr = MatSetValue(C, i, i, sign, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) C,"CongMatrix"); CHKERRQ(ierr);
  
  *Cout = C;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "GetWeightVec"
PetscErrorCode GetWeightVec(Vec weight,int Nx, int Ny, int Nz)
{
   PetscErrorCode ierr;
   int i, j, ns, ne, ix, iy, iz, ic;
   double value, tmp;

   int Nc = 3;
   ierr = VecGetOwnershipRange(weight,&ns,&ne); CHKERRQ(ierr);
   
   if (Nx == 1 || Ny == 1 || Nz == 1)
     {
       tmp = 4.0;
       PetscPrintf(PETSC_COMM_WORLD,"---Caution! Treat as a 2D problem and Weight is divieded by 2 \n");
     }
   else
     tmp = 8.0;

   for(i=ns; i<ne; i++)
     {
       iz = (j = i) % Nz;
       iy = (j /= Nz) % Ny;
       ix = (j /= Ny) % Nx;
       ic = (j /= Nx) % Nc;       

       value = tmp; // tmp = 8.0 for 3D and 4.0 for 2D

       if(ic==0)
	 value = value/(((iy==0)+1.0)*((iz==0)+1.0));
       if(ic==1)
	 value = value/(((ix==0)+1.0)*((iz==0)+1.0));
       if(ic==2)
	 value = value/(((ix==0)+1.0)*((iy==0)+1.0));

       VecSetValue(weight, i, value, INSERT_VALUES);
     }
   ierr = VecAssemblyBegin(weight); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(weight); CHKERRQ(ierr);

   PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "GetMediumVec"
PetscErrorCode GetMediumVec(Vec epsmedium,int Nz, int Mz, double epsair, double epssub)
{
   PetscErrorCode ierr;
   int i, iz, ns, ne;
   double value;
   ierr = VecGetOwnershipRange(epsmedium,&ns,&ne); CHKERRQ(ierr);
   for(i=ns;i<ne; i++)
     {
       iz = i%Nz;
       if (iz<Mz)
	 value = epssub;
       else
	 value = epsair;
        
       VecSetValue(epsmedium, i, value, INSERT_VALUES);

       }

   ierr = VecAssemblyBegin(epsmedium); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(epsmedium); CHKERRQ(ierr);

   PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "GetMediumVecwithSub"
PetscErrorCode GetMediumVecwithSub(Vec epsmedium,int Nz, int Mz, double epsair, double epssub)
{
   PetscErrorCode ierr;
   int i, iz, ns, ne;
   double value;
   ierr = VecGetOwnershipRange(epsmedium,&ns,&ne); CHKERRQ(ierr);
   for(i=ns;i<ne; i++)
     {
       iz = i%Nz;
       if (iz<Nz/2 + Mz/2)
	 value = epsair;
       else
	 value = epssub;
        
       VecSetValue(epsmedium, i, value, INSERT_VALUES);

       }

   ierr = VecAssemblyBegin(epsmedium); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(epsmedium); CHKERRQ(ierr);

   PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "GetRealPartVec"
PetscErrorCode GetRealPartVec(Vec vR, int N)
{
   PetscErrorCode ierr;
   int i, ns, ne;

   ierr = VecGetOwnershipRange(vR,&ns,&ne); CHKERRQ(ierr);

   for(i=ns; i<ne; i++)
     {
       if (i<N/2)
	 VecSetValue(vR,i,1.0,INSERT_VALUES);
       else
	 VecSetValue(vR,i,0.0,INSERT_VALUES);
     }

   ierr = VecAssemblyBegin(vR); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(vR); CHKERRQ(ierr);
   
   PetscFunctionReturn(0);
  
}

#undef __FUNCT__ 
#define __FUNCT__ "AddMuAbsorption"
PetscErrorCode AddMuAbsorption(double *muinv, Vec muinvpml, double Qabs, int add)
{
  //compute muinvpml/(1+i/Qabs)
  double Qinv = (add==0) ? 0.0: (1.0/Qabs);
  double d=1 + pow(Qinv,2);
  PetscErrorCode ierr;
  int N;
  ierr=VecGetSize(muinvpml,&N);CHKERRQ(ierr);

  double *ptmuinvpml;
  ierr=VecGetArray(muinvpml, &ptmuinvpml);CHKERRQ(ierr);

  int i;
  double a,b;
  for(i=0;i<N/2;i++)
    {
      a=ptmuinvpml[i];
      b=ptmuinvpml[i+N/2];      
      muinv[i]= (a+b*Qinv)/d;
      muinv[i+N/2]=(b-a*Qinv)/d;
    }
  ierr=VecRestoreArray(muinvpml,&ptmuinvpml);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "GetUnitVec"
PetscErrorCode GetUnitVec(Vec ej, int pol, int N)
{
   PetscErrorCode ierr;
   int i, j, ns, ne, ic;
   int Nc=3;
   int Nxyz=N/6;

   ierr = VecGetOwnershipRange(ej,&ns,&ne); CHKERRQ(ierr);

   for(i=ns; i<ne; i++)
     {
       j=i;
       ic = (j /= Nxyz) % Nc;
       
       if (ic==pol)
	 VecSetValue(ej,i,1.0,INSERT_VALUES);
       else
	 VecSetValue(ej,i,0.0,INSERT_VALUES);
     }

   ierr = VecAssemblyBegin(ej); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(ej); CHKERRQ(ierr);
   
   PetscFunctionReturn(0);
  
}

