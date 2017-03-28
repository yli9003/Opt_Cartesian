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

extern int pSIMP;
extern double bproj, etaproj;
extern Mat Hfilt;
extern KSP kspH;
extern int itsH;

extern char filenameComm[PETSC_MAX_PATH_LEN];

extern double mintrans;

#undef __FUNCT__ 
#define __FUNCT__ "batchmeta"
double batchmeta(int DegFree,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;

  Meta *ptdata = (Meta *) data;

  int Nx = ptdata->Nx;
  int Ny = ptdata->Ny;
  int Nz = ptdata->Nz;
  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  double omega = ptdata->omega;
  Mat M = ptdata->M;
  Mat A = ptdata->A;
  Vec b = ptdata->b;
  Vec x = ptdata->x;
  Vec epspmlQ  = ptdata->epspmlQ;
  Vec epsmedium = ptdata->epsmedium;
  Vec epsDiff = ptdata->epsDiff;
  Vec epscoef = ptdata->epscoef;  
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its; 
  Vec pvec = ptdata->pvec;
  Vec qvec = ptdata->qvec;
  int outputbase = ptdata->outputbase;

  PetscPrintf(PETSC_COMM_WORLD,"----Calculating Batch Metasurface Phase. ------- \n");

  Vec xconj,xmag,xmagsq,xmagrecp,xpq,xpqmagsq,phasesumvec,tmp,Grad,U,u,vgrad,epsgrad;
  
  VecDuplicate(vR,&xconj);
  VecDuplicate(vR,&xmag);
  VecDuplicate(vR,&xmagsq);
  VecDuplicate(vR,&xmagrecp);
  VecDuplicate(vR,&xpq);
  VecDuplicate(vR,&xpqmagsq);
  VecDuplicate(vR,&phasesumvec);
  VecDuplicate(vR,&tmp);
  VecDuplicate(vR,&Grad);
  VecDuplicate(vR,&U);
  VecDuplicate(vR,&u);
  VecDuplicate(epsSReal,&vgrad);
  VecDuplicate(epsSReal,&epsgrad);

  RegzProj(DegFree,epsopt,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);

  MatMult(A,epsSReal,epsFReal);

  // Update the diagonals of M;
  Mat Mtmp;
  MatDuplicate(M,MAT_COPY_VALUES,&Mtmp);
  ModifyMatDiag(Mtmp, D, epsFReal, epsDiff, epsmedium, epspmlQ, omega, Nx, Ny, Nz);

  // solve the two fundamental modes and their ldos
  SolveMatrix(PETSC_COMM_WORLD,ksp,Mtmp,b,x,its);
  
  /*
  OutputMat(PETSC_COMM_WORLD,M,"M",".m");
  OutputMat(PETSC_COMM_WORLD,Mtmp,"Mtmp",".m");
  OutputVec(PETSC_COMM_WORLD,epsDiff,"epsDiff",".m");
  OutputVec(PETSC_COMM_WORLD,b,"b",".m");
  OutputVec(PETSC_COMM_WORLD,x,"x",".m");
  */

  //Note: 
  //everything is vector field; PointwiseMult cannot be used except for vR.
  //vector * scalar = vector (PointwiseMult can be used.)
  //scalar * scalar = scalar (PointwiseMult can be used.)
  //vector * vector = vector (CmpVecProd must be used.)

  VecWAXPY(xpq,1.0,x,qvec);
  MatMult(C,xpq,tmp);
  CmpVecProd(xpq,tmp,xpqmagsq);
  MatMult(C,x,xconj);
  CmpVecProd(x,xconj,xmagsq);
  VecWAXPY(phasesumvec,-1.0,xmagsq,xpqmagsq);
  VecAXPY(phasesumvec,-1.0,pvec);
  VecCopy(xmagsq,xmag);
  VecSqrtAbs(xmag);
  MatMult(D,xmag,tmp);
  VecWAXPY(xmagrecp,1.0,xmag,tmp);
  VecReciprocal(xmagrecp);
  VecPointwiseMult(xmagrecp,xmagrecp,vR);
  CmpVecProd(xmagrecp,phasesumvec,tmp);
  CmpVecProd(tmp,pvec,phasesumvec);
  VecPointwiseMult(phasesumvec,phasesumvec,vR);

  double phasesum;
  VecSum(phasesumvec,&phasesum);
  PetscPrintf(PETSC_COMM_WORLD,"---phase sum for freq %.4e at step %d is: %.8e\n", omega/(2*PI),count,phasesum);

  double norm;
  VecSum(pvec,&norm);
  norm=2*norm;
  PetscPrintf(PETSC_COMM_WORLD,"---normalized_phasesum for freq %.4e and at step %d is: %.8e\n", omega/(2*PI),count,phasesum/norm);

  /*-------------- Now store the epsilon at each step--------------*/

  char buffer [100];

  int STORE=1;    
  if(STORE==1 && (count%outputbase==0))
    {
      sprintf(buffer,"%.5depsSReal.m",count);
      OutputVec(PETSC_COMM_WORLD, epsSReal, filenameComm, buffer);
      
      FILE *tmpfile;
      int i;
      tmpfile = fopen(strcat(buffer,"DOF.txt"),"w");
      for (i=0;i<DegFree;i++){
      fprintf(tmpfile,"%0.16e \n",epsopt[i]);}
      fclose(tmpfile);

    }
  
  /*------------------------------------------------*/
  /*------------------------------------------------*/

  /*-------take care of the gradient---------*/
  if (grad) {

    CmpVecProd(pvec,qvec,tmp);
    CmpVecProd(tmp,xmagrecp,U);
    VecScale(U,2.0);
    CmpVecProd(x,phasesumvec,tmp);
    CmpVecProd(tmp,xmagrecp,Grad);
    CmpVecProd(Grad,xmagrecp,tmp);
    VecAXPY(U,-1.0,tmp);
    KSPSolveTranspose(ksp,U,u);

    MatMult(C,u,Grad);
    CmpVecProd(Grad,epscoef,tmp);
    CmpVecProd(tmp,x,Grad);
    VecPointwiseMult(Grad,Grad,vR);

    VecScale(Grad,1/norm);

    MatMultTranspose(A,Grad,vgrad);
    
    //correction from filters
    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);
    KSPSolveTranspose(kspH,vgrad,epsgrad);
    
    // copy vgrad (distributed vector) to a regular array grad;
    ierr = VecToArray(epsgrad,grad,scatter,from,to,vgradlocal,DegFree);
  
  }

  count++;

  VecDestroy(&xconj);
  VecDestroy(&xmag);
  VecDestroy(&xmagsq);
  VecDestroy(&xmagrecp);
  VecDestroy(&xpq);
  VecDestroy(&xpqmagsq);
  VecDestroy(&phasesumvec);
  VecDestroy(&tmp);
  VecDestroy(&Grad);
  VecDestroy(&U);
  VecDestroy(&u);
  VecDestroy(&vgrad);
  VecDestroy(&epsgrad);

  MatDestroy(&Mtmp);

  return phasesum/norm;
}



#undef __FUNCT__ 
#define __FUNCT__ "makepq_defl"
PetscErrorCode makepq_defl(MPI_Comm comm, Vec *pout, Vec *qout, int Nx, int Ny, int Nz, int lx, int ux, int ly, int uy, int lz, int uz, int dir, double theta, double lambda, double refphi)
{
  int i, j, k, pos, N;
  N = Nx*Ny*Nz;

  Vec pvec, qvec, qvecconj;
  PetscErrorCode ierr;
  ierr = VecCreate(comm,&qvec);CHKERRQ(ierr);
  ierr = VecSetSizes(qvec,PETSC_DECIDE,6*N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(qvec); CHKERRQ(ierr);
  VecSet(qvec,0.0);

  int ns, ne;
  ierr = VecGetOwnershipRange(qvec, &ns, &ne); CHKERRQ(ierr);

  double dl, phi, ampr, ampi;
  
  for (i=0;i<Nx;i++)
    if ((i>=lx) && (i<ux))
      {for (j=0; j<Ny;j++)
	  if ((j>=ly) && (j<uy))
	    { 
	      for (k=0; k<Nz; k++)
		if ((k>=lz) && (k<uz)) // uncomment this if z direction is not trivial;
		  { pos = i*Ny*Nz + j*Nz + k;
		    if ( ns < pos+dir*N+1 && ne > pos+dir*N){
		      dl=((ux-lx>1)*(i-lx) + (uy-ly>1)*(j-ly) + (uz-lz>1)*(k-lz));
		      phi = refphi - (2*PI/lambda) * sin(theta) * dl;
		      ampr=cos(phi);
		      ampi=sin(phi);
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





#undef __FUNCT__ 
#define __FUNCT__ "makepq_lens"
PetscErrorCode makepq_lens(MPI_Comm comm, Vec *pout, Vec *qout, int Nx, int Ny, int Nz, int lx, int ux, int ly, int uy, int lz, int uz, int dir, double focallength, double lambda, double refphi)
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

  /*
  double *tmp;
  tmp=(double *)malloc(6*N*sizeof(double));
  for(i=0;i<6*N;i++) tmp[i]=0;
  for (i=0;i<Nx;i++)
    {if ((i>=lx) && (i<ux))
	{for (j=0; j<Ny;j++)
	    {if ((j>=ly) && (j<uy))
		{for (k=0; k<Nz; k++)
		    {if ((k>=lz) && (k<uz)) {
			pos = i*Ny*Nz + j*Nz + k;
			dl=((ux-lx>1)*(i-lx) + (uy-ly>1)*(j-ly) + (uz-lz>1)*(k-lz));
			phi = refphi - (2*PI/lambda) * (sqrt(dl*dl + focallength*focallength)-focallength);
			ampr=cos(phi);
			ampi=sin(phi);
			//PetscPrintf(comm,"DEBUG: dl %g phi %g \n",dl,phi);
			if(dir==0) {
			  tmp[pos+0*N]=ampr;
			  tmp[pos+3*N]=ampi;
			}else if(dir==1){
			  tmp[pos+1*N]=ampr;
			  tmp[pos+4*N]=ampi;
			}else{
			  tmp[pos+2*N]=ampr;
			  tmp[pos+5*N]=ampi;
			}
		      }
		    }
		}
	    }
	}
    }
  VecSet(qvec,0.0);
  VecAssemblyBegin(qvec);
  VecAssemblyEnd(qvec); 
  ArrayToVec(tmp,qvec);
  free(tmp);
  */

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

#undef __FUNCT__
#define __FUNCT__ "batchmaximin"
double batchmaximin(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  int DegFree=DegFreeAll-1;
  double *epsopt, *grad;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  grad = (double *) malloc(DegFree*sizeof(double));
  int i;
  for(i=0;i<DegFree;i++){
    epsopt[i]=epsoptAll[i];
  }
  double obj=batchmeta(DegFree,epsopt,grad,data);
  count=count-1;
  for(i=0;i<DegFree;i++){
    gradAll[i]=-1.0*grad[i];
  }
  gradAll[DegFreeAll-1]=1.0;

  return epsoptAll[DegFreeAll-1] - obj;
  
}

#undef __FUNCT__
#define __FUNCT__ "maximinobjfun"
double maximinobjfun(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{

  if(gradAll)
    {
      int i;
      for (i=0;i<DegFreeAll-1;i++)
	{
          gradAll[i]=0;
	}
      gradAll[DegFreeAll-1]=1;
    }

  PetscPrintf(PETSC_COMM_WORLD,"**the current value of dummy objective variable is %.8e**\n",epsoptAll[DegFreeAll-1]);

  count++;

  return epsoptAll[DegFreeAll-1];
}

#undef __FUNCT__
#define __FUNCT__ "refphiopt"
double refphiopt(int ndof,double *refphi, double *grad, void *data)
{

  PetscErrorCode ierr;

  Meta *ptdata = (Meta *) data;

  double omega = ptdata->omega;
  Vec x = ptdata->x;
  Vec pvec = ptdata->pvec;
  Vec qvec = ptdata->qvec;

  PetscPrintf(PETSC_COMM_WORLD,"----Modifying qvec. ------- \n");

  PetscPrintf(PETSC_COMM_WORLD,"---refphi0 for freq %g and at step %d is %g (formerly).\n", omega/(2*PI),count,*(ptdata->refphi));
  double phi=*refphi-*(ptdata->refphi);
  *(ptdata->refphi)=*refphi;
  PetscPrintf(PETSC_COMM_WORLD,"---refphi0 for freq %g and at step %d is %g (now).\n", omega/(2*PI),count,*(ptdata->refphi));

  Vec xconj,xmag,xmagsq,xmagrecp,xpq,xpqmagsq,phasesumvec,tmp;
  VecDuplicate(vR,&xconj);
  VecDuplicate(vR,&xmag);
  VecDuplicate(vR,&xmagsq);
  VecDuplicate(vR,&xmagrecp);
  VecDuplicate(vR,&xpq);
  VecDuplicate(vR,&xpqmagsq);
  VecDuplicate(vR,&phasesumvec);
  VecDuplicate(vR,&tmp);

  CmpVecScale(qvec, tmp, cos(phi), sin(phi));
  VecCopy(tmp,qvec);

  VecWAXPY(xpq,1.0,x,qvec);
  MatMult(C,xpq,tmp);
  CmpVecProd(xpq,tmp,xpqmagsq);
  MatMult(C,x,xconj);
  CmpVecProd(x,xconj,xmagsq);
  VecWAXPY(phasesumvec,-1.0,xmagsq,xpqmagsq);
  VecAXPY(phasesumvec,-1.0,pvec);
  VecCopy(xmagsq,xmag);
  VecSqrtAbs(xmag);
  VecPointwiseMult(xmag,xmag,vR);
  MatMult(D,xmag,tmp);
  VecWAXPY(xmagrecp,1.0,xmag,tmp);
  VecReciprocal(xmagrecp);
  VecPointwiseMult(xmagrecp,xmagrecp,vR);
  CmpVecProd(xmagrecp,phasesumvec,tmp);
  CmpVecProd(tmp,pvec,phasesumvec);
  VecPointwiseMult(phasesumvec,phasesumvec,vR);



  double phasesum;
  VecSum(phasesumvec,&phasesum);
  PetscPrintf(PETSC_COMM_WORLD,"---phase sum for freq %.4e and at step %d is: %.8e\n", omega/(2*PI),count,phasesum);

  double norm;
  VecSum(pvec,&norm);
  norm=2*norm;
  PetscPrintf(PETSC_COMM_WORLD,"---normalized_phasesum for freq %.4e and at step %d is: %.8e\n", omega/(2*PI),count,phasesum/norm);

  count++;

  VecDestroy(&xconj);
  VecDestroy(&xmag);
  VecDestroy(&xmagsq);
  VecDestroy(&xmagrecp);
  VecDestroy(&xpq);
  VecDestroy(&xpqmagsq);
  VecDestroy(&phasesumvec);
  VecDestroy(&tmp);

  if(grad)
    PetscPrintf(PETSC_COMM_WORLD,"---ERROR: you must not use a gradient algorithm for ref optimization.\n");

  return phasesum;
}
