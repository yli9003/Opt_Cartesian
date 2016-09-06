#include <petsc.h>
#include <time.h>
#include "libOPT.h"
#include <complex.h>
#include "petsctime.h"

#define vecproj vecHevproj //vectanhproj, vecHevproj
#define Ptime PetscTime

extern int maxit;
extern VecScatter scatter;
extern IS from, to;
extern Vec vgradlocal;

extern int pSIMP;
extern double bproj;
extern double etaproj;
extern Mat Hfilt;
extern int itsH;

#undef __FUNCT__ 
#define __FUNCT__ "vecdvpow"
void vecdvpow(double *u, double *v, double *dv, int n, int p)
{
  int i;
  double tmp1, tmp2;
  for(i=0;i<n;i++){
    tmp1 = (p==0)?1:pow(u[i],p);
    tmp2 = (p==1)?1:p*pow(u[i],p-1);
    v[i]=tmp1;
    dv[i]=tmp2;
  }
}

#undef __FUNCT__ 
#define __FUNCT__ "vectanhproj"
void vectanhproj(double *u, double *v, double *dv, int n, double b, double eta)
{
  double denom=tanh(b*eta)+tanh(b*(1-eta));

  int i;
  double tmp1, tmp2;
  for(i=0;i<n;i++){
    tmp1= ( tanh(b*eta)+tanh(b*(u[i]-eta)) )/denom;
    tmp2= b*(1-tanh(b*(u[i]-eta))*tanh(b*(u[i]-eta)))/denom;
    v[i]=tmp1;
    dv[i]=tmp2;
  }
}

#undef __FUNCT__ 
#define __FUNCT__ "vecHevproj"
void vecHevproj(double *u, double *v, double *dv, int n, double b, double eta)
{

  int i;
  double tmp1, tmp2;

  for(i=0;i<n;i++){

    if (u[i]<=eta){ 
	tmp1= eta * ( exp(-b*(1-u[i]/eta)) - (1-u[i]/eta)*exp(-b) );
	tmp2= eta * ( (b/eta)*exp(-b*(1-u[i]/eta)) + exp(-b)/eta );
    }else{
	tmp1= (1-eta) * ( 1 - exp(-b*(u[i]-eta)/(1-eta)) + (u[i] - eta)/(1-eta) * exp(-b) ) + eta;
	tmp2= (1-eta) * ( b/(1-eta) * exp(-b*(u[i]-eta)/(1-eta)) + exp(-b)/(1-eta) );
    }    
    v[i]=tmp1;
    dv[i]=tmp2;

  }
}

PetscErrorCode applyfilters(int DegFree, double *epsopt, Vec epsSReal, Vec epsgrad)
{

  PetscErrorCode ierr;

  double *epsarray, *tmpepsarray, *PROJepsgradarray, *SIMPepsgradarray;
  epsarray = (double *) malloc(DegFree*sizeof(double));
  tmpepsarray = (double *) malloc(DegFree*sizeof(double));
  PROJepsgradarray = (double *) malloc(DegFree*sizeof(double));
  SIMPepsgradarray = (double *) malloc(DegFree*sizeof(double));

  Vec PROJepsgrad, SIMPepsgrad;
  ierr=VecDuplicate(epsSReal,&PROJepsgrad); CHKERRQ(ierr);
  ierr=VecDuplicate(epsSReal,&SIMPepsgrad); CHKERRQ(ierr);

  vecproj(epsopt,tmpepsarray,PROJepsgradarray,DegFree,bproj,etaproj);
  vecdvpow(tmpepsarray,epsarray,SIMPepsgradarray,DegFree,pSIMP);
  ierr=ArrayToVec(epsarray,epsSReal); CHKERRQ(ierr);

  ierr=ArrayToVec(PROJepsgradarray,PROJepsgrad); CHKERRQ(ierr);
  ierr=ArrayToVec(SIMPepsgradarray,SIMPepsgrad); CHKERRQ(ierr);
  VecPointwiseMult(epsgrad,SIMPepsgrad,PROJepsgrad);

  ierr = VecDestroy(&PROJepsgrad); CHKERRQ(ierr);
  ierr = VecDestroy(&SIMPepsgrad); CHKERRQ(ierr);

  free(epsarray);
  free(tmpepsarray);
  free(PROJepsgradarray);
  free(SIMPepsgradarray);

return 0;
}

PetscErrorCode applyfiltersVer2(int DegFree, double *epsopt, Vec epsSReal, Vec epsgrad, double pSIMP, double bproj, double etaproj)
{

  PetscErrorCode ierr;

  double *epsarray, *tmpepsarray, *PROJepsgradarray, *SIMPepsgradarray;
  epsarray = (double *) malloc(DegFree*sizeof(double));
  tmpepsarray = (double *) malloc(DegFree*sizeof(double));
  PROJepsgradarray = (double *) malloc(DegFree*sizeof(double));
  SIMPepsgradarray = (double *) malloc(DegFree*sizeof(double));

  Vec PROJepsgrad, SIMPepsgrad;
  ierr=VecDuplicate(epsSReal,&PROJepsgrad); CHKERRQ(ierr);
  ierr=VecDuplicate(epsSReal,&SIMPepsgrad); CHKERRQ(ierr);

  vecproj(epsopt,tmpepsarray,PROJepsgradarray,DegFree,bproj,etaproj);
  vecdvpow(tmpepsarray,epsarray,SIMPepsgradarray,DegFree,pSIMP);
  ierr=ArrayToVec(epsarray,epsSReal); CHKERRQ(ierr);

  ierr=ArrayToVec(PROJepsgradarray,PROJepsgrad); CHKERRQ(ierr);
  ierr=ArrayToVec(SIMPepsgradarray,SIMPepsgrad); CHKERRQ(ierr);
  VecPointwiseMult(epsgrad,SIMPepsgrad,PROJepsgrad);

  ierr = VecDestroy(&PROJepsgrad); CHKERRQ(ierr);
  ierr = VecDestroy(&SIMPepsgrad); CHKERRQ(ierr);

  free(epsarray);
  free(tmpepsarray);
  free(PROJepsgradarray);
  free(SIMPepsgradarray);

  return 0;
}

PetscErrorCode GetH(MPI_Comm comm, Mat *Hout, int mx, int my, int mz, double s, double nR, int dim, KSP *kspHout, PC *pcHout)
{
  PetscErrorCode ierr;
  Mat H;
  int i,ns,ne;
  int N=mx*my*mz;

  MatCreate(comm, &H);
  MatSetType(H,MATMPIAIJ);
  MatSetSizes(H,PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatMPIAIJSetPreallocation(H, 3, PETSC_NULL, 3, PETSC_NULL);


  if (dim==1){ 
		double value[3];
  		int col[3];
		
		ierr = MatGetOwnershipRange(H, &ns, &ne); CHKERRQ(ierr);

  		for (i = ns; i < ne; ++i) {
    			if (i==0){ 
				col[0]= 0, value[0]= s*nR*nR*(-2.0)+1;
				col[1]= 1, value[1]= s*nR*nR*2.0;
		 		col[2]= 2, value[2]= 0;}
    			else if (i==N-1){ 
				col[0]= N-3, value[0]= 0;
				col[1]= N-2, value[1]= s*nR*nR*2.0;
		 		col[2]= N-1, value[2]= s*nR*nR*(-2.0)+1;}
			else{
				col[0]= i-1, value[0]= s*nR*nR;
				col[1]= i,   value[1]= s*nR*nR*(-2.0)+1;
		 		col[2]= i+1, value[2]= s*nR*nR;}

    			ierr = MatSetValues(H,1,&i,3,col,value,INSERT_VALUES); CHKERRQ(ierr);
  		}

		ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
		ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  
  }

  if (dim==2) {
    PetscPrintf(comm,"2D Helmholtz filter is not implemented yet!\n");
    MatShift(H,1.0);}

  if (dim==3) {
    PetscPrintf(comm,"3D Helmholtz filter is not implemented yet!\n");
    MatShift(H,1.0);}

  ierr = PetscObjectSetName((PetscObject) H,"Hop"); CHKERRQ(ierr);

  KSP ksp;
  PC pc; 
  
  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxit);CHKERRQ(ierr);

  ierr = PCSetFromOptions(pc);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  *Hout=H;
  *kspHout=ksp;
  *pcHout=pc;

  PetscFunctionReturn(0);

}

PetscErrorCode GetHdummy(MPI_Comm comm, Mat *Hout, int DegFree, KSP *kspHout, PC *pcHout)
{
  PetscErrorCode ierr;
  Mat H;
  int i,ns,ne;
  int N=DegFree;

  MatCreate(comm, &H);
  MatSetType(H,MATMPIAIJ);
  MatSetSizes(H,PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatMPIAIJSetPreallocation(H, 1, PETSC_NULL, 1, PETSC_NULL);
  MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);
  MatShift(H,1.0);

  ierr = PetscObjectSetName((PetscObject) H,"Hop"); CHKERRQ(ierr);

  KSP ksp;
  PC pc; 
  
  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxit);CHKERRQ(ierr);

  ierr = PCSetFromOptions(pc);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  *Hout=H;
  *kspHout=ksp;
  *pcHout=pc;

  PetscFunctionReturn(0);

}

PetscErrorCode GetH1d(MPI_Comm comm, Mat *Hout, int DegFree, double s, double nR, KSP *kspHout, PC *pcHout)
{
  PetscErrorCode ierr;
  Mat H;
  int i,ns,ne;
  int N=DegFree;

  MatCreate(comm, &H);
  MatSetType(H,MATMPIAIJ);
  MatSetSizes(H,PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatMPIAIJSetPreallocation(H, 3, PETSC_NULL, 3, PETSC_NULL);

  double value[3];
  int col[3];
  
  ierr = MatGetOwnershipRange(H, &ns, &ne); CHKERRQ(ierr);
  
  for (i = ns; i < ne; ++i) {
    if (i==0){ 
      col[0]= 0, value[0]= s*nR*nR*(-2.0)+1;
      col[1]= 1, value[1]= s*nR*nR*2.0;
      col[2]= 2, value[2]= 0;}
    else if (i==N-1){ 
      col[0]= N-3, value[0]= 0;
      col[1]= N-2, value[1]= s*nR*nR*2.0;
      col[2]= N-1, value[2]= s*nR*nR*(-2.0)+1;}
    else{
      col[0]= i-1, value[0]= s*nR*nR;
      col[1]= i,   value[1]= s*nR*nR*(-2.0)+1;
      col[2]= i+1, value[2]= s*nR*nR;}
    
    ierr = MatSetValues(H,1,&i,3,col,value,INSERT_VALUES); CHKERRQ(ierr);
  }
  
  ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) H,"Hop"); CHKERRQ(ierr);

  KSP ksp;
  PC pc; 
  
  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxit);CHKERRQ(ierr);

  ierr = PCSetFromOptions(pc);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  *Hout=H;
  *kspHout=ksp;
  *pcHout=pc;

  PetscFunctionReturn(0);

}

PetscErrorCode SolveH(MPI_Comm comm, KSP ksp, Mat H, Vec rhs, Vec sol)
{
  /*-----------------KSP Solving------------------*/   
  PetscErrorCode ierr;
  PetscLogDouble t1,t2,tpast;
  ierr = Ptime(&t1);CHKERRQ(ierr);

  if (itsH>(maxit-5)){
    ierr = KSPSetOperators(ksp,H,H);CHKERRQ(ierr);}
  else{
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);}

   ierr = KSPSolve(ksp,rhs,sol);CHKERRQ(ierr);
   ierr = KSPGetIterationNumber(ksp,&itsH);CHKERRQ(ierr);

   // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
   if(itsH>(maxit-2))
     {
	ierr = KSPSetOperators(ksp,H,H);CHKERRQ(ierr);
	ierr = KSPSolve(ksp,rhs,sol);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(ksp,&itsH);CHKERRQ(ierr);
     }

  //Print kspsolving information
  double norm;
  Vec xdiff;
  ierr=VecDuplicate(sol,&xdiff);CHKERRQ(ierr);
  ierr = MatMult(H,sol, xdiff);CHKERRQ(ierr);
  ierr = VecAXPY(xdiff,-1.0,rhs);CHKERRQ(ierr);
  ierr = VecNorm(xdiff,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"==> Helmholtz filter solution: norm of error %g, Kryolv Iterations %d----\n ",norm,itsH);CHKERRQ(ierr);    

  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0) PetscPrintf(PETSC_COMM_SELF,"==> Helmholtz filter solution: the runing time is %f s \n",tpast);
  /*--------------Finish KSP Solving---------------*/ 

  VecDestroy(&xdiff);
  PetscFunctionReturn(0);
}

PetscErrorCode RegzProj(int DegFree, double *epsopt,Vec epsSReal,Vec epsgrad,int pSIMP,double bproj,double etaproj,KSP kspH,Mat Hfilt,int *itsH)
{
  PetscErrorCode ierr; 
  Vec epsVec, epsH;
  double *epsoptH; 
  ierr=VecDuplicate(epsSReal,&epsVec); CHKERRQ(ierr);
  ierr=VecDuplicate(epsSReal,&epsH); CHKERRQ(ierr);
  epsoptH=(double *) malloc(DegFree*sizeof(double));
  ierr=ArrayToVec(epsopt,epsVec); CHKERRQ(ierr);
  SolveMatrix(PETSC_COMM_WORLD,kspH,Hfilt,epsVec,epsH,itsH);
  PetscPrintf(PETSC_COMM_WORLD,"*****DEBUG: I am here!******************\n");
  ierr = VecToArray(epsH,epsoptH,scatter,from,to,vgradlocal,DegFree);
  applyfiltersVer2(DegFree,epsoptH,epsSReal,epsgrad,pSIMP,bproj,etaproj);
  VecDestroy(&epsVec);
  VecDestroy(&epsH);
  free(epsoptH);
 
  PetscFunctionReturn(ierr);
}

PetscErrorCode RegzProjnoH(int DegFree, double *epsopt,Vec epsSReal,Vec epsgrad,int pSIMP,double bproj,double etaproj)
{
  PetscErrorCode ierr; 
  Vec epsVec, epsH;
  double *epsoptH; 
  ierr=VecDuplicate(epsSReal,&epsVec); CHKERRQ(ierr);
  ierr=VecDuplicate(epsSReal,&epsH); CHKERRQ(ierr);
  epsoptH=(double *) malloc(DegFree*sizeof(double));
  ierr=ArrayToVec(epsopt,epsVec); CHKERRQ(ierr);
  VecCopy(epsVec,epsH);
  ierr = VecToArray(epsH,epsoptH,scatter,from,to,vgradlocal,DegFree);
  applyfiltersVer2(DegFree,epsoptH,epsSReal,epsgrad,pSIMP,bproj,etaproj);
  VecDestroy(&epsVec);
  VecDestroy(&epsH);
  free(epsoptH);
 
  PetscFunctionReturn(ierr);
}

