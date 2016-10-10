#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

int mma_verbose;
int initdirect, maxit;

int count=1;
VecScatter scatter;
IS from, to;
Vec vgradlocal;
Mat B,C,D;
Vec vR, epsFReal;

int pSIMP;
double bproj, etaproj;
Mat Hfilt;
KSP kspH;
int itsH;

/*------------------------------------------------------*/

PetscErrorCode makeRefField(Maxwell maxwell, Universals params, Mat A, Mat C, Mat D, Vec vR, KSP ksp, int *its, Vec *ref, Vec *refconj, Vec VecPT);
PetscErrorCode setupKSP(MPI_Comm comm, KSP *ksp, PC *pc, int solver, int iteronly);
double pfunc(int DegFree, double *epsopt, double *grad, void *data);

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize ------*/
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0) 
    mma_verbose=1;

  /*************************************************************/
  PetscBool flg;

  getint("-initdirect",&initdirect,3);
  getint("-maxit",&maxit,15);
  int solver;
  PetscOptionsGetInt(PETSC_NULL,"-solver",&solver,&flg);  
  if(!flg) solver=1;
  PetscPrintf(PETSC_COMM_WORLD,"LU Direct solver choice (0 PASTIX, 1 MUMPS, 2 SUPERLU_DIST): %d\n",solver);
  double sH, nR;
  int dimH;
  PC pcH;
  getint("-pSIMP",&pSIMP,1);
  getreal("-bproj",&bproj,0);
  getreal("-etaproj",&etaproj,0.5);
  getreal("-sH",&sH,-1);
  getreal("-nR",&nR,0);
  getint("-dimH",&dimH,1);
  /***************************************************************/

  Universals flagparams;
  readfromflags(&flagparams);

  Mat A;
  Vec vI, weight;
  Vec epsSReal;

  setupMatVecs(flagparams, &A, &C, &D, &vR, &vI, &weight, &epsSReal, &epsFReal);

  ierr=VecCreateSeq(PETSC_COMM_SELF, flagparams.DegFree, &vgradlocal); CHKERRQ(ierr); 
  ISCreateStride(PETSC_COMM_SELF,flagparams.DegFree,0,1,&from); 
  ISCreateStride(PETSC_COMM_SELF,flagparams.DegFree,0,1,&to); 

  GetH1d(PETSC_COMM_WORLD,&Hfilt,flagparams.DegFree,sH,nR,&kspH,&pcH);
  /****************************************************************************/

  double metaphase;
  getreal("-metaphase",&metaphase,0);
  Vec VecPt;
  int ixref, iyref, izref, icref;
  int Nx=flagparams.Nx, Ny=flagparams.Ny, Nz=flagparams.Nz;
  int Mz=flagparams.Mz, Npmlz=flagparams.Npmlz;
  getint("-ixref",&ixref,floor(Nx/2));
  getint("-iyref",&iyref,floor(Ny/2));
  getint("-izref",&izref,floor((Nz+Mz)/2 + 0.5*(Nz/2-Mz/2-Npmlz)));
  getint("-icref",&icref,2);
  VecDuplicate(vR,&VecPt);
  MakeVecPt(VecPt,Nx,Ny,Nz,ixref,iyref,izref,icref-1);

  double *epsopt;
  FILE *ptf;
  epsopt = (double *) malloc(flagparams.DegFree*sizeof(double));
  ptf = fopen(flagparams.initialdatafile,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");
  int i;
  for (i=0;i<flagparams.DegFree;i++)
    { 
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);
  double *grad;
  grad = (double *) malloc(flagparams.DegFree*sizeof(double));
  /**********************************************************/

  char maxwellfile1[PETSC_MAX_PATH_LEN];
  Maxwell maxwell1;
  KSP ksp1;
  PC pc1;
  int its1=100;
  Vec refField1, refField1conj;
  PetscOptionsGetString(PETSC_NULL,"-maxwellfile1",maxwellfile1,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,maxwellfile1,"maxwellfile1","maxwellfile1");
  makemaxwell(maxwellfile1,flagparams,A,D,vR,weight,&maxwell1);
  setupKSP(PETSC_COMM_WORLD,&ksp1,&pc1,solver,0);
  makeRefField(maxwell1,flagparams,A,C,D,vR,ksp1,&its1, &refField1, &refField1conj, VecPt);
  MetaSurfGroup meta1={flagparams.Nx,flagparams.Ny,flagparams.Nz,flagparams.hxyz,epsSReal,epsFReal, maxwell1.omega, maxwell1.M, A, maxwell1.b, maxwell1.J, maxwell1.x, maxwell1.weightedJ, maxwell1.epspmlQ, maxwell1.epsbkg, maxwell1.epsdiff, maxwell1.epscoef, ksp1, &its1, metaphase, 1, refField1, refField1conj, VecPt, flagparams.outputbase, flagparams.filenameComm};

  /****TEST******/
  int printinitialvecs;
  getint("-printinitialvecs",&printinitialvecs,0);
  if(printinitialvecs){
    Vec tmpepsS, tmpepsF;
    MatCreateVecs(A,&tmpepsS,&tmpepsF);
    ArrayToVec(epsopt,tmpepsS);
    MatMult(A,tmpepsS,tmpepsF);
    VecPointwiseMult(tmpepsF,tmpepsF,maxwell1.epsdiff);
    VecAXPY(tmpepsF,1.0,maxwell1.epsbkg);
    OutputVec(PETSC_COMM_WORLD,tmpepsF,"epsinitial",".m");
    VecDestroy(&tmpepsS);
    VecDestroy(&tmpepsF);
    OutputVec(PETSC_COMM_WORLD,maxwell1.J,"J",".m");
    OutputVec(PETSC_COMM_WORLD,VecPt,"VecPt",".m");
    OutputVec(PETSC_COMM_WORLD,refField1,"refField",".m");
    double tmptest=metasurface(flagparams.DegFree,epsopt,grad,&meta1);
    OutputVec(PETSC_COMM_WORLD,meta1.x,"exmField",".m");
  }
  /***********************/
  int Job;
  getint("-Job",&Job,1);

  if(Job==0){

    /*---------Calculate the overlap and gradient--------*/
    int px, py, pz=0;
    double beta=0;
    double s1, ds, s2, epscen;

    getint("-px",&px,0);
    getint("-py",&py,0);
    getint("-pz",&pz,0);
    getreal("-s1",&s1,0);
    getreal("-s2",&s2,1);
    getreal("-ds",&ds,0.01);
  
    int posMj=(px*flagparams.My+ py)*(flagparams.Mzslab==0?flagparams.Mz:1) + pz*(1-flagparams.Mzslab);
    for (epscen=s1;epscen<s2;epscen+=ds)
      { 
	epsopt[posMj]=epscen; 
	beta = metasurface(flagparams.DegFree,epsopt,grad,&meta1);
	PetscPrintf(PETSC_COMM_WORLD,"epscen: %g objfunc: %g objfunc-grad: %g \n", epsopt[posMj], beta, grad[posMj]);
      }

  }

  if(Job==1){

    int DegFree=flagparams.DegFree;

    /*---------Optimization--------*/
    double frac;
    getreal("-penalfrac",&frac,1.0);

    double dummyvar;
    PetscOptionsGetReal(PETSC_NULL,"-dummyvar",&dummyvar,&flg);  MyCheckAndOutputDouble(flg,dummyvar,"dummyvar","Initial value of dummy variable t");
    int DegFreeAll=DegFree+1;
    double *epsoptAll;
    epsoptAll = (double *) malloc(DegFreeAll*sizeof(double));
    for (i=0;i<DegFree;i++){ epsoptAll[i]=epsopt[i]; }
    epsoptAll[DegFreeAll-1]=dummyvar;
  
    double mylb=0, myub=1.0, *lb=NULL, *ub=NULL;
    int maxeval, maxtime, mynloptalg;
    double maxf;
    nlopt_opt  opt;
    int mynloptlocalalg;
    nlopt_opt local_opt;
    nlopt_result result;

    PetscOptionsGetInt(PETSC_NULL,"-maxeval",&maxeval,&flg);  MyCheckAndOutputInt(flg,maxeval,"maxeval","max number of evaluation");
    PetscOptionsGetInt(PETSC_NULL,"-maxtime",&maxtime,&flg);  MyCheckAndOutputInt(flg,maxtime,"maxtime","max time of evaluation");
    PetscOptionsGetInt(PETSC_NULL,"-mynloptalg",&mynloptalg,&flg);  MyCheckAndOutputInt(flg,mynloptalg,"mynloptalg","The algorithm used ");
    PetscOptionsGetInt(PETSC_NULL,"-mynloptlocalalg",&mynloptlocalalg,&flg);  MyCheckAndOutputInt(flg,mynloptlocalalg,"mynloptlocalalg","The local optimization algorithm used ");

    lb = (double *) malloc(DegFreeAll*sizeof(double));
    ub = (double *) malloc(DegFreeAll*sizeof(double));
    int readlubsfromfile=flagparams.readlubsfromfile;
    if(!readlubsfromfile) {
      for(i=0;i<DegFree;i++)
	{
	  lb[i] = mylb;
	  ub[i] = myub;
	}
    }else {
      char lbfile[PETSC_MAX_PATH_LEN], ubfile[PETSC_MAX_PATH_LEN];
      PetscOptionsGetString(PETSC_NULL,"-lbfile",lbfile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,lbfile,"lbfile","Lower-bound file");
      PetscOptionsGetString(PETSC_NULL,"-ubfile",ubfile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,ubfile,"ubfile","Upper-bound file");
      ptf = fopen(lbfile,"r");
      for (i=0;i<DegFree;i++)
	{ 
	  fscanf(ptf,"%lf",&lb[i]);
	}
      fclose(ptf);

      ptf = fopen(ubfile,"r");
      for (i=0;i<DegFree;i++)
	{ 
	  fscanf(ptf,"%lf",&ub[i]);
	}
      fclose(ptf);

    }
    lb[DegFreeAll-1]=0.0000001;
    ub[DegFreeAll-1]=10;

    opt = nlopt_create(mynloptalg, DegFreeAll);
    nlopt_set_lower_bounds(opt,lb);
    nlopt_set_upper_bounds(opt,ub);
    nlopt_set_maxeval(opt,maxeval);
    nlopt_set_maxtime(opt,maxtime);
    if (mynloptalg==11) nlopt_set_vector_storage(opt,4000);
    if (mynloptlocalalg)
      { 
	PetscPrintf(PETSC_COMM_WORLD,"-----------Running with a local optimizer.---------\n"); 
	local_opt=nlopt_create(mynloptlocalalg,DegFreeAll);
	nlopt_set_ftol_rel(local_opt, 1e-14);
	nlopt_set_maxeval(local_opt,100000);
	nlopt_set_local_optimizer(opt,local_opt);
      }

    nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);

    if(frac<1.0) nlopt_add_inequality_constraint(opt,pfunc,&frac,1e-8);
    nlopt_set_max_objective(opt,minimaxobjfun,NULL);   

    result = nlopt_optimize(opt,epsoptAll,&maxf);

    PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);

    PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);

  }


  if(Job==2){

    int DegFree=flagparams.DegFree;

    /*---------Optimization--------*/
    double frac;
    getreal("-penalfrac",&frac,1.0);

    double mylb=0, myub=1.0, *lb=NULL, *ub=NULL;
    int maxeval, maxtime, mynloptalg;
    double maxf;
    nlopt_opt  opt;
    int mynloptlocalalg;
    nlopt_opt local_opt;
    nlopt_result result;

    PetscOptionsGetInt(PETSC_NULL,"-maxeval",&maxeval,&flg);  MyCheckAndOutputInt(flg,maxeval,"maxeval","max number of evaluation");
    PetscOptionsGetInt(PETSC_NULL,"-maxtime",&maxtime,&flg);  MyCheckAndOutputInt(flg,maxtime,"maxtime","max time of evaluation");
    PetscOptionsGetInt(PETSC_NULL,"-mynloptalg",&mynloptalg,&flg);  MyCheckAndOutputInt(flg,mynloptalg,"mynloptalg","The algorithm used ");
    PetscOptionsGetInt(PETSC_NULL,"-mynloptlocalalg",&mynloptlocalalg,&flg);  MyCheckAndOutputInt(flg,mynloptlocalalg,"mynloptlocalalg","The local optimization algorithm used ");

    lb = (double *) malloc(DegFree*sizeof(double));
    ub = (double *) malloc(DegFree*sizeof(double));
    int readlubsfromfile=flagparams.readlubsfromfile;
    if(!readlubsfromfile) {
      for(i=0;i<DegFree;i++)
	{
	  lb[i] = mylb;
	  ub[i] = myub;
	}
    }else {
      char lbfile[PETSC_MAX_PATH_LEN], ubfile[PETSC_MAX_PATH_LEN];
      PetscOptionsGetString(PETSC_NULL,"-lbfile",lbfile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,lbfile,"lbfile","Lower-bound file");
      PetscOptionsGetString(PETSC_NULL,"-ubfile",ubfile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,ubfile,"ubfile","Upper-bound file");
      ptf = fopen(lbfile,"r");
      for (i=0;i<DegFree;i++)
	{ 
	  fscanf(ptf,"%lf",&lb[i]);
	}
      fclose(ptf);

      ptf = fopen(ubfile,"r");
      for (i=0;i<DegFree;i++)
	{ 
	  fscanf(ptf,"%lf",&ub[i]);
	}
      fclose(ptf);

    }

    opt = nlopt_create(mynloptalg, DegFree);
    nlopt_set_lower_bounds(opt,lb);
    nlopt_set_upper_bounds(opt,ub);
    nlopt_set_maxeval(opt,maxeval);
    nlopt_set_maxtime(opt,maxtime);
    if (mynloptalg==11) nlopt_set_vector_storage(opt,4000);
    if (mynloptlocalalg)
      { 
	PetscPrintf(PETSC_COMM_WORLD,"-----------Running with a local optimizer.---------\n"); 
	local_opt=nlopt_create(mynloptlocalalg,DegFree);
	nlopt_set_ftol_rel(local_opt, 1e-14);
	nlopt_set_maxeval(local_opt,100000);
	nlopt_set_local_optimizer(opt,local_opt);
      }

    if(frac<1.0) nlopt_add_inequality_constraint(opt,pfunc,&frac,1e-8);
    nlopt_set_max_objective(opt,metasurface,&meta1);   

    result = nlopt_optimize(opt,epsopt,&maxf);

    PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);

    PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);

  }

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Barrier(PETSC_COMM_WORLD);

  
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}

PetscErrorCode makeRefField(Maxwell maxwell, Universals params, Mat A, Mat C, Mat D, Vec vR, KSP ksp, int *its, Vec *ref, Vec *refconj, Vec VecPt)
{

  Mat Mtmp;
  Vec x, xconj, mag, tmp;
  Vec epsS, epsF;
  VecDuplicate(vR,&x);
  VecDuplicate(vR,&xconj);
  VecDuplicate(vR,&mag);
  VecDuplicate(vR,&tmp);

  double amp;
  MatCreateVecs(A,&epsS,&epsF);
  VecSet(epsS,0.0);
  MatMult(A,epsS,epsF);
  MatDuplicate(maxwell.M,MAT_COPY_VALUES,&Mtmp);
  ModifyMatDiag(Mtmp,D,epsF,maxwell.epsdiff,maxwell.epsbkg,maxwell.epspmlQ,maxwell.omega,params.Nx,params.Ny,params.Nz);
  SolveMatrix(PETSC_COMM_WORLD,ksp,Mtmp,maxwell.b,x,its);
  MatMult(C,x,xconj);
  CmpVecProd(x,xconj,mag);
  VecPointwiseMult(mag,mag,vR);
  VecSqrtAbs(mag);
  VecPointwiseMult(mag,mag,VecPt);
  VecSum(mag,&amp);
  if(fabs(amp)<1e-6) PetscPrintf(PETSC_COMM_WORLD,"****WARNING: amplitude near zero; choose another ref point.\n");
  VecScale(x,1/amp);
  VecScale(xconj,1/amp);

  *ref=x;
  *refconj=xconj;

  VecDestroy(&mag);
  VecDestroy(&tmp);
  VecDestroy(&epsS);
  VecDestroy(&epsF);

  MatDestroy(&Mtmp);

  return 0;

}

double pfunc(int DegFree, double *epsopt, double *grad, void *data)
{
  int i;
  double sumeps;
  double max=DegFree/4;
  double *tmp  = (double *) data;
  double frac= *tmp;

  sumeps=0.0;
  for (i=0;i<DegFree-1;i++){
    sumeps+=fabs(epsopt[i]*(1-epsopt[i]));
    grad[i]=1-2*epsopt[i];
  }
  grad[DegFree-1]=0;

  PetscPrintf(PETSC_COMM_WORLD,"******the current binaryindex is %1.6e \n",sumeps);
  PetscPrintf(PETSC_COMM_WORLD,"******the current binaryexcess  is %1.6e \n",sumeps-frac*max);

  return sumeps - frac*max;
}

PetscErrorCode setupKSP(MPI_Comm comm, KSP *kspout, PC *pcout, int solver, int iteronly)
{
  PetscErrorCode ierr;
  KSP ksp;
  PC pc; 
  
  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  if (solver==0) {
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERPASTIX);CHKERRQ(ierr);
  }
  else if (solver==1){
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
  }
  else {
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU_DIST);CHKERRQ(ierr);
  }
  ierr = KSPSetTolerances(ksp,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxit);CHKERRQ(ierr);

  if (iteronly==1){
  ierr = KSPSetType(ksp, KSPLSQR);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPMonitorSet(ksp,KSPMonitorTrueResidualNorm,NULL,0);CHKERRQ(ierr);
  }

  ierr = PCSetFromOptions(pc);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  *kspout=ksp;
  *pcout=pc;

  PetscFunctionReturn(0);

}

