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

char filenameComm[PETSC_MAX_PATH_LEN];

/*------------------------------------------------------*/

PetscErrorCode makeRefField(Maxwell maxwell, Universals params, Mat A, Mat C, Mat D, Vec vR, KSP ksp, int *its, Vec *ref, Vec *refconj, Vec VecPT, int inverse);
PetscErrorCode setupKSP(MPI_Comm comm, KSP *ksp, PC *pc, int solver, int iteronly);
double pfunc(int DegFree, double *epsopt, double *grad, void *data);
double pfunc2(int DegFree, double *epsopt, double *grad, void *data);

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

  /**************/

  int nlayers;
  int *Mz, *Nzo;
  epsinfo eps1, eps2, eps3;
  int i;
  int DegFree=0;
  getint("-nlayers",&nlayers,1);
  Mz =(int *) malloc(nlayers*sizeof(int));
  Nzo=(int *) malloc(nlayers*sizeof(int));
  eps1.epsdiff=(double *) malloc(nlayers*sizeof(double));
  eps2.epsdiff=(double *) malloc(nlayers*sizeof(double));
  eps3.epsdiff=(double *) malloc(nlayers*sizeof(double));
  eps1.epsbkg=(double *) malloc(nlayers*sizeof(double));
  eps2.epsbkg=(double *) malloc(nlayers*sizeof(double));
  eps3.epsbkg=(double *) malloc(nlayers*sizeof(double));
  char tmpflg[PETSC_MAX_PATH_LEN];
  for (i=0;i<nlayers;i++) {
    sprintf(tmpflg,"-Mz[%d]",i+1);
    getint(tmpflg,Mz+i,10);
    sprintf(tmpflg,"-Nzo[%d]",i+1);
    getint(tmpflg,Nzo+i,LowerPMLz*(Nz-Mz[i])/2);
    DegFree=DegFree+Mx*My*((Mzslab==0)?Mz[i]:1);
  
    sprintf(tmpflg,"-eps1diff[%d]",i+1);
    getreal(tmpflg,eps1.epsdiff+i,3.6575);
    sprintf(tmpflg,"-eps1bkg[%d]",i+1);
    getreal(tmpflg,eps1.epsbkg+i,2.1025);

    sprintf(tmpflg,"-eps2diff[%d]",i+1);
    getreal(tmpflg,eps2.epsdiff+i,3.6575);
    sprintf(tmpflg,"-eps2bkg[%d]",i+1);
    getreal(tmpflg,eps2.epsbkg+i,2.1025);

    sprintf(tmpflg,"-eps3diff[%d]",i+1);
    getreal(tmpflg,eps3.epsdiff+i,3.6575);
    sprintf(tmpflg,"-eps3bkg[%d]",i+1);
    getreal(tmpflg,eps3.epsbkg+i,2.1025);

  }
  getreal("-eps1subdiff",&eps1.epssubdiff,0);
  getreal("-eps1airdiff",&eps1.epsairdiff,0);
  getreal("-eps1middiff",&eps1.epsmiddiff,0);
  getreal("-eps1sub",&eps1.epssub,2.1025);
  getreal("-eps1air",&eps1.epsair,1.0);
  getreal("-eps1mid",&eps1.epsmid,2.1025);

  getreal("-eps2subdiff",&eps2.epssubdiff,0);
  getreal("-eps2airdiff",&eps2.epsairdiff,0);
  getreal("-eps2middiff",&eps2.epsmiddiff,0);
  getreal("-eps2sub",&eps2.epssub,2.1025);
  getreal("-eps2air",&eps2.epsair,1.0);
  getreal("-eps2mid",&eps2.epsmid,2.1025);

  getreal("-eps3subdiff",&eps3.epssubdiff,0);
  getreal("-eps3airdiff",&eps3.epsairdiff,0);
  getreal("-eps3middiff",&eps3.epsmiddiff,0);
  getreal("-eps3sub",&eps3.epssub,2.1025);
  getreal("-eps3air",&eps3.epsair,1.0);
  getreal("-eps3mid",&eps3.epsmid,2.1025);

  /*************/

  Mat A;
  Vec vI, weight;
  Vec epsSReal;

  setupMatVecs(flagparams, &A, &C, &D, &vR, &vI, &weight, &epsSReal, &epsFReal);

  /***********/
  Mat Atmp;
  layeredA(PETSC_COMM_WORLD,&Atmp, Nx,Ny,Nz, nlayers,Nxo,Nyo,Nzo, Mx,My,Mz, Mzslab);
  MatCopy(A,Atmp,SAME_NONZERO_PATTERN);
  Vec epsI, epsII, epsIII;
  layeredepsdiff(epsI,   Nx,Ny,Nz, nlayers,Nzo,Mz, eps1.epsdiff, eps1.epssubdiff, eps1.epsairdiff, eps1.epsmiddiff);
  layeredepsdiff(epsII,  Nx,Ny,Nz, nlayers,Nzo,Mz, eps2.epsdiff, eps2.epssubdiff, eps2.epsairdiff, eps2.epsmiddiff);
  layeredepsdiff(epsIII, Nx,Ny,Nz, nlayers,Nzo,Mz, eps3.epsdiff, eps3.epssubdiff, eps3.epsairdiff, eps3.epsmiddiff);
  Vec epsmedium1, epsmedium2, epsmedium3;
  layeredepsbkg(epsmedium1, Nx,Ny,Nz, nlayers,Nzo,Mz, eps1.epsbkg, eps1.epssub, eps1.epsair, eps1.epsmid);
  layeredepsbkg(epsmedium2, Nx,Ny,Nz, nlayers,Nzo,Mz, eps2.epsbkg, eps2.epssub, eps2.epsair, eps2.epsmid);
  layeredepsbkg(epsmedium3, Nx,Ny,Nz, nlayers,Nzo,Mz, eps3.epsbkg, eps3.epssub, eps3.epsair, eps3.epsmid);
  Vec epspml1, epspml2, epspml3, epspmlQ1, epspmlQ2, epspmlQ3, epscoef1, epscoef2, epscoef3;
  ierr = VecDuplicate(vR,&epspml1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml3);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ3);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef3);CHKERRQ(ierr);
  EpsPMLGeneral(PETSC_COMM_WORLD, epspml1,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega1, LowerPMLx,LowerPMLy,LowerPMLz);
  EpsPMLGeneral(PETSC_COMM_WORLD, epspml2,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega2, LowerPMLx,LowerPMLy,LowerPMLz);
  EpsPMLGeneral(PETSC_COMM_WORLD, epspml3,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega3, LowerPMLx,LowerPMLy,LowerPMLz);
  EpsCombine(D, weight, epspml1, epspmlQ1, epscoef1, flagparams.Qabs, maxwell1.omega, epsI);
  EpsCombine(D, weight, epspml2, epspmlQ2, epscoef2, flagparams.Qabs, maxwell2.omega, epsII);
  EpsCombine(D, weight, epspml3, epspmlQ3, epscoef3, flagparams.Qabs, maxwell3.omega, epsIII);
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
  int inverse;
  getint("-inverse",&inverse,2);

  /******************/
  char maxwellfile1[PETSC_MAX_PATH_LEN];
  Maxwell maxwell1;
  PetscOptionsGetString(PETSC_NULL,"-maxwellfile1",maxwellfile1,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,maxwellfile1,"maxwellfile1","maxwellfile1");
  makemaxwell(maxwellfile1,flagparams,A,D,vR,weight,&maxwell1);
  VecCopy(epsmedium1,maxwell1.epsbkg);
  VecCopy(epsI,maxwell1.epsdiff);
  VecCopy(epspmlQ1,maxwell1.epspmlQ);
  VecCopy(epscoef1,maxwell1.epscoef);
  /******************/

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

  int optsuperpose;
  getint("-optsuperpose",&optsuperpose,1);

  KSP ksp1, refksp1;
  PC pc1, refpc1;
  int its1=100, refits1=100;
  Vec refField1, refField1conj;
  setupKSP(PETSC_COMM_WORLD,&ksp1,&pc1,solver,0);
  setupKSP(PETSC_COMM_WORLD,&refksp1,&refpc1,solver,0);
  makeRefField(maxwell1,flagparams,A,C,D,vR,refksp1,&refits1, &refField1, &refField1conj, VecPt, inverse);
  MetaSurfGroup meta1={flagparams.Nx,flagparams.Ny,flagparams.Nz,flagparams.hxyz,epsSReal,epsFReal, maxwell1.omega, maxwell1.M, A, maxwell1.b, maxwell1.J, maxwell1.x, maxwell1.weightedJ, maxwell1.epspmlQ, maxwell1.epsbkg, maxwell1.epsdiff, maxwell1.epscoef, ksp1, &its1, refksp1, &refits1, metaphase, optsuperpose, refField1, refField1conj, VecPt, flagparams.outputbase, flagparams.filenameComm};

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

    char maxwellfile2[PETSC_MAX_PATH_LEN];
    Maxwell maxwell2;
    KSP ksp2, refksp2;
    PC pc2, refpc2;
    int its2=100, refits2=100;
    Vec refField2, refField2conj;
    PetscOptionsGetString(PETSC_NULL,"-maxwellfile2",maxwellfile2,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,maxwellfile2,"maxwellfile2","maxwellfile2");
    makemaxwell(maxwellfile2,flagparams,A,D,vR,weight,&maxwell2);
    setupKSP(PETSC_COMM_WORLD,&ksp2,&pc2,solver,0);
    setupKSP(PETSC_COMM_WORLD,&refksp2,&refpc2,solver,0);
    makeRefField(maxwell2,flagparams,A,C,D,vR,refksp2,&refits2, &refField2, &refField2conj, VecPt, inverse);
    MetaSurfGroup meta2={flagparams.Nx,flagparams.Ny,flagparams.Nz,flagparams.hxyz,epsSReal,epsFReal, maxwell2.omega, maxwell2.M, A, maxwell2.b, maxwell2.J, maxwell2.x, maxwell2.weightedJ, maxwell2.epspmlQ, maxwell2.epsbkg, maxwell2.epsdiff, maxwell2.epscoef, ksp2, &its2, refksp2, &refits2, metaphase, optsuperpose, refField2, refField2conj, VecPt, flagparams.outputbase, flagparams.filenameComm};

    char maxwellfile3[PETSC_MAX_PATH_LEN];
    Maxwell maxwell3;
    KSP ksp3, refksp3;
    PC pc3, refpc3;
    int its3=100, refits3=100;
    Vec refField3, refField3conj;
    PetscOptionsGetString(PETSC_NULL,"-maxwellfile3",maxwellfile3,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,maxwellfile3,"maxwellfile3","maxwellfile3");
    makemaxwell(maxwellfile3,flagparams,A,D,vR,weight,&maxwell3);
    setupKSP(PETSC_COMM_WORLD,&ksp3,&pc3,solver,0);
    setupKSP(PETSC_COMM_WORLD,&refksp3,&refpc3,solver,0);
    makeRefField(maxwell3,flagparams,A,C,D,vR,refksp3,&refits3, &refField3, &refField3conj, VecPt, inverse);
    MetaSurfGroup meta3={flagparams.Nx,flagparams.Ny,flagparams.Nz,flagparams.hxyz,epsSReal,epsFReal, maxwell3.omega, maxwell3.M, A, maxwell3.b, maxwell3.J, maxwell3.x, maxwell3.weightedJ, maxwell3.epspmlQ, maxwell3.epsbkg, maxwell3.epsdiff, maxwell3.epscoef, ksp3, &its3, refksp3, &refits3, metaphase, optsuperpose, refField3, refField3conj, VecPt, flagparams.outputbase, flagparams.filenameComm};

    char maxwellfile4[PETSC_MAX_PATH_LEN];
    Maxwell maxwell4;
    KSP ksp4, refksp4;
    PC pc4, refpc4;
    int its4=100, refits4=100;
    Vec refField4, refField4conj;
    PetscOptionsGetString(PETSC_NULL,"-maxwellfile4",maxwellfile4,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,maxwellfile4,"maxwellfile4","maxwellfile4");
    makemaxwell(maxwellfile4,flagparams,A,D,vR,weight,&maxwell4);
    setupKSP(PETSC_COMM_WORLD,&ksp4,&pc4,solver,0);
    setupKSP(PETSC_COMM_WORLD,&refksp4,&refpc4,solver,0);
    makeRefField(maxwell4,flagparams,A,C,D,vR,refksp4,&refits4, &refField4, &refField4conj, VecPt, inverse);
    MetaSurfGroup meta4={flagparams.Nx,flagparams.Ny,flagparams.Nz,flagparams.hxyz,epsSReal,epsFReal, maxwell4.omega, maxwell4.M, A, maxwell4.b, maxwell4.J, maxwell4.x, maxwell4.weightedJ, maxwell4.epspmlQ, maxwell4.epsbkg, maxwell4.epsdiff, maxwell4.epscoef, ksp4, &its4, refksp4, &refits4, metaphase, optsuperpose, refField4, refField4conj, VecPt, flagparams.outputbase, flagparams.filenameComm};

    char maxwellfile5[PETSC_MAX_PATH_LEN];
    Maxwell maxwell5;
    KSP ksp5, refksp5;
    PC pc5, refpc5;
    int its5=100, refits5=100;
    Vec refField5, refField5conj;
    PetscOptionsGetString(PETSC_NULL,"-maxwellfile5",maxwellfile5,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,maxwellfile5,"maxwellfile5","maxwellfile5");
    makemaxwell(maxwellfile5,flagparams,A,D,vR,weight,&maxwell5);
    setupKSP(PETSC_COMM_WORLD,&ksp5,&pc5,solver,0);
    setupKSP(PETSC_COMM_WORLD,&refksp5,&refpc5,solver,0);
    makeRefField(maxwell5,flagparams,A,C,D,vR,refksp5,&refits5, &refField5, &refField5conj, VecPt, inverse);
    MetaSurfGroup meta5={flagparams.Nx,flagparams.Ny,flagparams.Nz,flagparams.hxyz,epsSReal,epsFReal, maxwell5.omega, maxwell5.M, A, maxwell5.b, maxwell5.J, maxwell5.x, maxwell5.weightedJ, maxwell5.epspmlQ, maxwell5.epsbkg, maxwell5.epsdiff, maxwell5.epscoef, ksp5, &its5, refksp5, &refits5, metaphase, optsuperpose, refField5, refField5conj, VecPt, flagparams.outputbase, flagparams.filenameComm};

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
    //make sure that the pixels near boundaries are fixed
    int fixedendpts;
    getint("-fixedendpts",&fixedendpts,5);
    for(i=0;i<fixedendpts;i++){
      lb[i]=0;
      lb[DegFreeAll-i-2]=0;
      ub[i]=0;
      ub[DegFreeAll-i-2]=0;
    }
    //make sure that the pixels near boundaries are fixed

    lb[DegFreeAll-1]=-3;
    ub[DegFreeAll-1]=3;

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

    int nummodes;
    getint("-nummodes",&nummodes,2);
    if(nummodes==1){
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);
    }else if(nummodes==2){
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta2,1e-8);
    }else if(nummodes==3){
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta2,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta3,1e-8);
    }else if(nummodes==4){
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta2,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta3,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta4,1e-8);
    }else{
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta1,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta2,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta3,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta4,1e-8);
      nlopt_add_inequality_constraint(opt,metasurfaceminimax,&meta5,1e-8);
    };
     

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
    //make sure that the pixels near the boundary are not considered as DOFs
    lb[0]=0;
    lb[1]=0;
    lb[DegFree-1]=0;
    lb[DegFree-2]=0;
    ub[0]=0;
    ub[1]=0;
    ub[DegFree-1]=0;
    ub[DegFree-2]=0;
    
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

    if(frac<1.0) nlopt_add_inequality_constraint(opt,pfunc2,&frac,1e-8);
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

PetscErrorCode makeRefField(Maxwell maxwell, Universals params, Mat A, Mat C, Mat D, Vec vR, KSP ksp, int *its, Vec *ref, Vec *refconj, Vec VecPt, int inverse)
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
  VecPointwiseMult(mag,mag,VecPt);
  VecSqrtAbs(mag);
  VecSum(mag,&amp);

  if(fabs(amp)<1e-6) PetscPrintf(PETSC_COMM_WORLD,"****WARNING: amplitude near zero; choose another ref point.\n");
  if(inverse==0){
    VecScale(x,1/amp);
    VecScale(xconj,1/amp);
  }else if(inverse==1){
    VecScale(x,1/(amp*amp));
    VecScale(xconj,1/(amp*amp));
  }else{
    VecScale(x,1.0);
    VecScale(xconj,1.0);
  };

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

double pfunc2(int DegFree, double *epsopt, double *grad, void *data)
{
  int i;
  double sumeps;
  double max=DegFree/4;
  double *tmp  = (double *) data;
  double frac= *tmp;

  sumeps=0.0;
  for (i=0;i<DegFree;i++){
    sumeps+=fabs(epsopt[i]*(1-epsopt[i]));
    grad[i]=1-2*epsopt[i];
  }

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