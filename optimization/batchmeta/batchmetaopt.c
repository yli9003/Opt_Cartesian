#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

double RRT, sigmax, sigmay, sigmaz;
int mma_verbose;
int initdirect, maxit;

int count=1;
Mat A,B,C,D;
Vec vR, weight, vgradlocal, epsFReal;
VecScatter scatter;
IS from, to;
char filenameComm[PETSC_MAX_PATH_LEN];

int pSIMP;
double bproj, etaproj;
Mat Hfilt;
KSP kspH;
int itsH;

double mintrans;

/*------------------------------------------------------*/

PetscErrorCode setupKSP(MPI_Comm comm, KSP *ksp, PC *pc, int solver, int iteronly);
double pfunc_constr(int DegFree, double *epsopt, double *grad, void *data);
double pfunc_noconstr(int DegFree, double *epsopt, double *grad, void *data);

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize ------*/
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;
  int iteronly=0;

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0) 
    mma_verbose=1;

/*****************************************************-------Set up the options parameters-------------********************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
  PetscBool flg;

  PetscOptionsGetInt(PETSC_NULL,"-initdirect",&initdirect,&flg);  MyCheckAndOutputInt(flg,initdirect,"initdirect","Initial number of direct LU solves");
  PetscOptionsGetInt(PETSC_NULL,"-maxit",&maxit,&flg);  MyCheckAndOutputInt(flg,maxit,"maxit","maximum krylov iterations before invoking direct solve");

  int Mx, My, Mzslab, Nx, Ny, Nz, Npmlx, Npmly, Npmlz, Nxyz, DegFree;
  double hx, hy, hz;
  PetscOptionsGetInt(PETSC_NULL,"-Mx",&Mx,&flg);  MyCheckAndOutputInt(flg,Mx,"Mx","Mx");
  PetscOptionsGetInt(PETSC_NULL,"-My",&My,&flg);  MyCheckAndOutputInt(flg,My,"My","My");

  PetscOptionsGetInt(PETSC_NULL,"-Mzslab",&Mzslab,&flg);  MyCheckAndOutputInt(flg,Mzslab,"Mzslab","Mzslab");

  PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg);  MyCheckAndOutputInt(flg,Nx,"Nx","Nx");
  PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg);  MyCheckAndOutputInt(flg,Ny,"Ny","Ny");
  PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg);  MyCheckAndOutputInt(flg,Nz,"Nz","Nz");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlx",&Npmlx,&flg);  MyCheckAndOutputInt(flg,Npmlx,"Npmlx","Npmlx");
  PetscOptionsGetInt(PETSC_NULL,"-Npmly",&Npmly,&flg);  MyCheckAndOutputInt(flg,Npmly,"Npmly","Npmly");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlz",&Npmlz,&flg);  MyCheckAndOutputInt(flg,Npmlz,"Npmlz","Npmlz");
  Nxyz=Nx*Ny*Nz;

  int LowerPMLx, LowerPMLy, LowerPMLz;
  getint("-LowerPMLx",&LowerPMLx,1);
  getint("-LowerPMLy",&LowerPMLy,1);
  getint("-LowerPMLz",&LowerPMLz,1);

  int Nxo,Nyo;
  getint("-Nxo",&Nxo,LowerPMLx*(Nx-Mx)/2);
  getint("-Nyo",&Nyo,LowerPMLy*(Ny-My)/2);

  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  getreal("-hy",&hy,hx);
  getreal("-hz",&hz,hx);

  RRT=1e-25;
  sigmax = pmlsigma(RRT,(double) Npmlx*hx);
  sigmay = pmlsigma(RRT,(double) Npmly*hy);
  sigmaz = pmlsigma(RRT,(double) Npmlz*hz);

  int BCPeriod;
  getint("-BCPeriod",&BCPeriod,2);
  int blochcondition;
  getint("-blochcondition",&blochcondition,0);

  int outputbase;
  getint("-outputbase",&outputbase,50);

  //same block source position for all currents
  double jlx, jux, jly, juy, jlz, juz;
  getreal("-jlx",&jlx,0);
  getreal("-jux",&jux,Nx*hx);
  getreal("-jly",&jly,0);
  getreal("-juy",&juy,Ny*hy);
  getreal("-jlz",&jlz,Npmlz*hz+1/5);
  getreal("-juz",&juz,jlz+hz);

  double Qabs;
  getreal("-Qabs",&Qabs,1e16);
  if (Qabs>1e15) Qabs=1.0/0.0;

  char initialdatafile[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-filenameprefix",filenameComm,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,filenameComm,"filenameprefix","Filename Prefix");
  PetscOptionsGetString(PETSC_NULL,"-initdatfile",initialdatafile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdatafile,"initialdatafile","Inputdata file");

  int solver;
  PetscOptionsGetInt(PETSC_NULL,"-solver",&solver,&flg);  MyCheckAndOutputInt(flg,solver,"solver","LU Direct solver choice (0 PASTIX, 1 MUMPS, 2 SUPERLU_DIST)");

  double sH, nR;
  int dimH;
  getint("-pSIMP",&pSIMP,1);
  getreal("-bproj",&bproj,0);
  getreal("-etaproj",&etaproj,0.5);
  getreal("-sH",&sH,-1);
  getreal("-nR",&nR,0);
  getint("-dimH",&dimH,1);

  int readlubsfromfile;
  getint("-readlubsfromfile",&readlubsfromfile,0);

  /**********************************************/
  int nlayers;
  int i;
  char tmpflg[PETSC_MAX_PATH_LEN];
  int Mz[nlayers], Nzo[nlayers];
  getint("-nlayers",&nlayers,1);
  DegFree=0;
  for(i=0;i<nlayers;i++){
    sprintf(tmpflg,"-Mz[%d]",i+1);
    getint(tmpflg,Mz+i,10);
    sprintf(tmpflg,"-Nzo[%d]",i+1);
    getint(tmpflg,Nzo+i,LowerPMLz*(Nz-Mz[i])/2);
    DegFree=DegFree+Mx*My*((Mzslab==0)?Mz[i]:1);
  }

  int nfreq;
  getint("-nfreq",&nfreq,1);

  int j;
  double tmp;
  double metaphase[nfreq];
  double betax[nfreq], betay[nfreq], betaz[nfreq];
  double epsdiff[nfreq][nlayers], epsbkg[nfreq][nlayers];
  double epssub[nfreq], epsair[nfreq], epsmid[nfreq];
  double epssubdiff[nfreq], epsairdiff[nfreq], epsmiddiff[nfreq];
  int bxl[nfreq], byl[nfreq], bzl[nfreq];
  int bxu[nfreq], byu[nfreq], bzu[nfreq];
  int Jdir[nfreq];
  double freq[nfreq], omega[nfreq], Jmag[nfreq];

  for(i=0;i<nfreq;i++){

    sprintf(tmpflg,"-metaphase[%d]",i+1);
    getreal(tmpflg,metaphase+i,0);

    sprintf(tmpflg,"-betax[%d]",i+1);
    getreal(tmpflg,betax+i,0);
    sprintf(tmpflg,"-betay[%d]",i+1);
    getreal(tmpflg,betay+i,0);
    sprintf(tmpflg,"-betaz[%d]",i+1);
    getreal(tmpflg,betaz+i,0);

    sprintf(tmpflg,"-epssub[%d]",i+1);
    getreal(tmpflg,epssub+1,2);
    sprintf(tmpflg,"-epsair[%d]",i+1);
    getreal(tmpflg,epsair+1,2);
    sprintf(tmpflg,"-epsmid[%d]",i+1);
    getreal(tmpflg,epsmid+1,2);
    sprintf(tmpflg,"-epssubdiff[%d]",i+1);
    getreal(tmpflg,epssubdiff+1,0);
    sprintf(tmpflg,"-epsairdiff[%d]",i+1);
    getreal(tmpflg,epsairdiff+1,0);
    sprintf(tmpflg,"-epsmiddiff[%d]",i+1);
    getreal(tmpflg,epsmiddiff+1,0);
    
    for(j=0;j<nlayers;j++){
      sprintf(tmpflg,"-epsdiff[%d][%d]",i+1,j+1);
      getreal(tmpflg,&tmp,3);
      epsdiff[i][j]=tmp;
      sprintf(tmpflg,"-epsbkg[%d][%d]",i+1,j+1);
      getreal(tmpflg,&tmp,3);
      epsbkg[i][j]=tmp;
    }

    sprintf(tmpflg,"-bxl[%d]",i+1);
    getint(tmpflg,bxl+i,0);
    sprintf(tmpflg,"-byl[%d]",i+1);
    getint(tmpflg,byl+i,0);
    sprintf(tmpflg,"-bzl[%d]",i+1);
    getint(tmpflg,bzl+i,0);
    sprintf(tmpflg,"-bxu[%d]",i+1);
    getint(tmpflg,bxu+i,0);
    sprintf(tmpflg,"-byu[%d]",i+1);
    getint(tmpflg,byu+i,0);
    sprintf(tmpflg,"-bzu[%d]",i+1);
    getint(tmpflg,bzu+i,0);

    sprintf(tmpflg,"-Jdir[%d]",i+1);
    getint(tmpflg,Jdir+i,2);
    sprintf(tmpflg,"-freq[%d]",i+1);
    getreal(tmpflg,freq+i,1);
    omega[i]=2*PI*freq[i];
    sprintf(tmpflg,"-Jmag[%d]",i+1);
    getreal(tmpflg,Jmag+i,1);

  }



/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
  /*--------Setup Helmholtz filter---------*/
  PC pcH;
  GetH1d(PETSC_COMM_WORLD,&Hfilt,DegFree,sH,nR,&kspH,&pcH);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the Hfilt DONE!--------\n ");CHKERRQ(ierr);
  /*--------Setup Helmholtz filter DONE---------*/

  /*------Set up the A, C, D matrices--------------*/
  layeredA(PETSC_COMM_WORLD,&A, Nx,Ny,Nz, nlayers,Nxo,Nyo,Nzo, Mx,My,Mz, Mzslab);

  int Arows, Acols;
  MatGetSize(A,&Arows,&Acols);
  PetscPrintf(PETSC_COMM_WORLD,"****Dimensions of A is %d by %d \n",Arows,Acols);

  CongMat(PETSC_COMM_WORLD, &C, 6*Nxyz);
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  /*-----Set up vR, weight, unitvecs------*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &vR);CHKERRQ(ierr);
  GetRealPartVec(vR,6*Nxyz);

  ierr = VecDuplicate(vR,&weight); CHKERRQ(ierr);
  GetWeightVecGeneralSym(weight,Nx,Ny,Nz,LowerPMLx,LowerPMLy,LowerPMLz); 

  Vec unitx,unity,unitz;
  ierr = VecDuplicate(vR,&unitx);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unity);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unitz);CHKERRQ(ierr);
  GetUnitVec(unitx,0,6*Nxyz);
  GetUnitVec(unity,1,6*Nxyz);
  GetUnitVec(unitz,2,6*Nxyz);

  Vec epsSReal, vgrad;
  ierr = MatCreateVecs(A,&epsSReal, &epsFReal); CHKERRQ(ierr);
  ierr = VecDuplicate(epsSReal, &vgrad); CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, DegFree, &vgradlocal); CHKERRQ(ierr);

  /*--------Create index sets for the vec scatter -------*/
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&to); CHKERRQ(ierr);

  /*----Set up the universal parts of M1, M2 and M3-------*/
  Mat M[nfreq];
  Vec muinvpml[nfreq];
  double *muinv;
  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  int tmpbx[2], tmpby[2], tmpbz[2];
  double tmpbeta[3];
  Vec epsVec[nfreq], epsmedium[nfreq], epspmlQ[nfreq], epscoef[nfreq];
  Vec epspml;
  VecDuplicate(vR,&epspml);
  Vec J[nfreq], b[nfreq], weightedJ[nfreq], x[nfreq];
  Vec pvec[nfreq], qvec[nfreq];
  KSP ksp[nfreq];
  PC pc[nfreq];
  int its[nfreq];
  Meta data[nfreq]; 
  for(i=0;i<nfreq;i++){
    MuinvPMLGeneral(PETSC_COMM_SELF, muinvpml+i, Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega[i],LowerPMLx,LowerPMLy,LowerPMLz);
    VecToArray(muinvpml[i],muinv,scatter,from,to,vgradlocal,6*Nxyz);
    tmpbx[0]=bxl[i];
    tmpbx[1]=bxu[i];
    tmpby[0]=byl[i];
    tmpby[1]=byu[i];
    tmpbz[0]=bzl[i];
    tmpbz[1]=bzu[i];
    tmpbeta[0]=betax[i];
    tmpbeta[1]=betay[i];
    tmpbeta[2]=betaz[i];
    MoperatorGeneralBloch(PETSC_COMM_WORLD, M+i, Nx,Ny,Nz, hx,hy,hz, tmpbx,tmpby,tmpbz, muinv, BCPeriod, tmpbeta);

    VecDuplicate(vR,epsVec+i);
    VecSet(epsVec[i],0.0);
    layeredepsdiff(epsVec[i],Nx,Ny,Nz, nlayers,Nzo,Mz, epsdiff[i], epssubdiff[i], epsairdiff[i], epsmiddiff[i]);
    VecDuplicate(vR,epsmedium+i);
    VecSet(epsmedium[i],0.0);
    layeredepsbkg(epsmedium[i],Nx,Ny,Nz, nlayers,Nzo,Mz, epsbkg[i], epssub[i], epsair[i], epsmid[i]);
    VecDuplicate(vR,epspmlQ+i);
    VecDuplicate(vR,epscoef+i);
    EpsPMLGeneral(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega[i], LowerPMLx,LowerPMLy,LowerPMLz);
    EpsCombine(D, weight, epspml, epspmlQ[i], epscoef[i], Qabs, omega[i], epsVec[i]);

    SourceBlock(PETSC_COMM_WORLD, J+i, Nx,Ny,Nz, hx,hy,hz, jlx,jux,jly,juy,jlz,juz, 1.0/hz,Jdir[i]);
    VecScale(J[i],Jmag[i]);
    VecDuplicate(vR,b+i);
    MatMult(D,J[i],b[i]);CHKERRQ(ierr);
    VecScale(b[i],omega[i]);
    VecDuplicate(vR,weightedJ+i);
    VecPointwiseMult(weightedJ[i],J[i],weight);
    VecDuplicate(vR,x+i);

    VecDuplicate(vR,pvec+i);
    VecDuplicate(vR,qvec+i);
    //TODO: MAKE PHASE VECTOR FOR LENS OR BEAM DEFLECTOR

    setupKSP(PETSC_COMM_WORLD,ksp+i,pc+i,solver,iteronly);
    its[i]=100;

    (data+i)->Nx=Nx;
    (data+i)->Ny=Ny;
    (data+i)->Nz=Nz;
    (data+i)->epsSReal=epsSReal;
    (data+i)->epsFReal=epsFReal;
    (data+i)->omega=omega[i];
    (data+i)->M=M[i];
    (data+i)->A=A;
    (data+i)->b=b[i];
    (data+i)->x=x[i];
    (data+i)->epspmlQ=epspmlQ[i];
    (data+i)->epsmedium=epsmedium[i];
    (data+i)->epsDiff=epsVec[i];
    (data+i)->epscoef=epscoef[i];
    (data+i)->ksp=ksp[i];
    (data+i)->its=its+i;
    (data+i)->pvec=pvec[i];
    (data+i)->qvec=qvec[i];
    (data+i)->outputbase=outputbase;

  }

  /*---------Setup the epsopt and grad arrays----------------*/
  double *epsopt;
  FILE *ptf;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  ptf = fopen(initialdatafile,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");
  for (i=0;i<DegFree;i++)
    { 
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);

  double *grad;
  grad = (double *) malloc(DegFree*sizeof(double));

  /*---------Setup Done!---------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Everything set up! Ready to calculate the overlap and gradient.--------\n ");CHKERRQ(ierr);

/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/

  int Job;
  getint("-Job",&Job,1);

  if(Job==1){

    /*---------Calculate the overlap and gradient--------*/
    double beta=0;
    double s1, ds, s2, epscen;
    int posMj;
    int ifreq;
    getint("-posMj",&posMj,0);
    getreal("-s1",&s1,0);
    getreal("-s2",&s2,1);
    getreal("-ds",&ds,0.01);
    getint("-ifreq",&ifreq,0);
    for (epscen=s1;epscen<s2;epscen+=ds)
      {
        epsopt[posMj]=epscen;
        beta = batchmeta(DegFree,epsopt,grad,data+ifreq);
        PetscPrintf(PETSC_COMM_WORLD,"epscen: %g objfunc: %g objfunc-grad: %g \n", epsopt[posMj], beta, grad[posMj]);
      }

  }

/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

/* ----------------------Destroy Vecs and Mats----------------------------*/ 
  MatDestroy(&A);
  MatDestroy(&C);
  MatDestroy(&D);

  MatDestroy(&Hfilt); 

  VecDestroy(&vR); 
  VecDestroy(&weight); 

  VecDestroy(&unitx); 
  VecDestroy(&unity); 
  VecDestroy(&unitz); 

  VecDestroy(&epspml); 

  VecDestroy(&epsSReal); 
  VecDestroy(&epsFReal); 
  VecDestroy(&vgrad);   
  VecDestroy(&vgradlocal); 

  KSPDestroy(&kspH);

  for(i=0;i<nfreq;i++){
    MatDestroy(M+i);
    VecDestroy(muinvpml+i); 
    VecDestroy(epsVec+i);
    VecDestroy(epsmedium+i);
    VecDestroy(epspmlQ+i);
    VecDestroy(epscoef+i);
    VecDestroy(x+i);
    VecDestroy(J+i);
    VecDestroy(weightedJ+i);
    VecDestroy(b+i);
    KSPDestroy(ksp+i);
  }


  ISDestroy(&from);
  ISDestroy(&to);

  free(muinv);
  free(epsopt);
  free(grad);

  /*------------ finalize the program -------------*/

  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Barrier(PETSC_COMM_WORLD);
  }
  
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}

double pfunc_constr(int DegFree, double *epsopt, double *grad, void *data)
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

double pfunc_noconstr(int DegFree, double *epsopt, double *grad, void *data)
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
