#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

int mma_verbose;

int Nxyz, count;
double hxyz;
Mat B, C, D;
Vec vR, weight;
Vec vgradlocal;
Vec epsSReal, epsFReal;
IS from, to;
VecScatter scatter;
char filenameComm[PETSC_MAX_PATH_LEN];
int outputbase;

int pSIMP;
double bproj, etaproj;
Mat Hfilt;
KSP kspH;
int itsH;

int maxit=15;

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

/*****************************************************-------Set up the options parameters-------------********************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
  PetscBool flg;
  
  int initdirect;
  PetscOptionsGetInt(PETSC_NULL,"-initdirect",&initdirect,&flg);  MyCheckAndOutputInt(flg,initdirect,"initdirect","Initial number of direct LU solves");
  PetscOptionsGetInt(PETSC_NULL,"-maxit",&maxit,&flg);  MyCheckAndOutputInt(flg,maxit,"maxit","maximum krylov iterations before invoking direct solve");

  int iteronly;
  getint("-iteronly",&iteronly,0);

  int blochcondition;
  double beta[3]={0,0,0};
  getint("-blochcondition",&blochcondition,0);
  if(blochcondition){
    PetscOptionsGetReal(PETSC_NULL,"-betax",beta  ,&flg); MyCheckAndOutputDouble(flg,beta[0],"betax","Bloch vector component betax");
    PetscOptionsGetReal(PETSC_NULL,"-betay",beta+1,&flg); MyCheckAndOutputDouble(flg,beta[1],"betay","Bloch vector component betay");
    PetscOptionsGetReal(PETSC_NULL,"-betaz",beta+2,&flg); MyCheckAndOutputDouble(flg,beta[2],"betaz","Bloch vector component betaz");
  }
  
  int Mx, My, Mz, Npmlx, Npmly, Npmlz, Nx, Ny, Nz, Mzslab;
  int anisotropicDOF=0;
  double hx, hy, hz;
  int DegFree;

  PetscOptionsGetInt(PETSC_NULL,"-Mx",&Mx,&flg);  MyCheckAndOutputInt(flg,Mx,"Mx","Mx");
  PetscOptionsGetInt(PETSC_NULL,"-My",&My,&flg);  MyCheckAndOutputInt(flg,My,"My","My");
  PetscOptionsGetInt(PETSC_NULL,"-Mz",&Mz,&flg);  MyCheckAndOutputInt(flg,Mz,"Mz","Mz");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlx",&Npmlx,&flg);  MyCheckAndOutputInt(flg,Npmlx,"Npmlx","Npmlx");
  PetscOptionsGetInt(PETSC_NULL,"-Npmly",&Npmly,&flg);  MyCheckAndOutputInt(flg,Npmly,"Npmly","Npmly");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlz",&Npmlz,&flg);  MyCheckAndOutputInt(flg,Npmlz,"Npmlz","Npmlz");

  PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg);  MyCheckAndOutputInt(flg,Nx,"Nx","Nx");
  PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg);  MyCheckAndOutputInt(flg,Ny,"Ny","Ny");
  PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg);  MyCheckAndOutputInt(flg,Nz,"Nz","Nz");
  PetscOptionsGetInt(PETSC_NULL,"-Mzslab",&Mzslab,&flg);  MyCheckAndOutputInt(flg,Mzslab,"Mzslab","Mzslab");

  int Mxyz;
  Mxyz=Mx*My*Mz, Nxyz=Nx*Ny*Nz;
  DegFree = (anisotropicDOF ? 3 : 1 )*Mx*My*((Mzslab==0)?Mz:1);
  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  getreal("-hy",&hy,hx);
  getreal("-hz",&hz,hx);
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;

  int BCPeriod, LowerPML;
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod");
  PetscOptionsGetInt(PETSC_NULL,"-LowerPML",&LowerPML,&flg);  MyCheckAndOutputInt(flg,LowerPML,"LowerPML","LowerPML");

  int bx[2], by[2], bz[2];
  PetscOptionsGetInt(PETSC_NULL,"-bxl",bx,&flg);    MyCheckAndOutputInt(flg,bx[0],"bxl","BC at x lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-bxu",bx+1,&flg);  MyCheckAndOutputInt(flg,bx[1],"bxu","BC at x upper for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-byl",by,&flg);    MyCheckAndOutputInt(flg,by[0],"byl","BC at y lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-byu",by+1,&flg);  MyCheckAndOutputInt(flg,by[1],"byu","BC at y upper for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-bzl",bz,&flg);    MyCheckAndOutputInt(flg,bz[0],"bzl","BC at z lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-bzu",bz+1,&flg);  MyCheckAndOutputInt(flg,bz[1],"bzu","BC at z upper for mode 1");

  int PrintEpsC;
  PetscOptionsGetInt(PETSC_NULL,"-outputbase",&outputbase,&flg); MyCheckAndOutputInt(flg,outputbase,"outputbase","outputbase");
  getint("-PrintEpsC",&PrintEpsC,outputbase);

  double freq, omega;
  getreal("-freq",&freq,1.0);
  omega=2.0*PI*freq;

  double epsx, epsy, epsz, epsair, epssub;
  PetscOptionsGetReal(PETSC_NULL,"-epsx",&epsx,&flg); MyCheckAndOutputDouble(flg,epsx,"epsx","epsx");
  getreal("-epsy",&epsy,epsx);
  getreal("-epsz",&epsz,epsx);
  getreal("-epsmed",&epsair,1.0);
  getreal("-epssub",&epssub,epsair);

  double RRT, sigmax, sigmay, sigmaz;
  RRT=1e-25;
  sigmax = pmlsigma(RRT,(double) Npmlx*hx);
  sigmay = pmlsigma(RRT,(double) Npmly*hy);
  sigmaz = pmlsigma(RRT,(double) Npmlz*hz);

  PetscPrintf(PETSC_COMM_WORLD,"sigma, omega, DegFree: %g, %g, %g, %g, %d \n", sigmax, sigmay, sigmaz, omega, DegFree);

  double Qabs;
  PetscOptionsGetReal(PETSC_NULL,"-Qabs",&Qabs,&flg);  MyCheckAndOutputDouble(flg,Qabs,"Qabs","Qabs");
  if (Qabs>1e15) Qabs=1.0/0.0;

  double JmagL, JmagR;
  getreal("-JmagL",&JmagL,1.0);
  getreal("-JmagR",&JmagR,1.0);

  char initialdatafile[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-filenameprefix",filenameComm,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,filenameComm,"filenameprefix","Filename Prefix");
  PetscOptionsGetString(PETSC_NULL,"-initdatfile",initialdatafile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdatafile,"initialdatafile","Inputdata file");

  int solver;
  PetscOptionsGetInt(PETSC_NULL,"-solver",&solver,&flg);  MyCheckAndOutputInt(flg,solver,"solver","LU Direct solver choice (0 PASTIX, 1 MUMPS, 2 SUPERLU_DIST)");

  getint("-pSIMP",&pSIMP,1);
  getreal("-bproj",&bproj,0);
  getreal("-etaproj",&etaproj,0.5);

  int readlubsfromfile;
  getint("-readlubsfromfile",&readlubsfromfile,0);

  double sH, nR;
  int dimH;
  getreal("-sH",&sH,-1);
  getreal("-nR",&nR,0);
  getint("-dimH",&dimH,1);

  double chiralweight;
  getreal("-chiralweight",&chiralweight,1.0);

  int fomopt;
  getint("-fomopt",&fomopt,2);

/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/


  /*------Set up the A, B, C, D matrices--------------*/
  Mat A;
  myinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, LowerPML*floor((Nx-Mx)/2),LowerPML*floor((Ny-My)/2),LowerPML*floor((Nz-Mz)/2), Mx,My,Mz,Mzslab, anisotropicDOF);
  GetDotMat(PETSC_COMM_WORLD, &B, Nx, Ny, Nz);
  CongMat(PETSC_COMM_WORLD, &C, 6*Nxyz);
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  /*-----Set up vR------*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &vR);CHKERRQ(ierr);
  GetRealPartVec(vR,6*Nxyz);
  ierr = PetscObjectSetName((PetscObject) vR, "vR");CHKERRQ(ierr);

  /*-----Set up the vectors-------*/
  Vec unitx, unity, unitz, epsDiff, muinvpml, epspml, epspmlQ, epscoef, epsmedium, xL, bL, JL, weightedJL, xR, bR, JR, weightedJR, vgrad;
  ierr = VecDuplicate(vR,&weight); CHKERRQ(ierr);

  ierr = VecDuplicate(vR,&unitx); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unity); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unitz); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsDiff); CHKERRQ(ierr);

  ierr = VecDuplicate(vR,&muinvpml); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsmedium); CHKERRQ(ierr);

  ierr = VecDuplicate(vR,&xL); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&bL);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&JL); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&weightedJL); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&xR); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&bR);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&JR); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&weightedJR); CHKERRQ(ierr);


  /*-----------------------*/
  ierr = PetscObjectSetName((PetscObject) weight, "weight");CHKERRQ(ierr);
  if(LowerPML==0)
    GetWeightVec(weight, Nx, Ny,Nz); 
  else
    VecSet(weight,1.0);
  
  /*----Set up the universal parts of M-------*/
  Mat M;
  double *muinv;
  int add=0;
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega,LowerPML);
  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  AddMuAbsorption(muinv,muinvpml,Qabs,add);

  if(blochcondition){
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M, Nx,Ny,Nz, hx,hy,hz, bx,by,bz, muinv, BCPeriod, beta);
  }else{
    MoperatorGeneral(PETSC_COMM_WORLD, &M, Nx,Ny,Nz, hx,hy,hz, bx,by,bz, muinv, BCPeriod);
  }
  ierr = PetscObjectSetName((PetscObject) M, "M"); CHKERRQ(ierr);

  /*----Set up the epsilon PML vectors--------*/
  GetUnitVec(unitx,0,6*Nxyz);
  GetUnitVec(unity,1,6*Nxyz);
  GetUnitVec(unitz,2,6*Nxyz);

  VecSet(epsDiff,0.0);
  VecAXPY(epsDiff,epsx,unitx);
  VecAXPY(epsDiff,epsy,unity);
  VecAXPY(epsDiff,epsz,unitz);

  EpsPMLFull(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);
  EpsCombine(D, weight, epspml, epspmlQ, epscoef, Qabs, omega, epsDiff);


  /*-----Set up epsmedium, epsSReal, epsFReal, epsC, epsCi, epsP, vgrad, vgradlocal ------*/
  GetMediumVec(epsmedium,Nz,Mz,epsair,epssub);

  ierr = MatCreateVecs(A,&epsSReal, &epsFReal); CHKERRQ(ierr);
  ierr = VecDuplicate(epsSReal, &vgrad); CHKERRQ(ierr);
  
  ierr = PetscObjectSetName((PetscObject) epsmedium,  "epsmedium");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsFReal, "epsFReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vgrad, "vgrad");CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, DegFree, &vgradlocal); CHKERRQ(ierr);

  /*---------Set up JL, JR, bL, bR, and weightedJL, weightedJR-------------*/
  char inputsrcL[PETSC_MAX_PATH_LEN], inputsrcR[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-inputsrcL",inputsrcL,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrcL,"inputsrcL","Input source current (left)");
  PetscOptionsGetString(PETSC_NULL,"-inputsrcR",inputsrcR,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrcR,"inputsrcR","Input source current (right)");
  
  double *jL,*jR;
  FILE *jLptf,*jRptf;
  jL = (double *) malloc(6*Nxyz*sizeof(double));
  jR = (double *) malloc(6*Nxyz*sizeof(double));
  jLptf = fopen(inputsrcL,"r");
  jRptf = fopen(inputsrcR,"r");
  int inJi;
  for (inJi=0;inJi<6*Nxyz;inJi++)
    { 
      fscanf(jLptf,"%lf",&jL[inJi]);
      fscanf(jRptf,"%lf",&jR[inJi]);
    }
  fclose(jLptf);
  fclose(jRptf);

  ArrayToVec(jL,JL);
  ArrayToVec(jR,JR);
  
  free(jL);
  free(jR);

  VecScale(JL,JmagL);
  VecScale(JR,JmagR);

  ierr = MatMult(D,JL,bL);CHKERRQ(ierr);
  VecScale(bL,omega);
  ierr = VecPointwiseMult(weightedJL,JL,weight);

  ierr = MatMult(D,JR,bR);CHKERRQ(ierr);
  VecScale(bR,omega);
  ierr = VecPointwiseMult(weightedJR,JR,weight);

  /*--------Create index sets for the vec scatter -------*/
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&to); CHKERRQ(ierr);

  /*--------Setup the KSP variables ---------------*/
  KSP ksp;
  PC pc;
  int its=100;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the KSP variables.--------\n ");CHKERRQ(ierr);
  setupKSP(PETSC_COMM_WORLD,&ksp,&pc,solver,iteronly);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the KSP variables DONE!--------\n ");CHKERRQ(ierr);
  /*--------Setup the KSP variables DONE. ---------------*/

  /*--------Setup Helmholtz filter---------*/
  PC pcH;
  GetH(PETSC_COMM_WORLD,&Hfilt,Mx,My,(Mzslab==0)?Mz:1,sH,nR,dimH,&kspH,&pcH);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the Hfilt DONE!--------\n ");CHKERRQ(ierr);
  /*--------Setup Helmholtz filter DONE---------*/


  /*---------Setup the epsopt and grad arrays----------------*/
  double tstart;
  getreal("-tstart",&tstart,0);
  int DegFreeAll = DegFree + (chiralweight > 0);
  double *epsoptAll;
  FILE *ptf;
  epsoptAll = (double *) malloc(DegFreeAll*sizeof(double));
  ptf = fopen(initialdatafile,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");
  int i;
  for (i=0;i<DegFree;i++)
    { 
      fscanf(ptf,"%lf",&epsoptAll[i]);
    }
  fclose(ptf);
  if(chiralweight) epsoptAll[DegFreeAll-1]=tstart;

  double *gradAll;
  gradAll = (double *) malloc(DegFreeAll*sizeof(double));

  /*---------Setup Done!---------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Everything set up! Ready to calculate the overlap and gradient.--------\n ");CHKERRQ(ierr);


/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/

  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------We proceed to Job.--------\n ");CHKERRQ(ierr);

  int Job;
  PetscOptionsGetInt(PETSC_NULL,"-Job",&Job,&flg); MyCheckAndOutputInt(flg,Job,"Job","Job (1 gradient check, 2 chiralopt, 3 maximin)"); 
  
  ChiraldataGroup chiraldata = {omega, M, A, xL, xR, bL, bR, weightedJL, weightedJR, epspmlQ, epsmedium, epsDiff,  &its, epscoef, vgrad, ksp, chiralweight, fomopt};
  LDOSdataGroup ldosLdata = {omega, M, A, xL, bL, weightedJL, epspmlQ, epsmedium, epsDiff,  &its, epscoef, vgrad, ksp};
  LDOSdataGroup ldosRdata = {omega, M, A, xR, bR, weightedJR, epspmlQ, epsmedium, epsDiff,  &its, epscoef, vgrad, ksp};

  /************** set up vectors for possible maximization of x and y polarizations ****************/ 
  Vec Ex, Ey, Jx, Jy, Bx, By, weightedJx, weightedJy;
  VecDuplicate(vR,&Ex);
  VecDuplicate(vR,&Ey);
  VecDuplicate(vR,&Jx);
  VecDuplicate(vR,&Jy);
  VecDuplicate(vR,&Bx);
  VecDuplicate(vR,&By);
  VecDuplicate(vR,&weightedJx);
  VecDuplicate(vR,&weightedJy);

  VecPointwiseMult(Jx,JL,unitx);
  MatMult(D,Jx,Bx);
  VecScale(Bx,omega);
  VecPointwiseMult(weightedJx,Jx,weight);

  MatMult(D,JL,Jy);
  VecScale(Jy,-1.0);
  VecPointwiseMult(Jy,Jy,unity);
  MatMult(D,Jy,By);
  VecScale(By,omega);
  VecPointwiseMult(weightedJy,Jy,weight);

  LDOSdataGroup ldosxdata = {omega, M, A, Ex, Bx, weightedJx, epspmlQ, epsmedium, epsDiff,  &its, epscoef, vgrad, ksp};
  LDOSdataGroup ldosydata = {omega, M, A, Ey, By, weightedJy, epspmlQ, epsmedium, epsDiff,  &its, epscoef, vgrad, ksp};
  /************** DONE: set up vectors for possible maximization of x and y polarizations ****************/     

if (Job==1){

  /*---------Calculate the overlap and gradient--------*/
  int px, py, pz=0;
  double beta=0;
  double s1, ds, s2, epscen;
  
  PetscOptionsGetInt(PETSC_NULL,"-px",&px,&flg);  MyCheckAndOutputInt(flg,px,"px","px");
  PetscOptionsGetInt(PETSC_NULL,"-py",&py,&flg);  MyCheckAndOutputInt(flg,py,"py","py");
  PetscOptionsGetReal(PETSC_NULL,"-s1",&s1,&flg); MyCheckAndOutputDouble(flg,s1,"s1","s1");
  PetscOptionsGetReal(PETSC_NULL,"-s2",&s2,&flg); MyCheckAndOutputDouble(flg,s2,"s2","s2");
  PetscOptionsGetReal(PETSC_NULL,"-ds",&ds,&flg); MyCheckAndOutputDouble(flg,ds,"ds","ds");

  int posMj=(px*My+ py)*Mz + pz;
  for (epscen=s1;epscen<s2;epscen+=ds)
  { 
      epsoptAll[posMj]=epscen; 
      beta = ldoskdiff(DegFreeAll,epsoptAll,gradAll,&chiraldata);
      
      PetscPrintf(PETSC_COMM_WORLD,"epscen: %g objfunc: %g objfunc-grad: %g \n", epsoptAll[posMj], beta, gradAll[posMj]);
  }
 }

 if (Job==2){

   PetscPrintf(PETSC_COMM_WORLD,"*******IMPORTANT: Optimizing chiral difference only. Make sure chiralweight = 0.****************\n");

   /*---------Optimization--------*/
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
   if(chiralweight) lb[DegFreeAll-1]=0.0000000001;
   if(chiralweight) ub[DegFreeAll-1]=1.0/0.0;

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

   double frac;
   getreal("-frac",&frac,1.0);
   if(frac<1.0) nlopt_add_inequality_constraint(opt,pfunc,&frac,1e-8);

   if(chiralweight){
     nlopt_add_inequality_constraint(opt,ldoskdiff,&chiraldata,1e-8);
     nlopt_set_max_objective(opt,maxminobjfun,NULL);
   }else{
     nlopt_set_max_objective(opt,ldoskdiff,&chiraldata);
   }

   PetscPrintf(PETSC_COMM_WORLD,"---------Optimization about to occur. \n");
   result = nlopt_optimize(opt,epsoptAll,&maxf);    //********************optimization here!********************//

   if(result<0){
     PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);
   }else{   
     PetscPrintf(PETSC_COMM_WORLD,"found extremum  %0.16e\n", maxf);
   }

   PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);

   free(lb);
   free(ub);
   nlopt_destroy(opt);

   int rankA;
   MPI_Comm_rank(PETSC_COMM_WORLD, &rankA);

   if(rankA==0)
     {
       ptf = fopen(strcat(filenameComm,"epsopt.txt"),"w");
       for (i=0;i<DegFree;i++)
	 fprintf(ptf,"%0.16e \n",epsoptAll[i]);
       fclose(ptf);
     }

 }

 if (Job==3){

   /*---------Optimization--------*/  
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
   lb[DegFreeAll-1]=0.00000001;
   ub[DegFreeAll-1]=1.0/0.0;

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

   int chopt, JLopt, JRopt, Jxopt, Jyopt;
   getint("-chopt",&chopt,0);
   getint("-JLopt",&JLopt,1);
   getint("-JRopt",&JRopt,1);
   getint("-Jxopt",&Jxopt,0);
   getint("-Jyopt",&Jyopt,0);

   if(chopt) nlopt_add_inequality_constraint(opt,ldoskdiff,&chiraldata,1e-8);
   if(JLopt) nlopt_add_inequality_constraint(opt,ldoskmaxconstraint,&ldosLdata,1e-8);
   if(JRopt) nlopt_add_inequality_constraint(opt,ldoskminconstraint,&ldosRdata,1e-8);
   if(Jxopt) nlopt_add_inequality_constraint(opt,ldoskmaxconstraint,&ldosxdata,1e-8);
   if(Jyopt) nlopt_add_inequality_constraint(opt,ldoskmaxconstraint,&ldosydata,1e-8);

   nlopt_set_max_objective(opt,maxminobjfun,NULL);   

   result = nlopt_optimize(opt,epsoptAll,&maxf);

   PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);

   PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);

   int rankA;
   MPI_Comm_rank(PETSC_COMM_WORLD, &rankA);

   if(rankA==0)
     {
       ptf = fopen(strcat(filenameComm,"epsopt.txt"),"w");
       for (i=0;i<DegFree;i++)
	 fprintf(ptf,"%0.16e \n",epsoptAll[i]);
       fclose(ptf);
     }

   free(lb);
   free(ub);
   nlopt_destroy(opt);

 }


 
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

/* ----------------------Destroy Vecs and Mats----------------------------*/ 
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = MatDestroy(&B); CHKERRQ(ierr);
  ierr = MatDestroy(&C); CHKERRQ(ierr);
  ierr = MatDestroy(&D); CHKERRQ(ierr);
  ierr = MatDestroy(&M); CHKERRQ(ierr);  
  ierr = MatDestroy(&Hfilt); CHKERRQ(ierr);

  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&weight); CHKERRQ(ierr);

  ierr = VecDestroy(&unitx); CHKERRQ(ierr);
  ierr = VecDestroy(&unity); CHKERRQ(ierr);
  ierr = VecDestroy(&unitz); CHKERRQ(ierr);
  ierr = VecDestroy(&epsDiff); CHKERRQ(ierr);

  ierr = VecDestroy(&muinvpml); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium); CHKERRQ(ierr);
  ierr = VecDestroy(&epsSReal); CHKERRQ(ierr);
  ierr = VecDestroy(&epsFReal); CHKERRQ(ierr);
  ierr = VecDestroy(&vgrad); CHKERRQ(ierr);  
  ierr = VecDestroy(&vgradlocal); CHKERRQ(ierr);

  ierr = VecDestroy(&xL); CHKERRQ(ierr);
  ierr = VecDestroy(&bL);CHKERRQ(ierr);
  ierr = VecDestroy(&JL); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJL); CHKERRQ(ierr);
  ierr = VecDestroy(&xR); CHKERRQ(ierr);
  ierr = VecDestroy(&bR);CHKERRQ(ierr);
  ierr = VecDestroy(&JR); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJR); CHKERRQ(ierr);

  ierr = VecDestroy(&Ex); CHKERRQ(ierr);
  ierr = VecDestroy(&Bx);CHKERRQ(ierr);
  ierr = VecDestroy(&Jx); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJx); CHKERRQ(ierr);
  ierr = VecDestroy(&Ey); CHKERRQ(ierr);
  ierr = VecDestroy(&By);CHKERRQ(ierr);
  ierr = VecDestroy(&Jy); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJy); CHKERRQ(ierr);




  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspH);CHKERRQ(ierr);

  ISDestroy(&from);
  ISDestroy(&to);

  free(muinv);
  free(epsoptAll);
  free(gradAll);

  /*------------ finalize the program -------------*/

  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Barrier(PETSC_COMM_WORLD);
  }
  
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
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

  PetscPrintf(PETSC_COMM_WORLD,"******the current binaryindex is %1.6e \n",sumeps);
  PetscPrintf(PETSC_COMM_WORLD,"******the current binaryexcess  is %1.6e \n",sumeps-frac*max);

  return sumeps - frac*max;
}
