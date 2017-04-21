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
double printfile(Vec x, const char *name, int N);
PetscErrorCode setupKSP(MPI_Comm comm, KSP *ksp, PC *pc, int solver, int iteronly);
double pfunc_constr(int DegFree, double *epsopt, double *grad, void *data);
double pfunc_noconstr(int DegFree, double *epsopt, double *grad, void *data);
double optimize_refphi(double phi0, int DegFree, double *epsopt, void *data, int alg, int localalg, int maxeval, int maxtime, double *lphi, double *uphi);
double optimize_eps(int DegFree, double *epsopt, void *data, int alg, int localalg, int maxeval, int maxtime);
double optimize_freqmaximin(int DegFree, double *epsopt, void *data, int alg, int localalg, int maxeval, int maxtime, int nmodes, double initdummy);

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
  int jx0,jy0,jz0;
  getreal("-jlx",&jlx,0);
  getreal("-jux",&jux,Nx*hx);
  getreal("-jly",&jly,0);
  getreal("-juy",&juy,Ny*hy);
  getreal("-jlz",&jlz,Npmlz*hz+1/5);
  getreal("-juz",&juz,jlz+hz);
  getint("-jx0",&jx0,0);
  getint("-jy0",&jy0,0);
  getint("-jz0",&jz0,0);

  //same measurement / opt plance for all currents
  int rlx, rux, rly, ruy, rlz, ruz, rx0,ry0,rz0;
  getint("-rlx",&rlx,0);
  getint("-rux",&rux,Nx);
  getint("-rly",&rly,0);
  getint("-ruy",&ruy,Ny);
  getint("-rlz",&rlz,Nz-Npmlz-5);
  getint("-ruz",&ruz,rlz+1);
  getint("-rx0",&rx0,(Nx-1)/2);
  getint("-ry0",&ry0,rly);
  getint("-rz0",&rz0,rlz);

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
  int tmpnlayers;
  getint("-nlayers",&tmpnlayers,1);
  const int nlayers=tmpnlayers;

  int tmpnfreq;
  getint("-nfreq",&tmpnfreq,1);
  const int nfreq=tmpnfreq;

  int tmpnangle;
  getint("-nangle",&tmpnangle,1);
  const int nangle=tmpnangle;

  int i;
  char tmpflg[PETSC_MAX_PATH_LEN];
  int *Mz, *Nzo;
  Mz=(int *) malloc(nlayers*sizeof(int));
  Nzo=(int *) malloc(nlayers*sizeof(int));

  DegFree=0;
  for(i=0;i<nlayers;i++){
    sprintf(tmpflg,"-Mz[%d]",i+1);
    getint(tmpflg,Mz+i,10);
    sprintf(tmpflg,"-Nzo[%d]",i+1);
    getint(tmpflg,Nzo+i,LowerPMLz*(Nz-Mz[i])/2);
    DegFree=DegFree+Mx*My*((Mzslab==0)?Mz[i]:1);
  }

  int j;
  double tmp;
  double betax[nfreq], betay[nfreq], betaz[nfreq];
  double epsdiff[nfreq][nlayers], epsbkg[nfreq][nlayers];
  double epssub[nfreq], epsair[nfreq], epsmid[nfreq];
  double epssubdiff[nfreq], epsairdiff[nfreq], epsmiddiff[nfreq];
  int bxl[nfreq], byl[nfreq], bzl[nfreq];
  int bxu[nfreq], byu[nfreq], bzu[nfreq];
  int Jdir[nfreq];
  double freq[nfreq], omega[nfreq], Jmag[nfreq];
  int devicedir[nfreq];
  double refphi[nfreq*nangle], focallength[nfreq];
  double inc_angle[nfreq*nangle];
  int jpt;
  for(i=0;i<nfreq;i++){

    sprintf(tmpflg,"-betax[%d]",i+1);
    getreal(tmpflg,betax+i,0);
    sprintf(tmpflg,"-betay[%d]",i+1);
    getreal(tmpflg,betay+i,0);
    sprintf(tmpflg,"-betaz[%d]",i+1);
    getreal(tmpflg,betaz+i,0);

    sprintf(tmpflg,"-epssub[%d]",i+1);
    getreal(tmpflg,epssub+i,2);
    sprintf(tmpflg,"-epsair[%d]",i+1);
    getreal(tmpflg,epsair+i,2);
    sprintf(tmpflg,"-epsmid[%d]",i+1);
    getreal(tmpflg,epsmid+i,2);
    sprintf(tmpflg,"-epssubdiff[%d]",i+1);
    getreal(tmpflg,epssubdiff+i,0);
    sprintf(tmpflg,"-epsairdiff[%d]",i+1);
    getreal(tmpflg,epsairdiff+i,0);
    sprintf(tmpflg,"-epsmiddiff[%d]",i+1);
    getreal(tmpflg,epsmiddiff+i,0);
    
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
    getint(tmpflg,Jdir+i,1);
    sprintf(tmpflg,"-freq[%d]",i+1);
    getreal(tmpflg,freq+i,1);
    omega[i]=2*PI*freq[i];
    sprintf(tmpflg,"-Jmag[%d]",i+1);
    getreal(tmpflg,Jmag+i,1);

    sprintf(tmpflg,"-devicedir[%d]",i+1);
    getint(tmpflg,devicedir+i,1);
    sprintf(tmpflg,"-focallength[%d]",i+1);
    getreal(tmpflg,focallength+i,0);

    for(j=0;j<nangle;j++){
      jpt=j*nfreq+i;
      //jpt=i*nangle+j;     
      sprintf(tmpflg,"-refphi[%d][%d]",i+1,j+1);
      getreal(tmpflg,refphi+jpt,0);
      sprintf(tmpflg,"-incangle[%d][%d]",i+1,j+1);
      getreal(tmpflg,inc_angle+jpt,0);
    }
  }

  int symlens;
  getint("-symlens",&symlens,1);

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
  if(symlens){
    Mat A1,A2;
    if(Mzslab==1){
      mirrorA1d(PETSC_COMM_WORLD,&A1,Mx,nlayers);
    }else{
      mirrorA2d(PETSC_COMM_WORLD,&A1,Mx,Mz[0],nlayers);
    }
    layeredA(PETSC_COMM_WORLD,&A2, Nx,Ny,Nz, nlayers,Nxo,Nyo,Nzo, 2*Mx-1,My,Mz, Mzslab);
    MatMatMult(A2,A1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A);
    MatDestroy(&A1);
    MatDestroy(&A2);
  }else{
    layeredA(PETSC_COMM_WORLD,&A, Nx,Ny,Nz, nlayers,Nxo,Nyo,Nzo, Mx,My,Mz, Mzslab);
  }

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
  Vec J[nfreq*nangle], b[nfreq*nangle], weightedJ[nfreq*nangle], x[nfreq*nangle];
  Vec pvec[nfreq*nangle], qvec[nfreq*nangle];
  KSP ksp[nfreq];
  PC pc[nfreq];
  int its[nfreq];
  Meta data[nfreq*nangle];
  double *tmpeps;
  double jkx,jky,jkz;
  tmpeps=(double *) malloc(nlayers*sizeof(double));
  for(i=0;i<nfreq;i++){
    MuinvPMLGeneral(PETSC_COMM_SELF, muinvpml+i, Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega[i],LowerPMLx,LowerPMLy,LowerPMLz);
    AddMuAbsorption(muinv,muinvpml[i],Qabs,1);
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
    for(j=0;j<nlayers;j++){
      tmpeps[j]=epsdiff[i][j];
    }
    VecSet(epsVec[i],0.0);
    layeredepsdiff(epsVec[i],Nx,Ny,Nz, nlayers,Nzo,Mz, tmpeps, epssubdiff[i], epsairdiff[i], epsmiddiff[i]);
    
    VecDuplicate(vR,epsmedium+i);
    for(j=0;j<nlayers;j++){
      tmpeps[j]=epsbkg[i][j];
    }
    VecSet(epsmedium[i],0.0);
    layeredepsbkg(epsmedium[i],Nx,Ny,Nz, nlayers,Nzo,Mz, tmpeps, epssub[i], epsair[i], epsmid[i]);
    
    VecDuplicate(vR,epspmlQ+i);
    VecDuplicate(vR,epscoef+i);
    EpsPMLGeneral(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega[i], LowerPMLx,LowerPMLy,LowerPMLz);
    EpsCombine(D, weight, epspml, epspmlQ[i], epscoef[i], Qabs, omega[i], epsVec[i]);

    for(j=0;j<nangle;j++){
      jpt=j*nfreq+i;
      //jpt=i*nangle+j;
      //jkx=2*PI*freq[i]*sqrt(epssub[i])*sin(inc_angle[jpt]*PI/180);
      jkx=2*PI*freq[i]*sin(inc_angle[jpt]*PI/180);
      jky=0;
      jkz=0;
      SourceAngled(PETSC_COMM_WORLD, J+jpt, Nx,Ny,Nz, hx,hy,hz, jlx,jux,jly,juy,jlz,juz, 1.0/hz,Jdir[i],jkx,jky,jkz,jx0,jy0,jz0);
      VecScale(J[jpt],Jmag[i]);
      VecDuplicate(vR,b+jpt);
      MatMult(D,J[jpt],b[jpt]);CHKERRQ(ierr);
      VecScale(b[jpt],omega[i]);
      VecDuplicate(vR,weightedJ+jpt);
      VecPointwiseMult(weightedJ[jpt],J[jpt],weight);
      VecDuplicate(vR,x+jpt);

      makepq_lens_inc(PETSC_COMM_WORLD,pvec+jpt,qvec+jpt, Nx,Ny,Nz, rlx,rux,rly,ruy,rlz,ruz, devicedir[i], focallength[i], inc_angle[jpt]*PI/180, 1/(freq[i]*hz),refphi[jpt], rx0,ry0,rz0);
    }

    setupKSP(PETSC_COMM_WORLD,ksp+i,pc+i,solver,iteronly);
    its[i]=100;

    for(j=0;j<nangle;j++){
      jpt=j*nfreq+i;
      //jpt=i*nangle+j;
      (data+jpt)->Nx=Nx;
      (data+jpt)->Ny=Ny;
      (data+jpt)->Nz=Nz;
      (data+jpt)->epsSReal=epsSReal;
      (data+jpt)->epsFReal=epsFReal;
      (data+jpt)->omega=omega[i];
      (data+jpt)->M=M[i];
      (data+jpt)->A=A;
      (data+jpt)->b=b[jpt];
      (data+jpt)->x=x[jpt];
      (data+jpt)->epspmlQ=epspmlQ[i];
      (data+jpt)->epsmedium=epsmedium[i];
      (data+jpt)->epsDiff=epsVec[i];
      (data+jpt)->epscoef=epscoef[i];
      (data+jpt)->ksp=ksp[i];
      (data+jpt)->its=its+i;
      (data+jpt)->pvec=pvec[jpt];
      (data+jpt)->qvec=qvec[jpt];
      (data+jpt)->outputbase=outputbase;
      (data+jpt)->refphi=refphi+jpt;
      (data+jpt)->inc_angle=inc_angle[jpt];
    }
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
    int ifreq, iangle;
    getint("-posMj",&posMj,0);
    getreal("-s1",&s1,0);
    getreal("-ds",&ds,0.01);
    getreal("-s2",&s2,1.0);
    getint("-ifreq",&ifreq,0);
    getint("-iangle",&iangle,0);
    int ipt=iangle*nfreq+ifreq;
    //int ipt=ifreq*nangle+iangle;
    for (epscen=s1;epscen<s2;epscen+=ds)
      {
        epsopt[posMj]=epscen;
        beta = batchmeta(DegFree,epsopt,grad,data+ipt);
        PetscPrintf(PETSC_COMM_WORLD,"epscen: %g objfunc: %g objfunc-grad: %g \n", epsopt[posMj], beta, grad[posMj]);
      }
    
  }

  if(Job==2){
  
    int ifreq,iangle;
    getint("-ifreq",&ifreq,0);
    getint("-iangle",&iangle,0);
    int ipt=iangle*nfreq+ifreq;
    //int ipt=ifreq*nangle+iangle;
    batchmeta(DegFree,epsopt,grad,data+ipt);
    sprintf(tmpflg,"freq%g_angle%g_x.dat",freq[ifreq],inc_angle[ipt]);
    printfile((data+ipt)->x,tmpflg,6*Nxyz);

    Vec epsFull;
    VecDuplicate(vR,&epsFull);
    ArrayToVec(epsopt,epsSReal);
    MatMult(A,epsSReal,epsFReal);
    VecPointwiseMult(epsFull,epsFReal,(data+ipt)->epsDiff);
    VecAXPY(epsFull,1.0,(data+ipt)->epsmedium);
    VecPointwiseMult(epsFull,epsFull,vR);
    sprintf(tmpflg,"freq%g_angle%g_eps.dat",freq[ifreq],inc_angle[ipt]);
    printfile(epsFull,tmpflg,6*Nxyz);

    sprintf(tmpflg,"freq%g_angle%g_J.dat",freq[ifreq],inc_angle[ipt]);
    printfile(J[ipt],tmpflg,6*Nxyz);

    sprintf(tmpflg,"freq%g_angle%g_pvec.dat",freq[ifreq],inc_angle[ipt]);
    printfile((data+ipt)->pvec,tmpflg,6*Nxyz);

    sprintf(tmpflg,"freq%g_angle%g_qvec.dat",freq[ifreq],inc_angle[ipt]);
    printfile((data+ipt)->qvec,tmpflg,6*Nxyz);
  
  }


  if(Job==3){

    /*---------Optimize refphi--------*/
    int phi_alg, phi_localalg, phi_maxeval, phi_maxtime;
    double lphi, uphi;
    getint("-phialg",&phi_alg,34);
    getint("-philocalalg",&phi_localalg,0);
    getint("-phimaxeval",&phi_maxeval,1000);
    getint("-phimaxtime",&phi_maxtime,100000);
    getreal("-lphi",&lphi,0);
    getreal("-uphi",&uphi,2*PI);

    int ifreq,iangle;
    getint("-ifreq",&ifreq,0);
    getint("-iangle",&iangle,0);
    int ipt=iangle*nfreq+ifreq;
    //int ipt=ifreq*nangle+iangle;

    optimize_refphi(refphi[ipt],DegFree,epsopt,data+ipt,phi_alg,phi_localalg,phi_maxeval,phi_maxtime,&lphi,&uphi);
    
  }

  if(Job==4){

    /*---------Optimize eps--------*/
    int eps_alg, eps_localalg, eps_maxeval, eps_maxtime;
    getint("-epsalg",&eps_alg,24);
    getint("-epslocalalg",&eps_localalg,0);
    getint("-epsmaxeval",&eps_maxeval,1000);
    getint("-epsmaxtime",&eps_maxtime,100000);

    int ifreq,iangle;
    getint("-ifreq",&ifreq,0);
    getint("-iangle",&iangle,0);
    int ipt=iangle*nfreq+ifreq;
    //int ipt=ifreq*nangle+iangle;

    optimize_eps(DegFree,epsopt,data+ipt,eps_alg,eps_localalg,eps_maxeval,eps_maxtime);
    
  }

  if(Job==5){

    /*---------Optimize eps--------*/
    int alg, localalg, maxeval, maxtime;
    getint("-alg",&alg,24);
    getint("-localalg",&localalg,0);
    getint("-maxeval",&maxeval,50000);
    getint("-maxtime",&maxtime,100000);

    int nmodes;
    getint("-nmodes",&nmodes,nfreq);

    double initdummy;
    getreal("-initdummy",&initdummy,-1);

    /********OPT*********/

    int ndof=DegFree+1;
    double *dof, *lb, *ub;
    dof = (double *) malloc(ndof*sizeof(double));
    lb = (double *) malloc(ndof*sizeof(double));
    ub = (double *) malloc(ndof*sizeof(double));
    
    for(i=0;i<DegFree;i++){
      dof[i]=epsopt[i];
      lb[i]=0;
      ub[i]=1;
    } 
    dof[ndof-1]=initdummy;
    lb[ndof-1]=-2;
    ub[ndof-1]=2;

    nlopt_opt opt;
    nlopt_opt local_opt;


    double maxf;
    opt = nlopt_create(alg, ndof);
    nlopt_set_lower_bounds(opt,lb);
    nlopt_set_upper_bounds(opt,ub);
    nlopt_set_maxeval(opt,maxeval);
    nlopt_set_maxtime(opt,maxtime);
    if(alg==11) nlopt_set_vector_storage(opt,4000);
    if(localalg){
      local_opt=nlopt_create(localalg, ndof);
      nlopt_set_ftol_rel(local_opt, 1e-14);
      nlopt_set_maxeval(local_opt,10000);
      nlopt_set_local_optimizer(opt,local_opt);
    }

    int ifreq,iangle,ipt;
    for(ifreq=0;ifreq<nmodes;ifreq++){
      for(iangle=0;iangle<nangle;iangle++){
	  ipt=iangle*nfreq+ifreq;
	  //ipt=ifreq*nangle+iangle;
	  nlopt_add_inequality_constraint(opt,batchmaximin,data+ipt,1e-8);
      }
    }

    nlopt_set_max_objective(opt,maximinobjfun,NULL);
    double result=nlopt_optimize(opt,dof,&maxf);
    
    PetscPrintf(PETSC_COMM_WORLD,"----nlopt maximin returns %d \n",result);
  
    nlopt_destroy(opt);
    if(localalg) nlopt_destroy(local_opt);

    /********OPT********/

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

double printfile(Vec x, const char *name, int N)
{

  VecScatter scat;
  Vec vlocal;
  IS scto, scfrom;
  VecCreateSeq(PETSC_COMM_SELF,N,&vlocal);
  ISCreateStride(PETSC_COMM_SELF,N,0,1,&scfrom);
  ISCreateStride(PETSC_COMM_SELF,N,0,1,&scto);

  // scatter x to vlocal;
  VecScatterCreate(x,scfrom,vlocal,scto,&scat);
  VecScatterBegin(scat,x,vlocal,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(scat,x,vlocal,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterDestroy(&scat);

  // copy from vlocal to ptvlocal;
  double *ptvlocal;
  VecGetArray(vlocal,&ptvlocal);

  int i;
  double *tmp;
  tmp=(double *) malloc(N*sizeof(double));
  for(i=0;i<N;i++)
    tmp[i] = ptvlocal[i];
  VecRestoreArray(vlocal,&ptvlocal);

  int rankA;
  FILE *ptf;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rankA);

  if(rankA==0)
    {
      ptf = fopen(name,"w");
      for (i=0;i<N;i++)
	fprintf(ptf,"%g \n",tmp[i]);
      fclose(ptf);
    }

  VecDestroy(&vlocal);

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

double optimize_refphi(double phi0, int DegFree, double *epsopt, void *data, int alg, int localalg, int maxeval, int maxtime, double *lphi, double *uphi)
{

  //initialize x;
  batchmeta(DegFree,epsopt,NULL,data);
  count=1;

  double phivar=phi0;
  nlopt_opt opt;
  nlopt_opt local_opt;
  
  int ndof=1;
  double maxf;
  opt = nlopt_create(alg, ndof);
  nlopt_set_lower_bounds(opt,lphi);
  nlopt_set_upper_bounds(opt,uphi);
  nlopt_set_maxeval(opt,maxeval);
  nlopt_set_maxtime(opt,maxtime);
  if(alg==11) nlopt_set_vector_storage(opt,4000);
  if(localalg){
    local_opt=nlopt_create(localalg, ndof);
    nlopt_set_ftol_rel(local_opt, 1e-14);
    nlopt_set_maxeval(local_opt,10000);
    nlopt_set_local_optimizer(opt,local_opt);
  }

  nlopt_set_max_objective(opt,refphiopt,data);
  double result=nlopt_optimize(opt,&phivar,&maxf);
  
  nlopt_destroy(opt);
  if(localalg) nlopt_destroy(local_opt);
  count=1;

  return result;

}

double optimize_eps(int DegFree, double *epsopt, void *data, int alg, int localalg, int maxeval, int maxtime)
{

  nlopt_opt opt;
  nlopt_opt local_opt;
  double *lb, *ub;
  lb = (double *) malloc(DegFree*sizeof(double));
  ub = (double *) malloc(DegFree*sizeof(double));
  int i;
  for(i=0;i<DegFree;i++){
    lb[i]=0;
    ub[i]=1;
  }

  double maxf;
  opt = nlopt_create(alg, DegFree);
  nlopt_set_lower_bounds(opt,lb);
  nlopt_set_upper_bounds(opt,ub);
  nlopt_set_maxeval(opt,maxeval);
  nlopt_set_maxtime(opt,maxtime);
  if(alg==11) nlopt_set_vector_storage(opt,4000);
  if(localalg){
    local_opt=nlopt_create(localalg, DegFree);
    nlopt_set_ftol_rel(local_opt, 1e-14);
    nlopt_set_maxeval(local_opt,10000);
    nlopt_set_local_optimizer(opt,local_opt);
  }

  nlopt_set_max_objective(opt,batchmeta,data);
  double result=nlopt_optimize(opt,epsopt,&maxf);
  
  nlopt_destroy(opt);
  if(localalg) nlopt_destroy(local_opt);

  return result;

}

double optimize_freqmaximin(int DegFree, double *epsopt, void *data, int alg, int localalg, int maxeval, int maxtime, int nmodes, double initdummy)
{

  int ndof=DegFree+1;
  double *dof, *lb, *ub;
  dof = (double *) malloc(ndof*sizeof(double));
  lb = (double *) malloc(DegFree*sizeof(double));
  ub = (double *) malloc(DegFree*sizeof(double));

  int i;
  for(i=0;i<DegFree;i++){
    dof[i]=epsopt[i];
    lb[i]=0;
    ub[i]=1;
  } 
  dof[ndof-1]=initdummy;
  lb[ndof-1]=-2;
  ub[ndof-1]=2;

  nlopt_opt opt;
  nlopt_opt local_opt;


  double maxf;
  opt = nlopt_create(alg, ndof);
  nlopt_set_lower_bounds(opt,lb);
  nlopt_set_upper_bounds(opt,ub);
  nlopt_set_maxeval(opt,maxeval);
  nlopt_set_maxtime(opt,maxtime);
  if(alg==11) nlopt_set_vector_storage(opt,4000);
  if(localalg){
    local_opt=nlopt_create(localalg, ndof);
    nlopt_set_ftol_rel(local_opt, 1e-14);
    nlopt_set_maxeval(local_opt,10000);
    nlopt_set_local_optimizer(opt,local_opt);
  }

  int ifreq;
  for(ifreq=0;ifreq<nmodes;ifreq++){
    nlopt_add_inequality_constraint(opt,batchmaximin,data+ifreq,1e-8);
  }

  nlopt_set_max_objective(opt,maximinobjfun,NULL);
  double result=nlopt_optimize(opt,dof,&maxf);
  
  nlopt_destroy(opt);
  if(localalg) nlopt_destroy(local_opt);

  return result;

}
