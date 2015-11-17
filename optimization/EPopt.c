#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

#define filteroverlap projsimpoverlap //projsimpoverlap, SIMPoverlap

int count=1;
int its1=100;
int its2=100;
int maxit;
int initdirect;
int mma_verbose;
/*------------------------------------------------------*/
int Nx, Ny, Nz, Mx, My, Mz, Npmlx, Npmly, Npmlz, Nxyz, Mxyz, Mzslab, DegFree;
int cx, cy, cz, posj;
double hx, hy, hz, hxyz; 
int J1direction, J2direction, minapproach, outputbase, PrintEpsC;
double omega1, omega2, Qabs;
double eps1x, eps1y, eps1z, eps2x, eps2y, eps2z, epsair, epssub1, epssub2;
Vec epsI, epsII;
double RRT, sigmax, sigmay, sigmaz;
char filenameComm[PETSC_MAX_PATH_LEN], initialdatafile[PETSC_MAX_PATH_LEN];
Mat A, B, C, D;
Vec vR, weight, ej, ek;
Mat M1, M2;
Vec epspmlQ1, epspmlQ2, epscoef1, epscoef2, epsmedium1, epsmedium2, epsSReal, epsFReal, epsC, epsCi, epsP, vgrad, vgradlocal;
Vec x1,x2,u1,u2,u3,b1,b2,J1,J2,weightedJ1,weightedJ2,Uone,Utwo,Uthree,E1j,E1jsqrek,tmp,tmp1,tmp2;
Vec Grad0, Grad1, Grad2, Grad3, Grad4;
IS from, to;
VecScatter scatter;
KSP ksp1, ksp2;
double scaleldos2, normfactor;
int pSIMP;
double bproj, etaproj;
Mat Hfilt;
KSP kspH;
int itsH=100;
double rho=0.5;

double ldospowerindex;
/*------------------------------------------------------*/

PetscErrorCode setupKSP(MPI_Comm comm, KSP *ksp, PC *pc, int solver, int iteronly);
double materialfraction(int DegFree,double *epsopt, double *grad, void *data);

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

  PetscOptionsGetInt(PETSC_NULL,"-initdirect",&initdirect,&flg);  MyCheckAndOutputInt(flg,initdirect,"initdirect","Initial number of direct LU solves");
  PetscOptionsGetInt(PETSC_NULL,"-maxit",&maxit,&flg);  MyCheckAndOutputInt(flg,maxit,"maxit","maximum krylov iterations before invoking direct solve");

  int iteronly;
  PetscOptionsGetInt(PETSC_NULL,"-iteronly",&iteronly,&flg);
  if(flg) MyCheckAndOutputInt(flg,iteronly,"iteronly","iteronly");
  if(!flg) iteronly=0;

  int blochcondition;
  double beta1[3]={0,0,0}, beta2[3]={0,0,0};
  PetscOptionsGetInt(PETSC_NULL,"-blochcondition",&blochcondition,&flg);
  if(!flg) blochcondition=0;
  PetscPrintf(PETSC_COMM_WORLD,"-------Use Bloch condition is %d\n",blochcondition);
  if(blochcondition){
    PetscOptionsGetReal(PETSC_NULL,"-beta1x",beta1  ,&flg); MyCheckAndOutputDouble(flg,beta1[0],"beta1x","Bloch vector component beta1x");
    PetscOptionsGetReal(PETSC_NULL,"-beta1y",beta1+1,&flg); MyCheckAndOutputDouble(flg,beta1[1],"beta1y","Bloch vector component beta1y");
    PetscOptionsGetReal(PETSC_NULL,"-beta1z",beta1+2,&flg); MyCheckAndOutputDouble(flg,beta1[2],"beta1z","Bloch vector component beta1z");
    PetscOptionsGetReal(PETSC_NULL,"-beta2x",beta2  ,&flg); MyCheckAndOutputDouble(flg,beta2[0],"beta2x","Bloch vector component beta2x");
    PetscOptionsGetReal(PETSC_NULL,"-beta2y",beta2+1,&flg); MyCheckAndOutputDouble(flg,beta2[1],"beta2y","Bloch vector component beta2y");
    PetscOptionsGetReal(PETSC_NULL,"-beta2z",beta2+2,&flg); MyCheckAndOutputDouble(flg,beta2[2],"beta2z","Bloch vector component beta2z");
  }

  int imposec4v;
  PetscOptionsGetInt(PETSC_NULL,"-imposec4v",&imposec4v,&flg);  MyCheckAndOutputInt(flg,imposec4v,"imposec4v","imposec4v");

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

  int anisotropicDOF=0;
  Mxyz=Mx*My*Mz, Nxyz=Nx*Ny*Nz;
  DegFree = (imposec4v==0)*(anisotropicDOF ? 3 : 1 )*Mx*My*((Mzslab==0)?Mz:1) + (imposec4v==1)*Mx*(Mx+1)/2;
  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  PetscOptionsGetReal(PETSC_NULL,"-hy",&hy,&flg);  MyCheckAndOutputDouble(flg,hy,"hy","hy");
  PetscOptionsGetReal(PETSC_NULL,"-hz",&hz,&flg);  MyCheckAndOutputDouble(flg,hz,"hz","hz");
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;

  int BCPeriod, LowerPML;
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod");
  PetscOptionsGetInt(PETSC_NULL,"-LowerPML",&LowerPML,&flg);  MyCheckAndOutputInt(flg,LowerPML,"LowerPML","LowerPML");

  int b1x[2], b1y[2], b1z[2], b2x[2], b2y[2], b2z[2];
  PetscOptionsGetInt(PETSC_NULL,"-b1xl",b1x,&flg);    MyCheckAndOutputInt(flg,b1x[0],"b1xl","BC at x lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1xu",b1x+1,&flg);  MyCheckAndOutputInt(flg,b1x[1],"b1xu","BC at x upper for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1yl",b1y,&flg);    MyCheckAndOutputInt(flg,b1y[0],"b1yl","BC at y lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1yu",b1y+1,&flg);  MyCheckAndOutputInt(flg,b1y[1],"b1yu","BC at y upper for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1zl",b1z,&flg);    MyCheckAndOutputInt(flg,b1z[0],"b1zl","BC at z lower for mode 1");
  PetscOptionsGetInt(PETSC_NULL,"-b1zu",b1z+1,&flg);  MyCheckAndOutputInt(flg,b1z[1],"b1zu","BC at z upper for mode 1");

  PetscOptionsGetInt(PETSC_NULL,"-b2xl",b2x,&flg);    MyCheckAndOutputInt(flg,b2x[0],"b2xl","BC at x lower for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2xu",b2x+1,&flg);  MyCheckAndOutputInt(flg,b2x[1],"b2xu","BC at x upper for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2yl",b2y,&flg);    MyCheckAndOutputInt(flg,b2y[0],"b2yl","BC at y lower for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2yu",b2y+1,&flg);  MyCheckAndOutputInt(flg,b2y[1],"b2yu","BC at y upper for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2zl",b2z,&flg);    MyCheckAndOutputInt(flg,b2z[0],"b2zl","BC at z lower for mode 2");
  PetscOptionsGetInt(PETSC_NULL,"-b2zu",b2z+1,&flg);  MyCheckAndOutputInt(flg,b2z[1],"b2zu","BC at z upper for mode 2");

  PetscOptionsGetInt(PETSC_NULL,"-cx",&cx,&flg);  MyCheckAndOutputInt(flg,cx,"cx","cx");
  PetscOptionsGetInt(PETSC_NULL,"-cy",&cy,&flg);  MyCheckAndOutputInt(flg,cy,"cy","cy");
  PetscOptionsGetInt(PETSC_NULL,"-cz",&cz,&flg);  MyCheckAndOutputInt(flg,cz,"cz","cz");  
  //cx=(LowerPML)*floor(Nx/2);
  //cy=(LowerPML)*floor(Ny/2);
  //cz=(LowerPML)*floor(Nz/2);
  posj = (cx*Ny+ cy)*Nz + cz;

  PetscOptionsGetInt(PETSC_NULL,"-minapproach",&minapproach,&flg);  MyCheckAndOutputInt(flg,minapproach,"minapproach","minapproach");

  PetscOptionsGetInt(PETSC_NULL,"-outputbase",&outputbase,&flg); MyCheckAndOutputInt(flg,outputbase,"outputbase","outputbase");
  PetscOptionsGetInt(PETSC_NULL,"-PrintEpsC",&PrintEpsC,&flg); MyCheckAndOutputInt(flg,PrintEpsC,"PrintEpsC","PrintEpsC");

  PetscOptionsGetInt(PETSC_NULL,"-J1dir",&J1direction,&flg);  MyCheckAndOutputInt(flg,J1direction,"J1direction","J1direction");
  PetscOptionsGetInt(PETSC_NULL,"-J2dir",&J2direction,&flg);  MyCheckAndOutputInt(flg,J2direction,"J2direction","J2direction");

  double freq1;
  PetscOptionsGetReal(PETSC_NULL,"-freq1",&freq1,&flg);
  if(!flg) freq1=1.0;
  PetscPrintf(PETSC_COMM_WORLD,"-------freq1: %g \n",freq1);
  double fratio;
  PetscOptionsGetReal(PETSC_NULL,"-fratio",&fratio,&flg);
  if(!flg) fratio=2.0;
  PetscPrintf(PETSC_COMM_WORLD,"-------fratio is %g \n",fratio);
  omega1=2.0*PI*freq1, omega2=fratio*omega1;

  PetscOptionsGetReal(PETSC_NULL,"-eps1x",&eps1x,&flg); MyCheckAndOutputDouble(flg,eps1x,"eps1x","eps1x");
  PetscOptionsGetReal(PETSC_NULL,"-eps1y",&eps1y,&flg); MyCheckAndOutputDouble(flg,eps1y,"eps1y","eps1y");
  PetscOptionsGetReal(PETSC_NULL,"-eps1z",&eps1z,&flg); MyCheckAndOutputDouble(flg,eps1z,"eps1z","eps1z");
  PetscOptionsGetReal(PETSC_NULL,"-eps2x",&eps2x,&flg); MyCheckAndOutputDouble(flg,eps2x,"eps2x","eps2x");
  PetscOptionsGetReal(PETSC_NULL,"-eps2y",&eps2y,&flg); MyCheckAndOutputDouble(flg,eps2y,"eps2y","eps2y");
  PetscOptionsGetReal(PETSC_NULL,"-eps2z",&eps2z,&flg); MyCheckAndOutputDouble(flg,eps2z,"eps2z","eps2z");

  PetscOptionsGetReal(PETSC_NULL,"-epsmed",&epsair,&flg);
  if(!flg) epsair=1.0;
  if(flg) MyCheckAndOutputDouble(flg,epsair,"epsair","Dielectric of surrounding medium");
  PetscOptionsGetReal(PETSC_NULL,"-epssub1",&epssub1,&flg);
  if(!flg) epssub1=1.0; 
  if(flg) MyCheckAndOutputDouble(flg,epssub1,"epssub1","Dielectric of substrate at freq1");
  PetscOptionsGetReal(PETSC_NULL,"-epssub2",&epssub2,&flg);
  if(!flg) epssub2=1.0;
  if(flg) MyCheckAndOutputDouble(flg,epssub2,"epssub2","Dielectric of substrate at freq2");

  RRT=1e-25;
  sigmax = pmlsigma(RRT,(double) Npmlx*hx);
  sigmay = pmlsigma(RRT,(double) Npmly*hy);
  sigmaz = pmlsigma(RRT,(double) Npmlz*hz);

  PetscPrintf(PETSC_COMM_WORLD,"sigma, omega, DegFree: %g, %g, %g, %g, %g, %d \n", sigmax, sigmay, sigmaz, omega1, omega2, DegFree);

  PetscOptionsGetReal(PETSC_NULL,"-Qabs",&Qabs,&flg);  MyCheckAndOutputDouble(flg,Qabs,"Qabs","Qabs");
  if (Qabs>1e15) Qabs=1.0/0.0;

  double Jmag;
  PetscOptionsGetReal(PETSC_NULL,"-Jmag",&Jmag,&flg);  MyCheckAndOutputDouble(flg,Jmag,"Jmag","Jmag");


  PetscOptionsGetString(PETSC_NULL,"-filenameprefix",filenameComm,PETSC_MAX_PATH_LEN,&flg);
  if (!flg) sprintf(filenameComm,"noname_");
  if (flg) MyCheckAndOutputChar(flg,filenameComm,"filenameprefix","Filename Prefix");
  PetscOptionsGetString(PETSC_NULL,"-initdatfile",initialdatafile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdatafile,"initialdatafile","Inputdata file");

  int solver;
  PetscOptionsGetInt(PETSC_NULL,"-solver",&solver,&flg);  MyCheckAndOutputInt(flg,solver,"solver","LU Direct solver choice (0 PASTIX, 1 MUMPS, 2 SUPERLU_DIST)");

  PetscOptionsGetInt(PETSC_NULL,"-pSIMP",&pSIMP,&flg);
  if(flg) MyCheckAndOutputInt(flg,pSIMP,"pSIMP","pSIMP");
  if(!flg) pSIMP=1;

  PetscOptionsGetReal(PETSC_NULL,"-bproj",&bproj,&flg);
  if(flg) MyCheckAndOutputDouble(flg,bproj,"bproj","bproj");
  if(!flg) bproj=0;
  PetscOptionsGetReal(PETSC_NULL,"-etaproj",&etaproj,&flg);
  if(flg) MyCheckAndOutputDouble(flg,etaproj,"etaproj","etaproj");
  if(!flg) etaproj=0.5;

  int readlubsfromfile;
  PetscOptionsGetInt(PETSC_NULL,"-readlubsfromfile",&readlubsfromfile,&flg);  
  if(!flg) readlubsfromfile=0;
  if(flg) MyCheckAndOutputInt(flg,readlubsfromfile,"readlubsfromfile","reading lower and upper bounds from files");

  double sH, nR;
  int dimH;
  PetscOptionsGetReal(PETSC_NULL,"-sH",&sH,&flg); MyCheckAndOutputDouble(flg,sH,"sH","sH");
  PetscOptionsGetReal(PETSC_NULL,"-nR",&nR,&flg); MyCheckAndOutputDouble(flg,nR,"nR","nR");
  PetscOptionsGetInt(PETSC_NULL,"-dimH",&dimH,&flg); MyCheckAndOutputDouble(flg,dimH,"dimH","dimH");

  int constr;
  double normalpha, normbeta;
  PetscOptionsGetReal(PETSC_NULL,"-constr",&constr,&flg); MyCheckAndOutputDouble(flg,constr,"constr","constr");
  PetscOptionsGetReal(PETSC_NULL,"-normalpha",&normalpha,&flg); MyCheckAndOutputDouble(flg,normalpha,"normalpha","normalpha");
  PetscOptionsGetReal(PETSC_NULL,"-normbeta",&normbeta,&flg); MyCheckAndOutputDouble(flg,normbeta,"normbeta","normbeta");
  

/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
/**************************************************************************************************************************************************************/
  
  //Accessory vectors and matrices for the case of c4v degrees of freedom
  Mat A1, A2;
  Vec epsSquare;

  /*------Set up the A, B, C, D matrices--------------*/
  if(imposec4v==0){
    myinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, LowerPML*floor((Nx-Mx)/2),LowerPML*floor((Ny-My)/2),LowerPML*floor((Nz-Mz)/2), Mx,My,Mz,Mzslab, anisotropicDOF);
  }else{
    myinterp(PETSC_COMM_WORLD, &A1, Nx,Ny,Nz, LowerPML*floor((Nx-Mx)/2),LowerPML*floor((Ny-My)/2),LowerPML*floor((Nz-Mz)/2), Mx,My,Mz,Mzslab, anisotropicDOF);
    c4v(PETSC_COMM_WORLD, &A2, Mx);
    MatMatMult(A1,A2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A);
  }

  int Arows, Acols;
  MatGetSize(A,&Arows,&Acols);
  PetscPrintf(PETSC_COMM_WORLD,"****Dimensions of A is %d by %d \n",Arows,Acols);
  GetDotMat(PETSC_COMM_WORLD, &B, Nx, Ny, Nz);
  CongMat(PETSC_COMM_WORLD, &C, 6*Nxyz);
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  /*-----Set up vR, weight, ej, ek------*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &vR);CHKERRQ(ierr);
  GetRealPartVec(vR,6*Nxyz);

  ierr = VecDuplicate(vR,&weight); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&ej); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&ek); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vR, "vR");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) weight, "weight");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) ej, "ej");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) ek, "ek");CHKERRQ(ierr);

  if(LowerPML==0)
    GetWeightVec(weight, Nx, Ny,Nz); 
  else
    VecSet(weight,1.0);
  GetUnitVec(ej,J1direction-1,6*Nxyz);
  GetUnitVec(ek,J2direction-1,6*Nxyz);
  
  /*----Set up the universal parts of M1 and M2-------*/
  Vec muinvpml1, muinvpml2;
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml1,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega1,LowerPML);
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml2,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega2,LowerPML); 
  double *muinv1, *muinv2;
  muinv1 = (double *) malloc(sizeof(double)*6*Nxyz);
  muinv2 = (double *) malloc(sizeof(double)*6*Nxyz);
  int add=0;
  AddMuAbsorption(muinv1,muinvpml1,Qabs,add);
  AddMuAbsorption(muinv2,muinvpml2,Qabs,add);

  if(blochcondition){
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M1, Nx,Ny,Nz, hx,hy,hz, b1x,b1y,b1z, muinv1, BCPeriod, beta1);
    MoperatorGeneralBloch(PETSC_COMM_WORLD, &M2, Nx,Ny,Nz, hx,hy,hz, b2x,b2y,b2z, muinv2, BCPeriod, beta2);
  }else{
    MoperatorGeneral(PETSC_COMM_WORLD, &M1, Nx,Ny,Nz, hx,hy,hz, b1x,b1y,b1z, muinv1, BCPeriod);
    MoperatorGeneral(PETSC_COMM_WORLD, &M2, Nx,Ny,Nz, hx,hy,hz, b2x,b2y,b2z, muinv2, BCPeriod);
  }
  ierr = PetscObjectSetName((PetscObject) M1, "M1"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) M2, "M2"); CHKERRQ(ierr);

  /*----Set up the epsilon PML vectors--------*/
  Vec unitx,unity,unitz;
  ierr = VecDuplicate(vR,&unitx);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unity);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unitz);CHKERRQ(ierr);
  GetUnitVec(unitx,0,6*Nxyz);
  GetUnitVec(unity,1,6*Nxyz);
  GetUnitVec(unitz,2,6*Nxyz);

  ierr = VecDuplicate(vR,&epsI); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsII);CHKERRQ(ierr);

  VecSet(epsI,0.0);
  VecAXPY(epsI,eps1x,unitx);
  VecAXPY(epsI,eps1y,unity);
  VecAXPY(epsI,eps1z,unitz);
  VecSet(epsII,0.0); 
  VecAXPY(epsII,eps2x,unitx);
  VecAXPY(epsII,eps2y,unity);
  VecAXPY(epsII,eps2z,unitz); 

  Vec epspml1, epspml2;
  ierr = VecDuplicate(vR,&epspml1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspml2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epspmlQ2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epspmlQ1,"EpsPMLQ1"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epspmlQ2,"EpsPMLQ2"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epscoef1,"Epscoef1"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epscoef2,"Epscoef2"); CHKERRQ(ierr);

  EpsPMLFull(PETSC_COMM_WORLD, epspml1,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega1, LowerPML);
  EpsPMLFull(PETSC_COMM_WORLD, epspml2,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega2, LowerPML);

  EpsCombine(D, weight, epspml1, epspmlQ1, epscoef1, Qabs, omega1, epsI);
  EpsCombine(D, weight, epspml2, epspmlQ2, epscoef2, Qabs, omega2, epsII);

  /*-----Set up epsmedium, epsSReal, epsFReal, epsC, epsCi, epsP, vgrad, vgradlocal ------*/
  ierr = VecDuplicate(vR,&epsmedium1); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsmedium2); CHKERRQ(ierr);
  GetMediumVecwithSub(epsmedium1,Nz,Mz,epsair,epssub1);
  GetMediumVecwithSub(epsmedium2,Nz,Mz,epsair,epssub2);
  //GetMediumVec(epsmedium1,Nz,Mz,epsair,epssub1);
  //GetMediumVec(epsmedium2,Nz,Mz,epsair,epssub2);

  ierr = MatGetVecs(A,&epsSReal, &epsFReal); CHKERRQ(ierr);
  
  ierr = VecDuplicate(vR, &epsC); CHKERRQ(ierr);
  ierr = VecDuplicate(vR, &epsCi); CHKERRQ(ierr);
  ierr = VecDuplicate(vR, &epsP); CHKERRQ(ierr);
  ierr = VecDuplicate(epsSReal, &vgrad); CHKERRQ(ierr);

  ierr = VecSet(epsP,0.0); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(epsP); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsP); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) epsmedium1,  "epsmedium1");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsmedium2,  "epsmedium2");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsC,  "epsC");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsCi, "epsCi");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsP,  "epsP");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsFReal, "epsFReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vgrad, "vgrad");CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, DegFree, &vgradlocal); CHKERRQ(ierr);

  /*---------Set up J1, b1 and weightedJ1-------------*/

  ierr = VecDuplicate(vR,&J1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) J1, "Source1");CHKERRQ(ierr);
  VecSet(J1,0.0);
  if (J1direction == 1)
    SourceSingleSetX(PETSC_COMM_WORLD, J1, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else if (J1direction ==2)
    SourceSingleSetY(PETSC_COMM_WORLD, J1, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else if (J1direction == 3)
    SourceSingleSetZ(PETSC_COMM_WORLD, J1, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else
    PetscPrintf(PETSC_COMM_WORLD," Please specify correct direction of current: x (1) , y (2) or z (3)\n "); 

  int Jopt;
  Vec Jextra;
  ierr = VecDuplicate(vR,&Jextra); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Jextra,  "Jextra");CHKERRQ(ierr);
  VecSet(Jextra,0.0);
  PetscOptionsGetInt(PETSC_NULL,"-Jopt",&Jopt,&flg);

  if (!flg) Jopt=0;
  if(Jopt==1){ //make a block source
    
    double lx,ux,ly,uy,lz,uz,amp;
    PetscOptionsGetReal(PETSC_NULL,"-lx",&lx,&flg); MyCheckAndOutputDouble(flg,lx,"lx","lx");
    PetscOptionsGetReal(PETSC_NULL,"-ux",&ux,&flg); MyCheckAndOutputDouble(flg,ux,"ux","ux");
    PetscOptionsGetReal(PETSC_NULL,"-ly",&ly,&flg); MyCheckAndOutputDouble(flg,ly,"ly","ly");
    PetscOptionsGetReal(PETSC_NULL,"-uy",&uy,&flg); MyCheckAndOutputDouble(flg,uy,"uy","uy");
    PetscOptionsGetReal(PETSC_NULL,"-lz",&lz,&flg); MyCheckAndOutputDouble(flg,lz,"lz","lz");
    PetscOptionsGetReal(PETSC_NULL,"-uz",&uz,&flg); MyCheckAndOutputDouble(flg,uz,"uz","uz");
    PetscOptionsGetReal(PETSC_NULL,"-amp",&amp,&flg); MyCheckAndOutputDouble(flg,amp,"amp","amp");

    VecSet(J1,0.0);
    SourceBlock(PETSC_COMM_WORLD,&J1,Nx,Ny,Nz,hx,hy,hz,lx,ux,ly,uy,lz,uz,amp,J1direction-1);
    
  }
  if(Jopt==2){
    char inputsrc[PETSC_MAX_PATH_LEN];
    PetscOptionsGetString(PETSC_NULL,"-inputsrc",inputsrc,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,inputsrc,"inputsrc","Input source current");

    double *Jdist;
    FILE *Jptf;
    Jdist = (double *) malloc(6*Nxyz*sizeof(double));
    Jptf = fopen(inputsrc,"r");
    int inJi;
    for (inJi=0;inJi<6*Nxyz;inJi++)
      { 
        fscanf(Jptf,"%lf",&Jdist[inJi]);
      }
    fclose(Jptf);

    ArrayToVec(Jdist,Jextra);
    ierr = VecPointwiseMult(Jextra,Jextra,ej); CHKERRQ(ierr);
    ierr = VecAXPY(Jextra,-1.0,J1); CHKERRQ(ierr);

    free(Jdist);
  }
    
  ierr = VecAXPY(J1,1.0,Jextra); CHKERRQ(ierr);
  ierr = VecDestroy(&Jextra); CHKERRQ(ierr);

  VecScale(J1,Jmag);

  if(Jopt) OutputVec(PETSC_COMM_WORLD, J1, filenameComm,"J1.m");

  ierr = VecDuplicate(vR,&b1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) b1, "rhs1");CHKERRQ(ierr);
  ierr = MatMult(D,J1,b1);CHKERRQ(ierr);
  VecScale(b1,omega1);

  ierr = VecDuplicate(vR,&weightedJ1); CHKERRQ(ierr);
  ierr = VecPointwiseMult(weightedJ1,J1,weight);
  ierr = PetscObjectSetName((PetscObject) weightedJ1, "weightedJ1");CHKERRQ(ierr);

  /*------Set up x1,x2,u1,u2,u3,b2,J2,weightedJ2,Uone,Utwo,Uthree,E1j,E1jsqrek,tmp,tmp1,tmp2, Grad0, Grad1, Grad2, Grad3, Grad4-------*/
  ierr = VecDuplicate(vR,&x1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&x2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&u1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&u2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&u3);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&b2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&J2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&weightedJ2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&Uone);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&Utwo);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&Uthree);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&E1j);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&E1jsqrek);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&tmp);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&tmp1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&tmp2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&Grad0);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&Grad1);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&Grad2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&Grad3);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&Grad4);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) x1, "E1");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x2, "E2");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u1, "u1");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u2, "u2");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u3, "u3");CHKERRQ(ierr);

  /*--------Create index sets for the vec scatter -------*/
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&to); CHKERRQ(ierr);

  /*--------Setup the KSP variables ---------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the KSP variables.--------\n ");CHKERRQ(ierr);
  PC pc1, pc2; 
  setupKSP(PETSC_COMM_WORLD,&ksp1,&pc1,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp2,&pc2,solver,iteronly);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the KSP variables DONE!--------\n ");CHKERRQ(ierr);
  /*--------Setup the KSP variables DONE. ---------------*/

  /*--------Setup Helmholtz filter---------*/
  PC pcH;
  if(imposec4v==0){
    GetH(PETSC_COMM_WORLD,&Hfilt,Mx,My,(Mzslab==0)?Mz:1,sH,nR,dimH,&kspH,&pcH);
  }else{
    GetH(PETSC_COMM_WORLD,&Hfilt,DegFree,1,1,sH,nR,dimH,&kspH,&pcH);
  }

  //OutputMat(PETSC_COMM_WORLD, Hfilt, filenameComm,"Hfilt.m");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the Hfilt DONE!--------\n ");CHKERRQ(ierr);
  /*--------Setup Helmholtz filter DONE---------*/


  /*****************************************This creates a slab out of a given 2d cross-section********************************************/
  /**/	int threeDim;															/**/
  /**/	PetscOptionsGetInt(PETSC_NULL,"-threeDim",&threeDim,&flg);									/**/
  /**/	if(!flg) threeDim=0;														/**/
  /**/	if(threeDim){															/**/
  /**/		ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Interpolation matrix A has been set up.--------\n ");CHKERRQ(ierr);	/**/
  /**/		Vec tmpepsSReal, tmpepsFReal;												/**/
  /**/		ierr = MatGetVecs(A,&tmpepsSReal, &tmpepsFReal); CHKERRQ(ierr);								/**/
  /**/																	/**/
  /**/		double *tmpepsopt;													/**/
  /**/		FILE *tmpptf;														/**/
  /**/		tmpepsopt = (double *) malloc(DegFree*sizeof(double));									/**/
  /**/		tmpptf = fopen(initialdatafile,"r");											/**/
  /**/		int itmp;														/**/
  /**/		for (itmp=0;itmp<DegFree;itmp++)											/**/
  /**/		{ 															/**/
  /**/ 			fscanf(tmpptf,"%lf",&tmpepsopt[itmp]);										/**/
  /**/		}															/**/
  /**/		fclose(tmpptf);														/**/
  /**/																	/**/
  /**/		ierr=ArrayToVec(tmpepsopt, tmpepsSReal); CHKERRQ(ierr);									/**/
  /**/		ierr=MatMult(A,tmpepsSReal,tmpepsFReal); CHKERRQ(ierr);									/**/
  /**/		ierr=VecPointwiseMult(tmpepsFReal,tmpepsFReal,epsI); CHKERRQ(ierr);							/**/
  /**/		ierr = VecAXPY(tmpepsFReal,1.0,epsmedium1); CHKERRQ(ierr);								/**/
  /**/		OutputVec(PETSC_COMM_WORLD, tmpepsFReal, "slab_","epsC.m");								/**/
  /**/		PetscPrintf(PETSC_COMM_WORLD,"--------Created the slab with given 2d cross-section.--------\n ");CHKERRQ(ierr);		/**/
  /**/		free(tmpepsopt);													/**/
  /**/		VecDestroy(&tmpepsSReal);												/**/
  /**/		VecDestroy(&tmpepsFReal);												/**/
  /**/	};																/**/
  /************************************************This creates a slab out of a given 2d cross-section*************************************/


  /*---------Setup the epsopt and grad arrays----------------*/
  double *epsopt;
  FILE *ptf;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  ptf = fopen(initialdatafile,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");
  int i;
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

/*-------------------------------------------------------------------------*/
  ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);								
  ierr=VecPointwiseMult(epsFReal,epsFReal,epsI); CHKERRQ(ierr);							
  ierr = VecAXPY(epsFReal,1.0,epsmedium1); CHKERRQ(ierr);
  OutputVec(PETSC_COMM_WORLD, epsFReal, "initial_","epsF.m");
  
  if(imposec4v){
    Vec tmp;
    ierr = MatGetVecs(A2,&tmp, &epsSquare); CHKERRQ(ierr);
    ierr = MatMult(A2,epsSReal,epsSquare); CHKERRQ(ierr);
    OutputVec(PETSC_COMM_WORLD, epsSquare, "c4veps",".m");
    OutputMat(PETSC_COMM_WORLD, A2, "c4vMatrix",".m");
    VecDestroy(&tmp);
  }

/*-------------------------------------------------------------------------*/

  EPdataGroup freq1data={omega1,M1,A,x1,b1,weightedJ1,epspmlQ1,epsmedium1,epsI,&its1,epscoef1,vgrad,ksp1,constr,normalpha,normbeta};

  int Job;
  PetscOptionsGetInt(PETSC_NULL,"-Job",&Job,&flg); MyCheckAndOutputInt(flg,Job,"Job","Job (1 gradient check [SOF, LDOS] ; 2 optimization)");

if (Job==1){

  /*---------Calculate the overlap and gradient--------*/
  int px, py, pz=0;
  double beta=0;
  double s1, ds, s2, epscen;
  int optjob;
  
  PetscOptionsGetInt(PETSC_NULL,"-px",&px,&flg);  MyCheckAndOutputInt(flg,px,"px","px");
  PetscOptionsGetInt(PETSC_NULL,"-py",&py,&flg);  MyCheckAndOutputInt(flg,py,"py","py");
  PetscOptionsGetReal(PETSC_NULL,"-s1",&s1,&flg); MyCheckAndOutputDouble(flg,s1,"s1","s1");
  PetscOptionsGetReal(PETSC_NULL,"-s2",&s2,&flg); MyCheckAndOutputDouble(flg,s2,"s2","s2");
  PetscOptionsGetReal(PETSC_NULL,"-ds",&ds,&flg); MyCheckAndOutputDouble(flg,ds,"ds","ds");
  PetscOptionsGetInt(PETSC_NULL,"-optjob",&optjob,&flg);  MyCheckAndOutputInt(flg,optjob,"optjob","Job option (1 SOF, 2 LDOS)");

  int posMj=(px*My+ py)*Mz + pz;
  for (epscen=s1;epscen<s2;epscen+=ds)
  { 
      epsopt[posMj]=epscen; 
      if (optjob==1){
	beta = EPSOF(DegFree,epsopt,grad,&freq1data);}
      else if (optjob==2){
	beta = EPLDOS(DegFree,epsopt,grad,&freq1data);}
      PetscPrintf(PETSC_COMM_WORLD,"epscen: %g objfunc: %g objfunc-grad: %g \n", epsopt[posMj], beta, grad[posMj]);
  }

}

//Job = 2 for multiple degenerate mode optimization

if (Job==2){

  //read the bc's for the third and other modes and set up the third matrix
  //we can just use the muinv1 and epspmlQ1 for all the matrices since freqs and Qabs are the same
  //also set up the ksp3, ksp4

  Vec epspmlQ,epsmedium,epsDiff,epscoef;
  Mat M3,M4;
  Vec x3,J3,weightedJ3,b3;
  Vec x4,J4,weightedJ4,b4;
  double *muinv;
  double omega;
  KSP ksp3, ksp4;
  PC pc3, pc4;
  int its3=100, its4=100;
  
  VecDuplicate(vR,&epspmlQ);
  VecDuplicate(vR,&epsmedium);
  VecDuplicate(vR,&epsDiff);
  VecDuplicate(vR,&epscoef);
  VecDuplicate(vR,&x3);
  VecDuplicate(vR,&J3);
  VecDuplicate(vR,&weightedJ3);
  VecDuplicate(vR,&b3);
  VecDuplicate(vR,&x4);
  VecDuplicate(vR,&J4);
  VecDuplicate(vR,&weightedJ4);
  VecDuplicate(vR,&b4);

  VecCopy(epspmlQ1,epspmlQ);
  VecCopy(epsmedium1,epsmedium);
  VecCopy(epsI,epsDiff);
  VecCopy(epscoef1,epscoef);
  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  AddMuAbsorption(muinv, muinvpml1, Qabs, add);
  omega=omega1;
  setupKSP(PETSC_COMM_WORLD,&ksp3,&pc3,solver,iteronly);
  setupKSP(PETSC_COMM_WORLD,&ksp4,&pc4,solver,iteronly);

  int b3x[2], b3y[2], b3z[2];
  PetscOptionsGetInt(PETSC_NULL,"-b3xl",b3x,&flg);    MyCheckAndOutputInt(flg,b3x[0],"b3xl","BC at x lower for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3xu",b3x+1,&flg);  MyCheckAndOutputInt(flg,b3x[1],"b3xu","BC at x upper for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3yl",b3y,&flg);    MyCheckAndOutputInt(flg,b3y[0],"b3yl","BC at y lower for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3yu",b3y+1,&flg);  MyCheckAndOutputInt(flg,b3y[1],"b3yu","BC at y upper for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3zl",b3z,&flg);    MyCheckAndOutputInt(flg,b3z[0],"b3zl","BC at z lower for mode 3");
  PetscOptionsGetInt(PETSC_NULL,"-b3zu",b3z+1,&flg);  MyCheckAndOutputInt(flg,b3z[1],"b3zu","BC at z upper for mode 3");

  int b4x[2], b4y[2], b4z[2];
  PetscOptionsGetInt(PETSC_NULL,"-b4xl",b4x,&flg);    MyCheckAndOutputInt(flg,b4x[0],"b4xl","BC at x lower for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4xu",b4x+1,&flg);  MyCheckAndOutputInt(flg,b4x[1],"b4xu","BC at x upper for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4yl",b4y,&flg);    MyCheckAndOutputInt(flg,b4y[0],"b4yl","BC at y lower for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4yu",b4y+1,&flg);  MyCheckAndOutputInt(flg,b4y[1],"b4yu","BC at y upper for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4zl",b4z,&flg);    MyCheckAndOutputInt(flg,b4z[0],"b4zl","BC at z lower for mode 4");
  PetscOptionsGetInt(PETSC_NULL,"-b4zu",b4z+1,&flg);  MyCheckAndOutputInt(flg,b4z[1],"b4zu","BC at z upper for mode 4");

  MoperatorGeneral(PETSC_COMM_WORLD, &M1, Nx,Ny,Nz, hx,hy,hz, b1x,b1y,b1z, muinv, BCPeriod);
  MoperatorGeneral(PETSC_COMM_WORLD, &M2, Nx,Ny,Nz, hx,hy,hz, b2x,b2y,b2z, muinv, BCPeriod);
  MoperatorGeneral(PETSC_COMM_WORLD, &M3, Nx,Ny,Nz, hx,hy,hz, b3x,b3y,b3z, muinv, BCPeriod);
  MoperatorGeneral(PETSC_COMM_WORLD, &M4, Nx,Ny,Nz, hx,hy,hz, b4x,b4y,b4z, muinv, BCPeriod);

  //Read the J's from input;
  VecSet(J1,0.0);
  VecSet(J2,0.0);
  VecSet(J3,0.0);
  VecSet(J4,0.0);
  double *J1array,*J2array,*J3array,*J4array;
  FILE *J1ptf,*J2ptf,*J3ptf,*J4ptf;
  J1array = (double *) malloc(6*Nxyz*sizeof(double));
  J2array = (double *) malloc(6*Nxyz*sizeof(double));
  J3array = (double *) malloc(6*Nxyz*sizeof(double));
  J4array = (double *) malloc(6*Nxyz*sizeof(double));
  J1ptf = fopen("J1input.txt","r");
  J2ptf = fopen("J2input.txt","r");
  J3ptf = fopen("J3input.txt","r"); 
  J4ptf = fopen("J4input.txt","r"); 
  int inJi;
  PetscPrintf(PETSC_COMM_WORLD,"---reading J1, J2, J3, J4-----\n");
  for (inJi=0;inJi<6*Nxyz;inJi++)
    { 
      fscanf(J1ptf,"%lf",&J1array[inJi]);
      fscanf(J2ptf,"%lf",&J2array[inJi]);
      fscanf(J3ptf,"%lf",&J3array[inJi]);	
      fscanf(J4ptf,"%lf",&J4array[inJi]);
    }
  fclose(J1ptf);
  fclose(J2ptf);
  fclose(J3ptf);
  fclose(J4ptf);
  PetscPrintf(PETSC_COMM_WORLD,"---Done reading J2, J2, J3, J4!-----\n");

  ArrayToVec(J1array,J1);
  ArrayToVec(J2array,J2);
  ArrayToVec(J3array,J3);
  ArrayToVec(J4array,J4);
  free(J1array);
  free(J2array);
  free(J3array);
  free(J4array);

  OutputVec(PETSC_COMM_WORLD,J1,"J1",".m");
  OutputVec(PETSC_COMM_WORLD,J2,"J2",".m");
  OutputVec(PETSC_COMM_WORLD,J3,"J3",".m");
  OutputVec(PETSC_COMM_WORLD,J4,"J4",".m");

  VecPointwiseMult(weightedJ1,weight,J1);
  VecPointwiseMult(weightedJ2,weight,J2);
  VecPointwiseMult(weightedJ3,weight,J3);
  VecPointwiseMult(weightedJ4,weight,J4);

  MatMult(D,J1,b1);
  VecScale(b1,omega);
  MatMult(D,J2,b2);
  VecScale(b2,omega);
  MatMult(D,J3,b3);
  VecScale(b3,omega);
  MatMult(D,J4,b4);
  VecScale(b4,omega);

  EPdataGroup freq2data={omega,M2,A,x2,b2,weightedJ2,epspmlQ,epsmedium,epsDiff,&its2,epscoef,vgrad,ksp2,constr,normalpha,normbeta};
  EPdataGroup freq3data={omega,M3,A,x3,b3,weightedJ3,epspmlQ,epsmedium,epsDiff,&its3,epscoef,vgrad,ksp3,constr,normalpha,normbeta};
  EPdataGroup freq4data={omega,M4,A,x4,b4,weightedJ4,epspmlQ,epsmedium,epsDiff,&its4,epscoef,vgrad,ksp4,constr,normalpha,normbeta};

 /*---------Optimization--------*/
  double tstart;
  PetscOptionsGetReal(PETSC_NULL,"-tstart",&tstart,&flg);  MyCheckAndOutputDouble(flg,tstart,"tstart","Initial value of dummy variable t");
  int DegFreeAll=DegFree+1;
  double *epsoptAll;
  epsoptAll = (double *) malloc(DegFreeAll*sizeof(double));
  for (i=0;i<DegFree;i++){ epsoptAll[i]=epsopt[i]; }
  epsoptAll[DegFreeAll-1]=tstart;
  
  int optrho;
  PetscOptionsGetInt(PETSC_NULL,"-optrho",&optrho,&flg);
  if(!flg) optrho=0;
  PetscPrintf(PETSC_COMM_WORLD,"optrho is %d \n", optrho);
  PetscOptionsGetReal(PETSC_NULL,"-rho",&rho,&flg);
  PetscPrintf(PETSC_COMM_WORLD,"material fraction rho is %0.16e \n", rho);
  
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
  lb[DegFreeAll-1]=0;
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

  if(optrho) nlopt_add_inequality_constraint(opt,materialfraction, NULL,1e-8);

  int nummodes;
  PetscOptionsGetInt(PETSC_NULL,"-nummodes",&nummodes,&flg);  MyCheckAndOutputInt(flg,nummodes,"nummodes","number of degenerate modes (2, 3 or 4) to collapse");

  if(nummodes==2){
    nlopt_add_inequality_constraint(opt,EPLDOS,&freq1data,1e-8);
    nlopt_add_inequality_constraint(opt,EPSOF,&freq1data,1e-8);
    nlopt_add_inequality_constraint(opt,EPLDOS,&freq2data,1e-8);
    nlopt_add_inequality_constraint(opt,EPSOF,&freq2data,1e-8);
  }else if(nummodes==3){
    nlopt_add_inequality_constraint(opt,EPLDOS,&freq1data,1e-8);
    nlopt_add_inequality_constraint(opt,EPSOF,&freq1data,1e-8);
    nlopt_add_inequality_constraint(opt,EPLDOS,&freq2data,1e-8);
    nlopt_add_inequality_constraint(opt,EPSOF,&freq2data,1e-8);
    nlopt_add_inequality_constraint(opt,EPLDOS,&freq3data,1e-8);
    nlopt_add_inequality_constraint(opt,EPSOF,&freq3data,1e-8);
  }else if(nummodes==4){
    nlopt_add_inequality_constraint(opt,EPLDOS,&freq1data,1e-8);
    nlopt_add_inequality_constraint(opt,EPSOF,&freq1data,1e-8);
    nlopt_add_inequality_constraint(opt,EPLDOS,&freq2data,1e-8);
    nlopt_add_inequality_constraint(opt,EPSOF,&freq2data,1e-8);
    nlopt_add_inequality_constraint(opt,EPLDOS,&freq3data,1e-8);
    nlopt_add_inequality_constraint(opt,EPSOF,&freq3data,1e-8);
    nlopt_add_inequality_constraint(opt,EPLDOS,&freq4data,1e-8);
    nlopt_add_inequality_constraint(opt,EPSOF,&freq4data,1e-8);
  }

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

  free(epsoptAll);
  free(lb);
  free(ub);
  nlopt_destroy(opt);

  VecDestroy(&epspmlQ);
  VecDestroy(&epsmedium);
  VecDestroy(&epsDiff);
  VecDestroy(&epscoef);
  MatDestroy(&M3);
  VecDestroy(&x3);
  VecDestroy(&J3);
  VecDestroy(&weightedJ3);
  VecDestroy(&b3);
  MatDestroy(&M4);
  VecDestroy(&x4);
  VecDestroy(&J4);
  VecDestroy(&weightedJ4);
  VecDestroy(&b4);
  free(muinv);
  KSPDestroy(&ksp3); 
  KSPDestroy(&ksp4); 

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
  ierr = MatDestroy(&M1); CHKERRQ(ierr);  
  ierr = MatDestroy(&M2); CHKERRQ(ierr);
  ierr = MatDestroy(&Hfilt); CHKERRQ(ierr);

  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&weight); CHKERRQ(ierr);
  ierr = VecDestroy(&ej); CHKERRQ(ierr);
  ierr = VecDestroy(&ek); CHKERRQ(ierr);

  ierr = VecDestroy(&unitx); CHKERRQ(ierr);
  ierr = VecDestroy(&unity); CHKERRQ(ierr);
  ierr = VecDestroy(&unitz); CHKERRQ(ierr);
  ierr = VecDestroy(&epsI); CHKERRQ(ierr);
  ierr = VecDestroy(&epsII); CHKERRQ(ierr);

  ierr = VecDestroy(&muinvpml1); CHKERRQ(ierr);
  ierr = VecDestroy(&muinvpml2); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml1); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml2); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ1); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ2); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef1); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef2); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium1); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium2); CHKERRQ(ierr);
  ierr = VecDestroy(&epsSReal); CHKERRQ(ierr);
  ierr = VecDestroy(&epsFReal); CHKERRQ(ierr);
  ierr = VecDestroy(&epsC); CHKERRQ(ierr);
  ierr = VecDestroy(&epsCi); CHKERRQ(ierr);
  ierr = VecDestroy(&epsP); CHKERRQ(ierr);
  ierr = VecDestroy(&vgrad); CHKERRQ(ierr);  
  ierr = VecDestroy(&vgradlocal); CHKERRQ(ierr);

  ierr = VecDestroy(&x1); CHKERRQ(ierr);
  ierr = VecDestroy(&x2); CHKERRQ(ierr);
  ierr = VecDestroy(&u1); CHKERRQ(ierr);
  ierr = VecDestroy(&u2); CHKERRQ(ierr);
  ierr = VecDestroy(&u3); CHKERRQ(ierr);
  ierr = VecDestroy(&b1);CHKERRQ(ierr);
  ierr = VecDestroy(&b2); CHKERRQ(ierr);
  ierr = VecDestroy(&J1); CHKERRQ(ierr);
  ierr = VecDestroy(&J2); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ1); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ2); CHKERRQ(ierr);
  ierr = VecDestroy(&Uone); CHKERRQ(ierr);
  ierr = VecDestroy(&Utwo); CHKERRQ(ierr);
  ierr = VecDestroy(&Uthree); CHKERRQ(ierr);
  ierr = VecDestroy(&E1j); CHKERRQ(ierr);
  ierr = VecDestroy(&E1jsqrek); CHKERRQ(ierr);
  ierr = VecDestroy(&tmp); CHKERRQ(ierr);
  ierr = VecDestroy(&tmp1); CHKERRQ(ierr);
  ierr = VecDestroy(&tmp2); CHKERRQ(ierr);
  ierr = VecDestroy(&Grad0); CHKERRQ(ierr);
  ierr = VecDestroy(&Grad1); CHKERRQ(ierr);
  ierr = VecDestroy(&Grad2); CHKERRQ(ierr);
  ierr = VecDestroy(&Grad3); CHKERRQ(ierr);
  ierr = VecDestroy(&Grad4); CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp1);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp2);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspH);CHKERRQ(ierr);

  MatDestroy(&A1);
  MatDestroy(&A2);
  VecDestroy(&epsSquare);

  ISDestroy(&from);
  ISDestroy(&to);

  free(muinv1);
  free(muinv2);
  free(epsopt);
  free(grad);

  /*------------ finalize the program -------------*/

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Barrier(PETSC_COMM_WORLD);
  
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}
  
double materialfraction(int DegFree,double *epsopt, double *grad, void *data)
{
  int i;
  double sumeps;
  
  sumeps=0.0;
  for (i=0;i<DegFree;i++){
          sumeps+=epsopt[i];
	  grad[i]=-1.0/(double)Mx;
  }

  PetscPrintf(PETSC_COMM_WORLD,"******the current material fraction is %1.6e \n",sumeps/(double)Mx);

  return rho - sumeps/(double)Mx;
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

