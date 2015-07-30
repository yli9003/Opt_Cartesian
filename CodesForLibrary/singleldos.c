#include <petsc.h>
#include <time.h>
#include "libOPT.h"
#include <complex.h>
#include "petsctime.h"

#define Ptime PetscTime

//define Global variables;
extern int count;
extern int its1;
extern int its2;
extern int maxit;
/*------------------------------------------------------*/
extern int Nx, Ny, Nz, Mx, My, Mz, Npmlx, Npmly, Npmlz, Nxyz, Mxyz, Mzslab, DegFree;
extern int cx, cy, cz, posj;
extern double hx, hy, hz, hxyz; 
extern int J1direction, J2direction, minapproach, outputbase;
extern double omega1, omega2, Qabs;
extern double eps1x, eps1y, eps1z, eps2x, eps2y, eps2z, epsair, epssub;
extern Vec epsI, epsII;
extern double RRT, sigmax, sigmay, sigmaz;
extern char filenameComm[PETSC_MAX_PATH_LEN], initialdatafile[PETSC_MAX_PATH_LEN];
extern Mat A, B, C, D;
extern Vec vR, weight, ej, ek;
extern Mat M1, M2;
extern Vec epspmlQ1, epspmlQ2, epscoef1, epscoef2, epsmedium1, epsmedium2, epsSReal, epsFReal, epsC, epsCi, epsP, vgrad, vgradlocal;
extern Vec x1,x2,u1,u2,u3,b1,b2,J1,J2,weightedJ1,weightedJ2,Uone,Utwo,Uthree,E1j,E1jsqrek,tmp,tmp1,tmp2;
extern Vec Grad0, Grad1, Grad2, Grad3, Grad4;
extern IS from, to;
extern VecScatter scatter;
extern KSP ksp1, ksp2;

extern int initdirect, PrintEpsC;
extern Mat Hfilt;
extern KSP kspH;
extern PC pcH; 

#undef __FUNCT__ 
#define __FUNCT__ "minLDOS"
double minLDOS(int DegFreeAll,double *epsopt, double *grad, void *data)
{
  double ldosmin= *((double *)data);
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the LDOS solver (Full Vectorial Version). Minimum approach NOT available.********** \n");
  
  /**Apply Helmholtz filter here**/
  char tmpfname[100];
  sprintf(tmpfname,"%.5d.m",count);
  double *epsoptH;
  Vec epsVec, epsH;
  ierr=VecDuplicate(epsSReal,&epsVec); CHKERRQ(ierr);
  ierr=VecDuplicate(epsSReal,&epsH); CHKERRQ(ierr);
  epsoptH = (double *) malloc(DegFree*sizeof(double));  
  ierr=ArrayToVec(epsopt,epsVec); CHKERRQ(ierr);
  SolveH(PETSC_COMM_WORLD,kspH,Hfilt,epsVec,epsH);
  //OutputVec(PETSC_COMM_WORLD, epsH, "epsH", tmpfname);
  ierr = VecToArray(epsH,epsoptH,scatter,from,to,vgradlocal,DegFree);
  ierr = VecDestroy(&epsVec); CHKERRQ(ierr);
  ierr = VecDestroy(&epsH); CHKERRQ(ierr);
  /**Apply Helmholtz filter here**/
  
  Vec epsgrad;
  ierr=VecDuplicate(epsSReal,&epsgrad); CHKERRQ(ierr);
  applyfilters(DegFree,epsoptH,epsSReal,epsgrad);
  
  // Update the diagonals of M1;
  Mat Mone;
  MatDuplicate(M1,MAT_COPY_VALUES,&Mone);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(Mone, A, D, epsSReal, epspmlQ1, epsmedium1, epsC, epsCi, epsP, Nxyz, omega1, epsI);

  /*-----------------KSP1 Solving------------------*/   
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = Ptime(&t1);CHKERRQ(ierr);

  if (its1> (maxit-5) || count< initdirect )
    {
      PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern, LU is redone! \n");
      ierr = KSPSetOperators(ksp1,Mone,Mone);CHKERRQ(ierr);}
  else
    {ierr = KSPSetReusePreconditioner(ksp1,PETSC_TRUE);CHKERRQ(ierr);}

   ierr = KSPSolve(ksp1,b1,x1);CHKERRQ(ierr);
   ierr = KSPGetIterationNumber(ksp1,&its1);CHKERRQ(ierr);
   ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations (KSP1) in this step is %d----\n ",its1);CHKERRQ(ierr);

   // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
    if(its1>(maxit-2))
      {
	PetscPrintf(PETSC_COMM_WORLD,"Too many iterations needed! Recomputing \n");
	ierr = KSPSetOperators(ksp1,Mone,Mone);CHKERRQ(ierr);
	ierr = KSPSolve(ksp1,b1,x1);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(ksp1,&its1);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %d---\n ",its1);CHKERRQ(ierr);
      }

  //Print kspsolving information
  double norm;
  Vec xdiff;
  ierr=VecDuplicate(x1,&xdiff);CHKERRQ(ierr);
  ierr = MatMult(Mone,x1, xdiff);CHKERRQ(ierr);
  ierr = VecAXPY(xdiff,-1.0,b1);CHKERRQ(ierr);
  ierr = VecNorm(xdiff,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"---Norm of error %g, Kryolv Iterations %d----\n ",norm,its1);CHKERRQ(ierr);    

  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0)
  PetscPrintf(PETSC_COMM_SELF,"---The runing time is %f s \n",tpast);
  /*--------------Finish KSP1 Solving---------------*/  

  /*-------------Calculate and print out the LDOS1 at fundamental freq----------*/
  //tmpldos1 = -Re((wt.*J^*)'*E) 
  double tmpldos1, tmpldos1r, tmpldos1i, ldos1;
  CmpVecDot(x1,weightedJ1,&tmpldos1r,&tmpldos1i);
  tmpldos1=-1.0*tmpldos1r;
  ldos1=tmpldos1*hxyz;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current ldos1 at step %.5d is %.16e \n", count,ldos1);

  if (grad) {
  // Derivative of LDOS1 wrt eps = Re [ x^2 wt I/omega epscoef ];
  CmpVecProd(x1,x1,Grad0);
  CmpVecProd(Grad0,epscoef1,tmp);
  ierr = MatMult(D,tmp,Grad0); CHKERRQ(ierr);
  ierr = VecPointwiseMult(Grad0,Grad0,weight); CHKERRQ(ierr);
  VecScale(Grad0,1.0/omega1);
  VecScale(Grad0,hxyz);
  VecScale(Grad0,-1.0); //Note that ldos here is a constraint: ldosmin - ldos < 0
  ierr = VecPointwiseMult(Grad0,Grad0,vR); CHKERRQ(ierr);

  ierr = MatMultTranspose(A,Grad0,vgrad);CHKERRQ(ierr);

  ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);

  KSPSolveTranspose(kspH,vgrad,epsgrad);   //gradient for the Helmholtz filter

  // copy vgrad (distributed vector) to a regular array grad;
  ierr = VecToArray(epsgrad,grad,scatter,from,to,vgradlocal,DegFree);
  }

  /*-------------- Now store the epsilon at each step--------------*/
  char buffer [100];

  int STORE=1;    
  if(STORE==1 && (count%outputbase==0))
    {
      sprintf(buffer,"%.5depsSReal.m",count);
      OutputVec(PETSC_COMM_WORLD, epsSReal, filenameComm, buffer);
    }
  if(STORE==1 && (count%PrintEpsC==0))
    {
      MatMult(A,epsSReal,epsC);
      VecPointwiseMult(epsC,epsC,epsI); 
      VecAXPY(epsC,1.0,epsmedium1);
      sprintf(buffer,"%.5depsCI.m",count);
      OutputVec(PETSC_COMM_WORLD, epsC, filenameComm, buffer);

      MatMult(A,epsSReal,epsC);
      VecPointwiseMult(epsC,epsC,epsII); 
      VecAXPY(epsC,1.0,epsmedium2);
      sprintf(buffer,"%.5depsCII.m",count);
      OutputVec(PETSC_COMM_WORLD, epsC, filenameComm, buffer);
    }

  /*------------------------------------------------*/

  ierr = MatDestroy(&Mone); CHKERRQ(ierr);
  ierr = VecDestroy(&epsgrad); CHKERRQ(ierr);
  ierr = VecDestroy(&xdiff); CHKERRQ(ierr);
  free(epsoptH);

  return ldosmin - ldos1;
}
