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
extern double scaleldos2;

#undef __FUNCT__ 
#define __FUNCT__ "ldos1constraint"
double ldos1constraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the LDOS1 solver (Full Vectorial Version). Minimum approach NOT available.********** \n");
  
  // copy epsoptAll to epsSReal, fills the first DegFree elements;
  ierr=ArrayToVec(epsoptAll, epsSReal); CHKERRQ(ierr);
  
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

  if (gradAll) {
  // Derivative of LDOS1 wrt eps = Re [ x^2 wt I/omega epscoef ];
  // Since the constraint is t - LDOS1, the derivative of the constraint (wrt eps) is -Re [ x^2 wt I/omega epscoef ];
  CmpVecProd(x1,x1,Grad0);
  CmpVecProd(Grad0,epscoef1,tmp);
  ierr = MatMult(D,tmp,Grad0); CHKERRQ(ierr);
  ierr = VecPointwiseMult(Grad0,Grad0,weight); CHKERRQ(ierr);
  VecScale(Grad0,-1.0/omega1);
  VecScale(Grad0,hxyz);
  ierr = VecPointwiseMult(Grad0,Grad0,vR); CHKERRQ(ierr);

  ierr = MatMultTranspose(A,Grad0,vgrad);CHKERRQ(ierr);
  ierr = VecToArray(vgrad,gradAll,scatter,from,to,vgradlocal,DegFree);

  gradAll[DegFreeAll-1]=1;

  }

  ierr = MatDestroy(&Mone); CHKERRQ(ierr);
  ierr = VecDestroy(&xdiff); CHKERRQ(ierr);


  return epsoptAll[DegFreeAll-1]-ldos1;
}

#undef __FUNCT__ 
#define __FUNCT__ "ldos2constraint"
double ldos2constraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the LDOS2 solver (Full Vectorial Version). Minimum approach NOT available.********** \n");
  
  // copy epsoptAll to epsSReal, fills the first DegFree elements;
  ierr=ArrayToVec(epsoptAll, epsSReal); CHKERRQ(ierr);
  
  // Update the diagonals of M2;
  Mat Mtwo;
  MatDuplicate(M2,MAT_COPY_VALUES,&Mtwo);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(Mtwo, A, D, epsSReal, epspmlQ2, epsmedium2, epsC, epsCi, epsP, Nxyz, omega2, epsII);

  /*-----------------KSP2 Solving------------------*/   
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = Ptime(&t1);CHKERRQ(ierr);

  if (its2> (maxit-5) || count< initdirect )
    {
      PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern, LU is redone! \n");
      ierr = KSPSetOperators(ksp2,Mtwo,Mtwo);CHKERRQ(ierr);}
  else
    {ierr = KSPSetReusePreconditioner(ksp2,PETSC_TRUE);CHKERRQ(ierr);}

   ierr = KSPSolve(ksp2,b2,x2);CHKERRQ(ierr);
   ierr = KSPGetIterationNumber(ksp2,&its2);CHKERRQ(ierr);
   ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations (KSP2) in this step is %d----\n ",its2);CHKERRQ(ierr);

   // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
    if(its2>(maxit-2))
      {
	PetscPrintf(PETSC_COMM_WORLD,"Too many iterations needed! Recomputing \n");
	ierr = KSPSetOperators(ksp2,Mtwo,Mtwo);CHKERRQ(ierr);
	ierr = KSPSolve(ksp2,b2,x2);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(ksp2,&its2);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %d---\n ",its2);CHKERRQ(ierr);
      }

  //Print kspsolving information
  double norm;
  Vec xdiff;
  ierr=VecDuplicate(x2,&xdiff);CHKERRQ(ierr);
  ierr = MatMult(Mtwo,x2, xdiff);CHKERRQ(ierr);
  ierr = VecAXPY(xdiff,-1.0,b2);CHKERRQ(ierr);
  ierr = VecNorm(xdiff,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"---Norm of error %g, Kryolv Iterations %d----\n ",norm,its2);CHKERRQ(ierr);    

  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0)
  PetscPrintf(PETSC_COMM_SELF,"---The runing time is %f s \n",tpast);
  /*--------------Finish KSP2 Solving---------------*/  

  /*-------------Calculate and print out the LDOS2 at fundamental freq----------*/
  //tmpldos2 = -Re((wt.*J^*)'*E) 
  double tmpldos2, tmpldos2r, tmpldos2i, ldos2;
  CmpVecDot(x2,weightedJ2,&tmpldos2r,&tmpldos2i);
  tmpldos2=-1.0*tmpldos2r;
  ldos2=tmpldos2*hxyz;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current ldos2 at step %.5d is %.16e \n", count,ldos2);
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current scaled-ldos2 at step %.5d is %.16e \n", count,scaleldos2*ldos2);

  if (gradAll) {
  // Derivative of LDOS2 wrt eps = Re [ x^2 wt I/omega epscoef ];
  // Since the constraint is t - LDOS2, the derivative of the constraint (wrt eps) is -Re [ x^2 wt I/omega epscoef ];
  CmpVecProd(x2,x2,Grad0);
  CmpVecProd(Grad0,epscoef2,tmp);
  ierr = MatMult(D,tmp,Grad0); CHKERRQ(ierr);
  ierr = VecPointwiseMult(Grad0,Grad0,weight); CHKERRQ(ierr);
  VecScale(Grad0,-1.0/omega2);
  VecScale(Grad0,hxyz);
  VecScale(Grad0,scaleldos2);
  ierr = VecPointwiseMult(Grad0,Grad0,vR); CHKERRQ(ierr);

  ierr = MatMultTranspose(A,Grad0,vgrad);CHKERRQ(ierr);
  ierr = VecToArray(vgrad,gradAll,scatter,from,to,vgradlocal,DegFree);

  gradAll[DegFreeAll-1]=1;

  }

  ierr = MatDestroy(&Mtwo); CHKERRQ(ierr);
  ierr = VecDestroy(&xdiff); CHKERRQ(ierr);

  return epsoptAll[DegFreeAll-1]-scaleldos2*ldos2;
}

