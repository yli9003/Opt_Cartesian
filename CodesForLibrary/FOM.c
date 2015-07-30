#include <petsc.h>
#include <time.h>
#include "libOPT.h"
#include <complex.h>
#include "petsctime.h"

#define Ptime PetscTime

/*The Objective function in this code is P2/LDOS1^2 which is calculated to be proportional to Q1^2 Q2 |beta|^2*/

extern double ldospowerindex;

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
#define __FUNCT__ "FOM"
double FOM(int DegFree,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the Proj & SIMP OverlapSolver (Full Vectorial Version). Minimum approach NOT available.********** \n");

  Vec gradP2, gradP1;
  ierr=VecDuplicate(J1,&gradP1); CHKERRQ(ierr);
  ierr=VecDuplicate(J1,&gradP2); CHKERRQ(ierr);
  

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
  
  // Update the diagonals of M1 and M2 Matrices;
  Mat Mone, Mtwo;
  MatDuplicate(M1,MAT_COPY_VALUES,&Mone);
  MatDuplicate(M2,MAT_COPY_VALUES,&Mtwo);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(Mone, A, D, epsSReal, epspmlQ1, epsmedium1, epsC, epsCi, epsP, Nxyz, omega1, epsI);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(Mtwo, A, D, epsSReal, epspmlQ2, epsmedium2, epsC, epsCi, epsP, Nxyz, omega2, epsII);

  /*-----------------KSP1 Solving------------------*/   
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = Ptime(&t1);CHKERRQ(ierr);

  if (its1> (maxit-5) || count< initdirect )
    {
      PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern, LU is redone! \n");
      ierr = KSPSetOperators(ksp1,Mone,Mone);CHKERRQ(ierr);}
  else
    { PetscPrintf(PETSC_COMM_WORLD,"Same preconditioner for ksp1! \n");
      ierr = KSPSetOperators(ksp1,Mone,Mone);CHKERRQ(ierr);
      ierr = KSPSetReusePreconditioner(ksp1,PETSC_TRUE);CHKERRQ(ierr);}

   ierr = KSPSolve(ksp1,b1,x1);CHKERRQ(ierr);
   ierr = KSPGetIterationNumber(ksp1,&its1);CHKERRQ(ierr);
   ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations (KSP1) in this step is %d----\n ",its1);CHKERRQ(ierr);

   // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
    if(its1>(maxit-2))
      {
	PetscPrintf(PETSC_COMM_WORLD,"Too many iterations needed! Recomputing \n");
	ierr = KSPSetOperators(ksp1,Mone,Mone);CHKERRQ(ierr);
	ierr = KSPSetReusePreconditioner(ksp1,PETSC_FALSE);CHKERRQ(ierr);
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

  /*------set up b2 and wt.*Conj(J2) for overlap calculation-------------*/
  //E1jsqrek=(ej.x1)*(ej.x1)*ek; 
  ierr = VecPointwiseMult(E1j,ej,x1); CHKERRQ(ierr);
  CmpVecProd(E1j,E1j,tmp);
  ierr = MatMult(B,tmp,E1jsqrek); CHKERRQ(ierr);
  ierr = VecPointwiseMult(E1jsqrek,E1jsqrek,ek); CHKERRQ(ierr);

  //J2=epsFReal*E1jsqrek;
  ierr=VecPointwiseMult(J2,E1jsqrek,epsFReal); CHKERRQ(ierr);

  //b2=i * omega2 * J2;
  ierr=MatMult(D,J2,b2); CHKERRQ(ierr);
  VecScale(b2,omega2);

  //wtJ2= wt*Conj(J2);
  ierr=MatMult(C,J2,weightedJ2); CHKERRQ(ierr);
  ierr=VecPointwiseMult(weightedJ2,weightedJ2,weight); CHKERRQ(ierr);

  /************DELETE THIS AFTER DEBUGGING**********/
  //VecCopy(b1,b2);
  //VecCopy(weightedJ1,weightedJ2);
  /************DELETE THIS AFTER DEBUGGING**********/
  

  /*-----------------KSP2 Solving------------------*/   
  ierr = PetscTime(&t1);CHKERRQ(ierr);
 
  if (its2> (maxit-5) || count< initdirect )
    {
      PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern, LU is redone! \n");
      ierr = KSPSetOperators(ksp2,Mtwo,Mtwo);CHKERRQ(ierr);}
  else
    { PetscPrintf(PETSC_COMM_WORLD,"Same preconditioner for ksp2! \n");
      ierr = KSPSetOperators(ksp2,Mtwo,Mtwo);CHKERRQ(ierr);
      ierr = KSPSetReusePreconditioner(ksp2,PETSC_TRUE);CHKERRQ(ierr);}

   ierr = KSPSolve(ksp2,b2,x2);CHKERRQ(ierr);
   ierr = KSPGetIterationNumber(ksp2,&its2);CHKERRQ(ierr);
   ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations (KSP2) in this step is %d----\n ",its2);CHKERRQ(ierr);

   // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
    ierr = KSPGetIterationNumber(ksp2,&its2);CHKERRQ(ierr);
    if(its2>(maxit-2))
      {
	PetscPrintf(PETSC_COMM_WORLD,"Too many iterations needed! Recomputing \n");
	ierr = KSPSetOperators(ksp2,Mtwo,Mtwo);CHKERRQ(ierr);
        ierr = KSPSetReusePreconditioner(ksp2,PETSC_FALSE);CHKERRQ(ierr);
	ierr = KSPSolve(ksp2,b2,x2);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(ksp2,&its2);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %d---\n ",its2);CHKERRQ(ierr);
      }

  //Print kspsolving information
  ierr = MatMult(Mtwo,x2, xdiff);CHKERRQ(ierr);
  ierr = VecAXPY(xdiff,-1.0,b2);CHKERRQ(ierr);
  ierr = VecNorm(xdiff,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"---Norm of error %g, Kryolv Iterations %d----\n ",norm,its2);CHKERRQ(ierr);    
  ierr=VecDestroy(&xdiff);CHKERRQ(ierr);

  ierr = PetscTime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0)
  PetscPrintf(PETSC_COMM_SELF,"---The runing time is %f s \n",tpast);

  /*--------------Finish KSP2 Solving---------------*/

  /*-------------Calculate the overlap----------------*/
  //tmpbeta = -Re((wt.*J^*)'*E)
  double beta, tmpbeta, tmpbetar, tmpbetai; 
  CmpVecDot(x2,weightedJ2,&tmpbetar,&tmpbetai);
  tmpbeta=-1.0*tmpbetar;
  beta=tmpbeta*hxyz;

  PetscPrintf(PETSC_COMM_WORLD,"---The current overlap (maxapp) at step %.5d is %.16e \n", count,beta);

  PetscPrintf(PETSC_COMM_WORLD,"-------------------------------------------------------------- \n");
  
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
/*------------------------------------------------*/

 /*-------take care of the gradient---------*/
  if (grad) {
    //Uone = ej x1 (B 2 p ek conj(x2));
    ierr = MatMult(C,x2,tmp2); CHKERRQ(ierr);
    ierr = VecPointwiseMult(tmp2,tmp2,ek); CHKERRQ(ierr);
    ierr = VecPointwiseMult(tmp2,tmp2,epsFReal); CHKERRQ(ierr);
    VecScale(tmp2,2.0);
    ierr = MatMult(B,tmp2,tmp); CHKERRQ(ierr);
    CmpVecProd(tmp,E1j,Uone);

    ierr = Ptime(&t1);CHKERRQ(ierr);
    ierr = KSPSolve(ksp1,Uone,tmp);CHKERRQ(ierr);
    ierr = MatMult(C,tmp,u1);
    ierr = Ptime(&t2);CHKERRQ(ierr);
    tpast=t2-t1;
    if(rank==0)
	PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving Mone u1 = Uone is %f s \n",tpast);

    //Utwo = conj(J2);
    ierr = MatMult(C,J2,Utwo); CHKERRQ(ierr);

    ierr = Ptime(&t1);CHKERRQ(ierr);
    ierr = KSPSolve(ksp2,Utwo,u2);CHKERRQ(ierr);
    ierr = Ptime(&t2);CHKERRQ(ierr);
    tpast=t2-t1;
    if(rank==0)
	PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving Mtwo u2 = Utwo is %f s \n",tpast);

    //Uthree = 2 ej x1 B p u2 ek;
    ierr = VecPointwiseMult(tmp,u2,ek); CHKERRQ(ierr);
    ierr = VecPointwiseMult(tmp,tmp,epsFReal); CHKERRQ(ierr);
    ierr = MatMult(B,tmp,tmp2); CHKERRQ(ierr);
    CmpVecProd(tmp2,E1j,Uthree);
    VecScale(Uthree,2.0);

    ierr = Ptime(&t1);CHKERRQ(ierr);
    ierr = KSPSolve(ksp1,Uthree,u3);CHKERRQ(ierr);
    ierr = Ptime(&t2);CHKERRQ(ierr);
    tpast=t2-t1;
    if(rank==0)
	PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving Mone u3 = Uthree is %f s \n",tpast);


    //Grad0 = < (ek.x2) conj(ej.x1)^2 >;
    ierr = MatMult(C,E1jsqrek,tmp1);
    CmpVecProd(tmp1,x2,Grad0);
    ierr = VecPointwiseMult(Grad0,Grad0,weight); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Grad0,Grad0,vR); CHKERRQ(ierr);

    //Grad1 = < u1 conj(epscoef1 x1) >;
    CmpVecProd(epscoef1,x1,tmp1);
    ierr = MatMult(C,tmp1,tmp); CHKERRQ(ierr);
    CmpVecProd(tmp,u1,Grad1);
    ierr = VecPointwiseMult(Grad1,Grad1,weight); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Grad1,Grad1,vR); CHKERRQ(ierr);

    //Grad2 = < u2 epscoef2 x2 >
    CmpVecProd(epscoef2,x2,tmp); 
    CmpVecProd(tmp,u2,Grad2);
    ierr = VecPointwiseMult(Grad2,Grad2,weight); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Grad2,Grad2,vR); CHKERRQ(ierr);

    //Grad3 = i omega2 < u2 B [ej x1^2] ek >
    CmpVecProd(u2,E1jsqrek,tmp);
    ierr = MatMult(D,tmp,Grad3); CHKERRQ(ierr);
    VecScale(Grad3,omega2);
    ierr = VecPointwiseMult(Grad3,Grad3,weight); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Grad3,Grad3,vR); CHKERRQ(ierr);

    //Grad4 = i oemga2 < u3 epscoef1 x1 >
    CmpVecProd(epscoef1,x1,tmp1);
    CmpVecProd(tmp1,u3,tmp);
    ierr = MatMult(D,tmp,Grad4); CHKERRQ(ierr);
    VecScale(Grad4,omega2);
    ierr = VecPointwiseMult(Grad4,Grad4,weight); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Grad4,Grad4,vR); CHKERRQ(ierr);
     
    VecSet(gradP2,0.0);
    VecAXPY(gradP2,1.0,Grad0);
    VecAXPY(gradP2,1.0,Grad1);
    VecAXPY(gradP2,1.0,Grad2);
    VecAXPY(gradP2,1.0,Grad3);
    VecAXPY(gradP2,1.0,Grad4);

    VecScale(gradP2,-hxyz);

    //Calculate gradP1;
    // Derivative of LDOS1 wrt eps = Re [ x^2 wt I/omega epscoef ];
    CmpVecProd(x1,x1,gradP1);
    CmpVecProd(gradP1,epscoef1,tmp);
    ierr = MatMult(D,tmp,gradP1); CHKERRQ(ierr);
    ierr = VecPointwiseMult(gradP1,gradP1,weight); CHKERRQ(ierr);
    VecScale(gradP1,1.0/omega1);
    VecScale(gradP1,hxyz);
    ierr = VecPointwiseMult(gradP1,gradP1,vR); CHKERRQ(ierr);


    VecScale(gradP2,1.0/pow(ldos1,ldospowerindex));
    VecWAXPY(tmp,-1.0*ldospowerindex*beta/pow(ldos1,ldospowerindex+1.0),gradP1,gradP2);


    MatMultTranspose(A,tmp,vgrad);
    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);

    KSPSolveTranspose(kspH,vgrad,epsgrad);   //gradient for the Helmholtz filter

    // copy vgrad (distributed vector) to a regular array grad;
    ierr = VecToArray(epsgrad,grad,scatter,from,to,vgradlocal,DegFree);
  }  

  double fom=beta/pow(ldos1,ldospowerindex);
  PetscPrintf(PETSC_COMM_WORLD,"---The current fom (maxapp) at step %.5d is %.16e \n", count,fom);

  count++;

  ierr = MatDestroy(&Mone); CHKERRQ(ierr);
  ierr = MatDestroy(&Mtwo); CHKERRQ(ierr);
  
  ierr = VecDestroy(&gradP1); CHKERRQ(ierr);
  ierr = VecDestroy(&gradP2); CHKERRQ(ierr);
  ierr = VecDestroy(&epsgrad); CHKERRQ(ierr);
  free(epsoptH);

  return fom;
}
