#include <petsc.h>
#include <time.h>
#include "libOPT.h"
#include <complex.h>
#include "petsctime.h"

#define Ptime PetscTime

extern double penalty;

extern Mat Hfilt;
extern KSP kspH;
extern PC pcH;

extern int Nxyz;
extern double hxyz;

extern Vec vR, weight;
extern Mat A, C, D;

extern int its1, its2;

extern int minapproach;

extern VecScatter scatter;
extern IS from, to;
extern Vec vgradlocal;

extern int count;
extern int maxit;
extern int initdirect;

extern char filenameComm[PETSC_MAX_PATH_LEN];

#undef __FUNCT__ 
#define __FUNCT__ "thgfom"
double thgfom(int DegFree,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;

  THGdataGroup *ptdata = (THGdataGroup *) data;

  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  Mat M1 = ptdata->M1;
  Vec x1 = ptdata->x1;
  Vec weightedJ1 = ptdata->weightedJ1;
  Vec b1 = ptdata->b1;
  Vec ej = ptdata->ej;
  Vec epsI = ptdata->epsI;
  Vec epspmlQ1 = ptdata->epspmlQ1;
  Vec epsmedium1 = ptdata->epsmedium1;
  Vec epscoef1 = ptdata->epscoef1;
  double omega1 = ptdata->omega1;
  Mat M3 = ptdata->M3;
  Vec epsIII = ptdata->epsIII;
  Vec epspmlQ3 = ptdata->epspmlQ3;
  Vec epsmedium3 = ptdata->epsmedium3;
  Vec epscoef3 = ptdata->epscoef3;
  double omega3 = ptdata->omega3;
  KSP ksp1 = ptdata->ksp1;
  KSP ksp2 = ptdata->ksp2;
  double ldospowerindex = ptdata->ldospowerindex;
  int outputbase = ptdata->outputbase;
  
  Vec x3,J3,weightedJ3,b3;
  Vec epsC, epsCi, epsP, tmp, tmp1, tmp2, E1cube, u1, u2, u3, Uone, Utwo, Uthree, Grad0, Grad1, Grad2, Grad3, Grad4;
  Vec vgrad;
  VecDuplicate(epsFReal,&x3);
  VecDuplicate(epsFReal,&J3);
  VecDuplicate(epsFReal,&weightedJ3);
  VecDuplicate(epsFReal,&b3);
  VecDuplicate(epsFReal,&epsC);
  VecDuplicate(epsFReal,&epsCi);
  VecDuplicate(epsFReal,&epsP);
  VecDuplicate(epsFReal,&tmp);
  VecDuplicate(epsFReal,&tmp1);
  VecDuplicate(epsFReal,&tmp2);
  VecDuplicate(epsFReal,&E1cube);
  VecDuplicate(epsFReal,&u1);
  VecDuplicate(epsFReal,&u2);
  VecDuplicate(epsFReal,&u3);
  VecDuplicate(epsFReal,&Uone);
  VecDuplicate(epsFReal,&Utwo);
  VecDuplicate(epsFReal,&Uthree);
  VecDuplicate(epsFReal,&Grad0);
  VecDuplicate(epsFReal,&Grad1);
  VecDuplicate(epsFReal,&Grad2);
  VecDuplicate(epsFReal,&Grad3);
  VecDuplicate(epsFReal,&Grad4);
  VecDuplicate(epsSReal,&vgrad);
  
  PetscPrintf(PETSC_COMM_WORLD,"----Third harmonic overlap. Using Mone, Mthree. Minapproach available. ------- \n");
  
  /**Apply Helmholtz filter here**/
  double *epsoptH;
  Vec epsVec, epsH;
  ierr=VecDuplicate(epsSReal,&epsVec); CHKERRQ(ierr);
  ierr=VecDuplicate(epsSReal,&epsH); CHKERRQ(ierr);
  epsoptH = (double *) malloc(DegFree*sizeof(double));  
  ierr=ArrayToVec(epsopt,epsVec); CHKERRQ(ierr);
  SolveH(PETSC_COMM_WORLD,kspH,Hfilt,epsVec,epsH);
  ierr = VecToArray(epsH,epsoptH,scatter,from,to,vgradlocal,DegFree);
  ierr = VecDestroy(&epsVec); CHKERRQ(ierr);
  ierr = VecDestroy(&epsH); CHKERRQ(ierr);
  /**Apply Helmholtz filter here**/
  
  Vec epsgrad;
  ierr=VecDuplicate(epsSReal,&epsgrad); CHKERRQ(ierr);
  applyfilters(DegFree,epsoptH,epsSReal,epsgrad);
  
  // Update the diagonals of M1 and M3 Matrices;
  Mat Mone, Mthree;
  MatDuplicate(M1,MAT_COPY_VALUES,&Mone);
  MatDuplicate(M3,MAT_COPY_VALUES,&Mthree);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(Mone, A, D, epsSReal, epspmlQ1, epsmedium1, epsC, epsCi, epsP, Nxyz, omega1, epsI);
  VecSet(epsP,0.0);
  ModifyMatDiagonals(Mthree, A, D, epsSReal, epspmlQ3, epsmedium3, epsC, epsCi, epsP, Nxyz, omega3, epsIII);

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

  /*------set up b3 and wt.*Conj(J3) for overlap calculation-------------*/
  //E1cube=E1*E1*E1*ej; 
  CmpVecProd(x1,x1,tmp);
  CmpVecProd(tmp,x1,E1cube);
  ierr = VecPointwiseMult(E1cube,E1cube,ej); CHKERRQ(ierr);

  //J3=epsFReal*E1cube;
  ierr=VecPointwiseMult(J3,E1cube,epsFReal); CHKERRQ(ierr);

  //b3=i * omega3 * J3;
  ierr=MatMult(D,J3,b3); CHKERRQ(ierr);
  VecScale(b3,omega3);

  //wtJ3= wt*Conj(J3);
  ierr=MatMult(C,J3,weightedJ3); CHKERRQ(ierr);
  ierr=VecPointwiseMult(weightedJ3,weightedJ3,weight); CHKERRQ(ierr);

  /*-----------------KSP2 Solving------------------*/
  ierr = PetscTime(&t1);CHKERRQ(ierr);

  if (its2> (maxit-5) || count< initdirect )
    {
      PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern, LU is redone! \n");
      ierr = KSPSetOperators(ksp2,Mthree,Mthree);CHKERRQ(ierr);}
  else
    { PetscPrintf(PETSC_COMM_WORLD,"Same preconditioner for ksp2! \n");
      ierr = KSPSetOperators(ksp2,Mthree,Mthree);CHKERRQ(ierr);
      ierr = KSPSetReusePreconditioner(ksp2,PETSC_TRUE);CHKERRQ(ierr);}

  ierr = KSPSolve(ksp2,b3,x3);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp2,&its2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations (KSP2) in this step is %d----\n ",its2);CHKERRQ(ierr);

  // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
  ierr = KSPGetIterationNumber(ksp2,&its2);CHKERRQ(ierr);
  if(its2>(maxit-2))
    {
      PetscPrintf(PETSC_COMM_WORLD,"Too many iterations needed! Recomputing \n");
      ierr = KSPSetOperators(ksp2,Mthree,Mthree);CHKERRQ(ierr);
      ierr = KSPSetReusePreconditioner(ksp2,PETSC_FALSE);CHKERRQ(ierr);
      ierr = KSPSolve(ksp2,b3,x3);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(ksp2,&its2);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %d---\n ",its2);CHKERRQ(ierr);
    }

  //Print kspsolving information
  ierr = MatMult(Mthree,x3, xdiff);CHKERRQ(ierr);
  ierr = VecAXPY(xdiff,-1.0,b3);CHKERRQ(ierr);
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
  CmpVecDot(x3,weightedJ3,&tmpbetar,&tmpbetai);
  tmpbeta=-1.0*tmpbetar;
  beta=tmpbeta*hxyz;

  PetscPrintf(PETSC_COMM_WORLD,"---The current overlap at step %.5d is %.16e \n", count,beta);

  double fom;
  fom=beta/pow(ldos1,ldospowerindex);

  if(minapproach)
    {
      PetscPrintf(PETSC_COMM_WORLD,"---The current fom (minapp) at step %.5d is %.16e \n", count,fom);
      fom = 1.0/fom;
      PetscPrintf(PETSC_COMM_WORLD,"---The current invfom (minapp) at step %.5d is %.16e \n", count,fom);
    }
  else
      PetscPrintf(PETSC_COMM_WORLD,"---The current fom (maxapp) at step %.5d is %.16e \n", count,fom);



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

/*------------------------------------------------*/
/*------------------------------------------------*/

 /*-------take care of the gradient---------*/
  if (grad) {
    //Uone = wt * eps * 3 * conj(x1)^2 * x3 * ej;
    CmpVecProd(x1,x1,tmp);
    ierr = MatMult(C,tmp,tmp1); CHKERRQ(ierr);
    CmpVecProd(tmp1,x3,Uone);
    ierr = VecPointwiseMult(Uone,Uone,ej); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Uone,Uone,epsFReal); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Uone,Uone,weight); CHKERRQ(ierr);
    VecScale(Uone,3.0);

    ierr = Ptime(&t1);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(ksp1,Uone,u1);CHKERRQ(ierr);
    ierr = Ptime(&t2);CHKERRQ(ierr);
    tpast=t2-t1;
    if(rank==0)
	PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving Mone^*T u1 = Uone is %f s \n",tpast);

    //Utwo = wt * eps * x1^3 * ej;
    ierr = VecPointwiseMult(Utwo,J3,weight); CHKERRQ(ierr);
   
    ierr = Ptime(&t1);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(ksp2,Utwo,u2);CHKERRQ(ierr);
    ierr = Ptime(&t2);CHKERRQ(ierr);
    tpast=t2-t1;
    if(rank==0)
	PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving Mthree^*T u2 = Utwo is %f s \n",tpast);

    //Uthree = 3 * eps * conj(x1)^2 * u2 * ej;
    CmpVecProd(tmp1,u2,Uthree);
    ierr = VecPointwiseMult(Uthree,Uthree,epsFReal); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Uthree,Uthree,ej); CHKERRQ(ierr);
    VecScale(Uthree,3.0);

    ierr = Ptime(&t1);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(ksp1,Uthree,u3);CHKERRQ(ierr);
    ierr = Ptime(&t2);CHKERRQ(ierr);
    tpast=t2-t1;
    if(rank==0)
	PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving Mone^*T u3 = Uthree is %f s \n",tpast);

    //Grad0 = wt * conj(x1)^3 * x3 * ej;
    ierr = MatMult(C,E1cube,tmp);
    CmpVecProd(tmp,x3,Grad0);
    ierr = VecPointwiseMult(Grad0,Grad0,weight); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Grad0,Grad0,vR); CHKERRQ(ierr);

    //Grad1 = u1 * conj(epscoef1 * x1);
    CmpVecProd(epscoef1,x1,tmp1);
    ierr = MatMult(C,tmp1,tmp); CHKERRQ(ierr);
    CmpVecProd(tmp,u1,Grad1);
    ierr = VecPointwiseMult(Grad1,Grad1,vR); CHKERRQ(ierr);

    //Grad2 = conj(u2) * epscoef3 * x3;
    ierr = MatMult(C,u2,tmp2); CHKERRQ(ierr); 
    CmpVecProd(tmp2,epscoef3,tmp);
    CmpVecProd(tmp,x3,Grad2);
    ierr = VecPointwiseMult(Grad2,Grad2,vR); CHKERRQ(ierr);

    //Grad3 = i*omega3 * conj(u2) * x1^3 * ej;
    ierr = MatMult(C,u2,tmp2); CHKERRQ(ierr);
    CmpVecProd(tmp2,E1cube,tmp);
    ierr = MatMult(D,tmp,Grad3); CHKERRQ(ierr);
    VecScale(Grad3,omega3);
    ierr = VecPointwiseMult(Grad3,Grad3,vR); CHKERRQ(ierr);

    //Grad4 = i*omega3 * conj(u3) * epscoef1 * x1;
    ierr = MatMult(C,u3,tmp1); CHKERRQ(ierr);
    CmpVecProd(tmp1,epscoef1,tmp);
    CmpVecProd(tmp,x1,tmp2);
    ierr = MatMult(D,tmp2,Grad4); CHKERRQ(ierr);
    VecScale(Grad4,omega3);
    ierr = VecPointwiseMult(Grad4,Grad4,vR); CHKERRQ(ierr);
     
    VecSet(tmp,0.0);
    VecAXPY(tmp,1.0,Grad0);
    VecAXPY(tmp,1.0,Grad1);
    VecAXPY(tmp,1.0,Grad2);
    VecAXPY(tmp,1.0,Grad3);
    VecAXPY(tmp,1.0,Grad4);

    VecScale(tmp,-hxyz);

    //Calculate gradient of LDOS1;
    // Derivative of LDOS1 wrt eps = Re [ x^2 wt I/omega epscoef ];
    CmpVecProd(x1,x1,Grad0);
    CmpVecProd(Grad0,epscoef1,tmp1);
    ierr = MatMult(D,tmp1,Grad0); CHKERRQ(ierr);
    ierr = VecPointwiseMult(Grad0,Grad0,weight); CHKERRQ(ierr);
    VecScale(Grad0,1.0/omega1);
    VecScale(Grad0,hxyz);
    ierr = VecPointwiseMult(Grad0,Grad0,vR); CHKERRQ(ierr);

    // gradient = 1/ldos1^n * gr(P3) - n*P3/ldos1^(n+1) * gr(P1)
    VecScale(tmp,1.0/pow(ldos1,ldospowerindex));
    VecWAXPY(tmp1,-1.0*ldospowerindex*beta/pow(ldos1,ldospowerindex+1),Grad0,tmp);

    MatMultTranspose(A,tmp1,vgrad);

    ierr=VecPointwiseMult(vgrad,vgrad,epsgrad); CHKERRQ(ierr);

    KSPSolveTranspose(kspH,vgrad,epsgrad);   //gradient for the Helmholtz filter

    if(minapproach){
      VecScale(vgrad,-1.0*fom*fom);
    }

    // copy vgrad (distributed vector) to a regular array grad;
    ierr = VecToArray(epsgrad,grad,scatter,from,to,vgradlocal,DegFree);
  
    /**START: penalty calculation in objective and gradient**/
    if(penalty){
      int ip;
      for (ip=0;ip<DegFree;ip++) {
        fom = fom - penalty * epsopt[ip] * (1-epsopt[ip]);
        grad[ip] = grad[ip] -1.0 * penalty * (1 - 2 * epsopt[ip]);}}
    /**FINISHED: penalty calculation in objective and gradient**/ 


  }  

  ierr=MatDestroy(&Mone); CHKERRQ(ierr);
  ierr=MatDestroy(&Mthree); CHKERRQ(ierr);  

  if(count==1) OutputVec(PETSC_COMM_WORLD, x1, "sample_", "E1.txt");
  
  count++;

  ierr = VecDestroy(&epsgrad); CHKERRQ(ierr);
  free(epsoptH);

  VecDestroy(&x3);
  VecDestroy(&J3);
  VecDestroy(&weightedJ3);
  VecDestroy(&b3);
  VecDestroy(&epsC);
  VecDestroy(&epsCi);
  VecDestroy(&epsP);
  VecDestroy(&tmp);
  VecDestroy(&tmp1);
  VecDestroy(&tmp2);
  VecDestroy(&E1cube);
  VecDestroy(&u1);
  VecDestroy(&u2);
  VecDestroy(&u3);
  VecDestroy(&Uone);
  VecDestroy(&Utwo);
  VecDestroy(&Uthree);
  VecDestroy(&Grad0);
  VecDestroy(&Grad1);
  VecDestroy(&Grad2);
  VecDestroy(&Grad3);
  VecDestroy(&Grad4);
  VecDestroy(&vgrad);
  
  return fom;
}



