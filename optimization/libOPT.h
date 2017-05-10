//define macros;
#undef PI
#define PI 3.14159265358979e+00

typedef struct{
  int Nx;
  int Ny;
  int Nz;
  Vec epsSReal;
  Vec epsFReal;
  double omega;
  Mat M;
  Mat A;
  Vec b;
  Vec J;
  Vec x;
  Vec epspmlQ;
  Vec epsmedium;
  Vec epsDiff;
  Vec epscoef;
  KSP ksp;
  int *its;
  Vec VecVol;
  int outputbase;
} QGroup;

typedef struct{
  int Mx;
  int My;
  int Mz;
  int Lz;
  int Mzslab;
  int Nx;
  int Ny;
  int Nz;
  int Npmlx;
  int Npmly;
  int Npmlz;
  int Nxyz;
  int DegFree;
  double hx;
  double hy;
  double hz;
  double hxyz;
  double sigmax;
  double sigmay;
  double sigmaz;
  int BCPeriod;
  int LowerPMLx;
  int LowerPMLy;
  int LowerPMLz;
  int Nxo;
  int Nyo;
  int Nzo;
  double Qabs;
  char initialdatafile[PETSC_MAX_PATH_LEN];
  char filenameComm[PETSC_MAX_PATH_LEN];
  int outputbase;
  int readlubsfromfile;
} Universals;

typedef struct{
  Mat M;
  Vec epsdiff;
  Vec epsbkg;
  Vec epspmlQ;
  Vec epscoef;
  Vec J;
  Vec weightedJ;
  Vec b;
  Vec x;
  double omega;
} Maxwell;

typedef struct{
  double omega;
  Mat M;
  Mat A;
  Vec x;
  Vec b;
  Vec weightedJ;
  Vec epspmlQ;
  Vec epsmedium;
  Vec epsDiff;
  int *its;
  Vec epscoef;
  Vec vgrad;
  KSP ksp;
} LDOSdataGroup;

typedef struct{
  double omega;
  Mat M;
  Mat A;
  Vec x;
  Vec b;
  Vec weightedJ;
  Vec epspmlQ;
  Vec epsmedium;
  Vec epsDiff;
  int *its;
  Vec epscoef;
  Vec vgrad;
  KSP ksp;
  int nfreq;
} LDOSdataGroupEpsOmega;

typedef struct{
  double omega;
  Mat M;
  Mat A;
  Vec x;
  Vec b;
  Vec weightedJ;
  Vec epspmlQ;
  Vec epsmedium;
  Vec epsDiff;
  int *its;
  Vec epscoef;
  Vec vgrad;
  KSP ksp;
  double optweight;
} LDOSdataGroupConstr;

typedef struct{
  int Nx;
  int Ny;
  int Nz;
  double hxyz;
  Vec epsSReal;
  Vec epsFReal;
  double omega1;
  double omega2;
  double omega3;
  Mat M1;
  Mat M2;
  Mat M3;
  Mat A;
  Vec b1;
  Vec b2;
  Vec x1;
  Vec x2;
  Vec weightedJ1;
  Vec weightedJ2;
  Vec epspmlQ1;
  Vec epspmlQ2;
  Vec epspmlQ3;
  Vec epsmedium1;
  Vec epsmedium2;
  Vec epsmedium3;
  Vec epsDiff1;
  Vec epsDiff2;
  Vec epsDiff3;
  Vec epscoef1;
  Vec epscoef2;
  Vec epscoef3;
  Mat B1;
  Mat B2;
  KSP ksp1;
  KSP ksp2;
  KSP ksp3;
  int *its1;
  int *its2;
  int *its3;
  double p1;
  double p2;
  int outputbase;
} SFGdataGroup;

typedef struct{
  int Nx;
  int Ny;
  int Nz;
  double hxyz;
  Vec epsSReal;
  Vec epsFReal;
  double omega1;
  double omega2;
  double omega3;
  Mat M1;
  Mat M2;
  Mat M3;
  Mat A;
  Vec b1;
  Vec b2;
  Vec x1;
  Vec x2;
  Vec weightedJ1;
  Vec weightedJ2;
  Vec epspmlQ1;
  Vec epspmlQ2;
  Vec epspmlQ3;
  Vec epsmedium1;
  Vec epsmedium2;
  Vec epsmedium3;
  Vec epsDiff1;
  Vec epsDiff2;
  Vec epsDiff3;
  Vec epscoef1;
  Vec epscoef2;
  Vec epscoef3;
  Mat B1;
  Mat B2;
  KSP ksp1;
  KSP ksp2;
  KSP ksp3;
  int *its1;
  int *its2;
  int *its3;
  double p1;
  double p2;
  int outputbase;
  Vec vecNL;
} SFGdataGroupGraphene;


typedef struct{
  double omega;
  Mat M;
  Mat A;
  Vec xL;
  Vec xR;
  Vec bL;
  Vec bR;
  Vec weightedJL;
  Vec weightedJR;
  Vec epspmlQ;
  Vec epsmedium;
  Vec epsDiff;
  int *its;
  Vec epscoef;
  Vec vgrad;
  KSP ksp;
  double chiralweight;
  int fomopt;
} ChiraldataGroup;

typedef struct{
  int Nx;
  int Ny;
  int Nz;
  double hxyz;
  double omega;
  KSP ksp;
  int *its;
  Mat M;
  Vec b;
  Vec x;
  Vec VecFocalpt;
  Vec epsSReal;
  Vec epsFReal;
  Vec epsDiff;
  Vec epsMed;
  Vec epspmlQ;
  Vec epscoef;
  Vec Vecgrad;
  int outputbase;
  Mat Diff;
} LensGroup;

typedef struct{
  double omega;
  Mat M;
  Mat A;
  Vec x;
  Vec b;
  Vec weightedJ;
  Vec epspmlQ;
  Vec epsmedium;
  Vec epsDiff;
  int *its;
  Vec epscoef;
  Vec vgrad;
  KSP ksp;
  int constr;
  double multipurposescalar;
} EPdataGroup;

typedef struct{
  Vec epsSReal;
  Vec epsFReal;
  Mat M1;
  Vec x1;
  Vec weightedJ1;
  Vec b1;
  Vec ej;
  Vec epsI;
  Vec epspmlQ1;
  Vec epsmedium1;
  Vec epscoef1;
  double omega1;
  Mat M3;
  Vec epsIII;
  Vec epspmlQ3;
  Vec epsmedium3;
  Vec epscoef3;
  double omega3;
  KSP ksp1;
  KSP ksp2;
  double ldospowerindex;
  int outputbase;
} THGdataGroup;

typedef struct{
  int Nx;
  int Ny;
  int Nz;
  double hxyz;
  Vec epsSReal;
  Vec epsFReal;
  double omega;
  Mat M;
  Mat A;
  Vec b;
  Vec J;
  Vec x;
  Vec weightedJ;
  Vec epspmlQ;
  Vec epsmedium;
  Vec epsDiff;
  Vec epscoef;
  KSP ksp;
  int *its;
  KSP refksp;
  int *refits;
  double metaphase;
  int trigoption;
  Vec refField;
  Vec refFieldconj;
  //double refmag;
  Vec VecPt;
  int outputbase;
  char *filenameComm;
} MetaSurfGroup;

typedef struct{
  int Nx;
  int Ny;
  int Nz;
  Vec epsSReal;
  Vec epsFReal;
  double omega;
  Mat M;
  Mat A;
  Vec b;
  Vec x;
  Vec epspmlQ;
  Vec epsmedium;
  Vec epsDiff;
  Vec epscoef;
  KSP ksp;
  int *its;
  Vec pvec;
  Vec qvec;
  int outputbase;
  double *refphi;
  double inc_angle;
} Meta;

// from initialize.c
PetscErrorCode readfromflags(Universals *params);

PetscErrorCode setupMatVecs(Universals params, Mat *A, Mat *C, Mat *D, Vec *vR, Vec *vI, Vec *weight, Vec *epsSReal, Vec *epsFReal);

PetscErrorCode makemaxwell(char file[PETSC_MAX_PATH_LEN], Universals params, Mat A, Mat D, Vec vR, Vec weight, Maxwell *fdfd);

// from MoperatorGeneral.c
PetscErrorCode MoperatorGeneral(MPI_Comm comm, Mat *Mout, int Nx, int Ny, int Nz, double hx, double hy, double hz, int bx[2], int by[2], int bz[2], double *muinv,int DimPeriod);

// from MopertorGeneralBloch_XdL.c
PetscErrorCode MoperatorGeneralBloch(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double hx, double hy, double hz, int bx[2], int by[2], int bz[2], double *muinv, int DimPeriod, double beta[3]);

// from SourceGeneration.c
PetscErrorCode SourceSingleSetX(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);

PetscErrorCode SourceSingleSetY(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);

PetscErrorCode SourceSingleSetZ(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);

PetscErrorCode SourceSingleSetGlobal(MPI_Comm comm, Vec J, int globalpos, double amp);

PetscErrorCode SourceDuplicate(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);

PetscErrorCode SourceBlock(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, double hx, double hy, double hz, double lx, double ux, double ly, double uy, double lz, double uz, double amp, int Jdir);

// from PML.c
double sqr(double a);

double pmlsigma(double RRT, double d);

PetscErrorCode EpsPMLFull(MPI_Comm comm, Vec epspml, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPML);

PetscErrorCode MuinvPMLFull(MPI_Comm comm, Vec *muinvout, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPML);

// from PMLGeneral.c
PetscErrorCode EpsPMLGeneral(MPI_Comm comm, Vec epspml, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPMLx, int LowerPMLy, int LowerPMLz);

PetscErrorCode MuinvPMLGeneral(MPI_Comm comm, Vec *muinvout, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPMLx, int LowerPMLy, int LowerPMLz);

// from Eps.c 
PetscErrorCode EpsCombine(Mat D, Vec weight, Vec epspml, Vec epspmlQ, Vec epscoef, double Qabs, double omega, Vec epsilon);

PetscErrorCode ModifyMatDiagonals( Mat M, Mat A, Mat D, Vec epsSReal, Vec epspmlQ, Vec epsmedium, Vec epsC, Vec epsCi, Vec epsP, int Nxyz, double omega, Vec epsilon);

PetscErrorCode  myinterp(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, int Nxo, int Nyo, int Nzo, int Mx, int My, int Mz, int Mzslab, int anisotropic);

PetscErrorCode AddEpsToM( Mat M,  Mat D, Vec epsC, int Nxyz, double omega);

// from MathTools.c
PetscErrorCode CmpVecScale(Vec vin, Vec vout, double a, double b);

PetscErrorCode CmpVecProd(Vec va, Vec vb, Vec vout);

PetscErrorCode CmpVecProdScaF(Vec v1, Vec v2, Vec v);

PetscErrorCode CmpVecDot(Vec v1, Vec v2,  double *preal, double *pimag);

PetscErrorCode  ArrayToVec(double *pt, Vec V);

PetscErrorCode VecToArray(Vec V, double *pt, VecScatter scatter, IS from, IS to, Vec Vlocal, int DegFree);

PetscErrorCode MatSetTwoDiagonals(Mat M, Vec epsC, Mat D, double sign);

PetscErrorCode SolveMatrix(MPI_Comm comm, KSP ksp, Mat M, Vec b, Vec x, int *its);

PetscErrorCode ModifyMatDiag(Mat Mopr, Mat D, Vec epsF, Vec epsDiff, Vec epsMedium, Vec epspmlQ, double omega, int Nx, int Ny, int Nz);

// from MatVecMaker.c
PetscErrorCode GetDotMat(MPI_Comm comm, Mat *Bout, int Nx, int Ny, int Nz);

PetscErrorCode GetProjMat(MPI_Comm comm, Mat *Bout, int c1, int c2, int Nx, int Ny, int Nz);

PetscErrorCode ImagIMat(MPI_Comm comm, Mat *Dout, int N);

PetscErrorCode CongMat(MPI_Comm comm, Mat *Cout, int N);

PetscErrorCode GetWeightVec(Vec weight,int Nx, int Ny, int Nz);

PetscErrorCode GetMediumVec(Vec epsmedium,int Nz, int Mz, double epsair, double epssub);

PetscErrorCode GetMediumVecwithSub(Vec epsmedium,int Nz, int Mz, double epsair, double epssub);

PetscErrorCode GetMediumVecwithSub2(Vec epsmedium,int Nz, int Mz, double epsair, double epssub);

PetscErrorCode GetRealPartVec(Vec vR, int N);

PetscErrorCode AddMuAbsorption(double *muinv, Vec muinvpml, double Qabs, int add);

PetscErrorCode GetUnitVec(Vec ej, int pol, int N);

// from Output.c
PetscErrorCode getreal(const char *flag, double *var, double autoval);

PetscErrorCode getint(const char *flag, int *var, int autoval);

PetscErrorCode  OutputVec(MPI_Comm comm, Vec x, const char *filenameComm, const char *filenameProperty);

PetscErrorCode  OutputMat(MPI_Comm comm, Mat A, const char *filenameComm, const char *filenameProperty);

PetscErrorCode RetrieveVecPoints(Vec x, int Npt, int *Pos, double *ptValues);

PetscErrorCode MyCheckAndOutputInt(PetscBool flg, int CmdVar, const char *strCmdVar, const char *strCmdVarDetail);

PetscErrorCode MyCheckAndOutputDouble(PetscBool flg, double CmdVar, const char *strCmdVar, const char *strCmdVarDetail);

PetscErrorCode MyCheckAndOutputChar(PetscBool flg, char *CmdVar, const char *strCmdVar, const char *strCmdVarDetail);

PetscErrorCode GetIntParaCmdLine(int *ptCmdVar, const char *strCmdVar, const char *strCmdVarDetail);

// from DifferentialOps.c
PetscErrorCode firstorderDeriv(MPI_Comm comm, Mat *Dout, int Nx, int Ny, int Nz, double dh, int c2, int p, int c1);

// from mympisetup.c
int mympisetup();

// from ldos.c
double ldos1constraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

double ldos2constraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from projsimpoverlap.c
double projsimpoverlap(int DegFree,double *epsopt, double *grad, void *data);

// from ldosonly.c and singleldos.c
double ldosonly(int DegFreeAll,double *epsopt, double *grad, void *data);

double minLDOS(int DegFreeAll,double *epsopt, double *grad, void *data);

// from filters.c
void vecdvpow(double *u, double *v, double *dv, int n, int p);

void vectanhproj(double *u, double *v, double *dv, int n, double b, double eta);

void vecHevproj(double *u, double *v, double *dv, int n, double b, double eta);

PetscErrorCode applyfilters(int DegFree, double *epsopt, Vec epsSReal, Vec epsgrad);

PetscErrorCode applyfiltersVer2(int DegFree, double *epsopt, Vec epsSReal, Vec epsgrad, double pSIMP, double bproj, double etaproj);

PetscErrorCode GetH(MPI_Comm comm, Mat *Hout, int mx, int my, int mz, double s, double nR, int dim, KSP *kspHout, PC *pcHout);

PetscErrorCode GetHdummy(MPI_Comm comm, Mat *Hout, int DegFree, KSP *kspHout, PC *pcHout);

PetscErrorCode GetH1d(MPI_Comm comm, Mat *Hout, int DegFree, double s, double nR, KSP *kspHout, PC *pcHout);

PetscErrorCode SolveH(MPI_Comm comm, KSP ksp, Mat H, Vec rhs, Vec sol);

PetscErrorCode RegzProj(int DegFree, double *epsopt,Vec epsSReal,Vec epsgrad,int pSIMP,double bproj,double etaproj,KSP kspH,Mat Hfilt,int *itsH);

PetscErrorCode RegzProjnoH(int DegFree, double *epsopt,Vec epsSReal,Vec epsgrad,int pSIMP,double bproj,double etaproj);

// from alpha.c
double alpha(int DegFree,double *epsopt, double *grad, void *data);

// from FOM.c
double FOM(int DegFree,double *epsopt, double *grad, void *data);

// from ldosconstraint.c
double ldosconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from ldoskconstraint.c
double ldoskconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from ldoskconstraintepsomega
double ldoskconstraintepsomega(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from ldoskminconstraint.c
double ldoskminconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from ldoskconstraintnofilter.c
double ldoskconstraintnofilter(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

double maxminobjfun(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from lens.c
double optfocalpt(int DegFree, double *epsopt, double *grad, void *data);

PetscErrorCode MakeVecFocalpt(Vec VecFocalpt, int Nx, int Ny, int Nz, int ix, int iy, int iz, int ic1, int ic2);

// from c4v.c
PetscErrorCode c4v(MPI_Comm comm, Mat *Aout, int M);

// from EP.c
double EPSOF(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

double EPLDOS(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from eigsolver.c
int eigsolver(Mat M, Vec epsC, Mat D);

// from eigsolver2.c
int eigsolver2(Mat M, Vec epsC, Mat D, int printeigenvec);

// from eigsolvertrans.c
int eigsolvertrans(Mat M, Vec epsC, Mat D, Mat C);

// from thgfom_arbitraryPol.c
double thgfom_arbitraryPol(int DegFree,double *epsopt, double *grad, void *data);

// from ldoskonly.c
double ldoskonly(int DegFree,double *epsopt, double *grad, void *data);

// from chiral.c
double ldoskdiff(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from chiral2.c
double ldoskchiralconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from ldoskmin/maxconstraint.c
double ldoskminconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

double ldoskmaxconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from metasurface.c
double metasurface(int DegFree,double *epsopt, double *grad, void *data);

double metasurfaceminimax(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

double minimaxobjfun(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

PetscErrorCode MakeVecPt(Vec VecPt, int Nx, int Ny, int Nz, int ix, int iy, int iz, int ic);

double transmissionmeta(int DegFree,double *epsopt, double *grad, void *data);

double transmissionmetaconstr(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

double transmissionminimax(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from batchmeta.c
double batchmeta(int DegFree,double *epsopt, double *grad, void *data);

PetscErrorCode makepq_defl(MPI_Comm comm, Vec *pout, Vec *qout, int Nx, int Ny, int Nz, int lx, int ux, int ly, int uy, int lz, int uz, int dir, double theta, double lambda, double refphi);

PetscErrorCode makepq_lens(MPI_Comm comm, Vec *pout, Vec *qout, int Nx, int Ny, int Nz, int lx, int ux, int ly, int uy, int lz, int uz, int dir, double focallength, double lambda, double refphi);

double batchmaximin(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

double maximinobjfun(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

double refphiopt(int ndof,double *refphi, double *grad, void *data);

// from metascatter.c
double metascat(int DegFree,double *epsopt, double *grad, void *data);

double metascatminimax(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from sfg_arbitraryPol.c
double sfg_arbitraryPol(int DegFree,double *epsopt, double *grad, void *data);

// from sfg_singleldos.c
double sfg_singleldos(int DegFree,double *epsopt, double *grad, void *data);

// from sfg_graphene.c
double sfg_graphene(int DegFree,double *epsopt, double *grad, void *data);

// from c3v.c
PetscErrorCode c3vinterp(MPI_Comm comm, Mat *Aout, int Mx, int My, int Nx, int Ny);

// from Qbooster.c
PetscErrorCode boosterinterp(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, int Nxo, int Nyo, int Nzo, int Mx, int My, int Mz, int Mzslab, int Lz, int anisotropic);

PetscErrorCode makethreelayeredepsbkg(Vec epsBkg, int Nx, int Ny, int Nz, int Nzo, int Mz, double epsbkg1, double epsbkg2, double epsbkg3);

PetscErrorCode makethreelayeredepsdiff(Vec epsDiff, int Nx, int Ny, int Nz, int Nzo, int Mz, double epsdiff1, double epsdiff2, double epsdiff3);

PetscErrorCode GetWeightVecGeneralSym(Vec weight,int Nx, int Ny, int Nz, int lx, int ly, int lz);

// from layeredA.c
PetscErrorCode layeredA(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, int nlayers, int Nxo, int Nyo, int* Nzo, int Mx, int My, int* Mz, int Mzslab);

PetscErrorCode layeredepsbkg(Vec epsBkg, int Nx, int Ny, int Nz, int nlayers, int* Nzo, int* Mz, double* epsbkg, double epssub, double epsair, double epsmid);

PetscErrorCode layeredepsdiff(Vec epsDiff, int Nx, int Ny, int Nz, int nlayers, int* Nzo, int* Mz, double* epsdiff, double epssubdiff, double epsairdiff, double epsmiddiff);

// from chi3dsfg.c
double chi3dsfg(int DegFree,double *epsopt, double *grad, void *data);

// from qfactor.c
double qfactor(int DegFree,double *epsopt, double *grad, void *data);

PetscErrorCode makeBlock(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, int lx, int ux, int ly, int uy, int lz, int uz);

// from batchlens.c
PetscErrorCode SourceAngled(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, double hx, double hy, double hz, double lx, double ux, double ly, double uy, double lz, double uz, double amp, int Jdir, double kx, double ky, double kz, int jx0, int jy0, int jz0);

PetscErrorCode makepq_lens_inc(MPI_Comm comm, Vec *pout, Vec *qout, int Nx, int Ny, int Nz, int lx, int ux, int ly, int uy, int lz, int uz, int dir, double fcl, double theta_inc, double lambda, double refphi, int ix0, int iy0, int iz0, int chiefray);

PetscErrorCode mirrorA1d(MPI_Comm comm, Mat *Aout, int Mx, int nlayers);

PetscErrorCode mirrorA2d(MPI_Comm comm, Mat *Aout, int Mx, int My, int nlayers);
