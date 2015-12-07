//define macros;
#undef PI
#define PI 3.14159265358979e+00

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
} LensGroup;

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

PetscErrorCode SourceBlock(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, double hx, double hy, double hz, double lx, double ux, double ly, double uy, double lz, double uz, double amp);

// from PML.c
double pmlsigma(double RRT, double d);

PetscErrorCode EpsPMLFull(MPI_Comm comm, Vec epspml, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPML);

PetscErrorCode MuinvPMLFull(MPI_Comm comm, Vec *muinvout, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPML);

// from Eps.c 
PetscErrorCode EpsCombine(Mat D, Vec weight, Vec epspml, Vec epspmlQ, Vec epscoef, double Qabs, double omega, Vec epsilon);

PetscErrorCode ModifyMatDiagonals( Mat M, Mat A, Mat D, Vec epsSReal, Vec epspmlQ, Vec epsmedium, Vec epsC, Vec epsCi, Vec epsP, int Nxyz, double omega, Vec epsilon);

PetscErrorCode  myinterp(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, int Nxo, int Nyo, int Nzo, int Mx, int My, int Mz, int Mzslab, int anisotropic);

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

PetscErrorCode ImagIMat(MPI_Comm comm, Mat *Dout, int N);

PetscErrorCode CongMat(MPI_Comm comm, Mat *Cout, int N);

PetscErrorCode GetWeightVec(Vec weight,int Nx, int Ny, int Nz);

PetscErrorCode GetMediumVec(Vec epsmedium,int Nz, int Mz, double epsair, double epssub);

PetscErrorCode GetMediumVecwithSub(Vec epsmedium,int Nz, int Mz, double epsair, double epssub);

PetscErrorCode GetRealPartVec(Vec vR, int N);

PetscErrorCode AddMuAbsorption(double *muinv, Vec muinvpml, double Qabs, int add);

PetscErrorCode GetUnitVec(Vec ej, int pol, int N);

// from Output.c
PetscErrorCode  OutputVec(MPI_Comm comm, Vec x, const char *filenameComm, const char *filenameProperty);

PetscErrorCode  OutputMat(MPI_Comm comm, Mat A, const char *filenameComm, const char *filenameProperty);

PetscErrorCode RetrieveVecPoints(Vec x, int Npt, int *Pos, double *ptValues);

PetscErrorCode MyCheckAndOutputInt(PetscBool flg, int CmdVar, const char *strCmdVar, const char *strCmdVarDetail);

PetscErrorCode MyCheckAndOutputDouble(PetscBool flg, double CmdVar, const char *strCmdVar, const char *strCmdVarDetail);

PetscErrorCode MyCheckAndOutputChar(PetscBool flg, char *CmdVar, const char *strCmdVar, const char *strCmdVarDetail);

PetscErrorCode GetIntParaCmdLine(int *ptCmdVar, const char *strCmdVar, const char *strCmdVarDetail);

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

PetscErrorCode SolveH(MPI_Comm comm, KSP ksp, Mat H, Vec rhs, Vec sol);

PetscErrorCode RegzProj(int DegFree, double *epsopt,Vec epsSReal,Vec epsgrad,int pSIMP,double bproj,double etaproj,KSP kspH,Mat Hfilt,int *itsH);

// from alpha.c
double alpha(int DegFree,double *epsopt, double *grad, void *data);

// from FOM.c
double FOM(int DegFree,double *epsopt, double *grad, void *data);

// from ldosconstraint.c
double ldosconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

double maxminobjfun(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);

// from lens.c
double optfocalpt(int DegFree, double *epsopt, double *grad, void *data);

PetscErrorCode MakeVecFocalpt(Vec VecFocalpt, int Nx, int Ny, int Nz, int ix, int iy, int iz, int ic1, int ic2);

// from eigsolver.c
int eigsolver(Mat M, Vec epsC, Mat D);

// from thgfom.c
double thgfom(int DegFree,double *epsopt, double *grad, void *data);

// from c4v.c
PetscErrorCode c4v(MPI_Comm comm, Mat *Aout, int M);
