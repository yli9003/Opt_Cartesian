all: MoperatorGeneral.o MoperatorGeneralBloch_XdL.o SourceGeneration.o PML.o Eps.o MathTools.o MatVecMaker.o Output.o mympisetup.o ldos.o projsimpoverlap.o singleldos.o ldosonly.o filters.o alpha.o FOM.o ldosconstraint.o lens.o eigsolver.o

export PETSC_DIR=/home/zlin/MyLocal/petsc-3.6.0
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/home/zlin/MyLocal/slepc-3.6.0

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
CLEANFILES =  MoperatorGeneral.o MoperatorGeneralBloch_XdL.o SourceGeneration.o PML.o Eps.o MathTools.o MatVecMaker.o Output.o mympisetup.o ldos.o projsimpoverlap.o singleldos.o ldosonly.o filters.o alpha.o FOM.o ldosconstraint.o lens.o eigsolver.o

MoperatorGeneral: MoperatorGeneral.o
	$(CLINKER) -Wall -g -o $@ MoperatorGeneral.o  
MoperatorGeneralBloch: MoperatorGeneralBloch_XdL.o
	$(CLINKER) -Wall -g -o $@ MoperatorGeneralBloch_XdL.o
SourceGeneration: SourceGeneration.o
	$(CLINKER) -Wall -g -o $@ SourceGeneration.o  
PML:PML.o
	$(CLINKER) -Wall -g -o $@ PML.o  
Eps:Eps.o
	$(CLINKER) -Wall -g -o $@ Eps.o   
MathTools:MathTools.o
	$(CLINKER) -Wall -g -o $@ MathTools.o
MatVecMaker:MatVecMaker.o
	$(CLINKER) -Wall -g -o $@ MatVecMaker.o  
Output:Output.o
	$(CLINKER) -Wall -g -o $@ Output.o  
mympisetup:mympisetup.o
	$(CLINKER) -Wall -g -o $@ mympisetup.o
ldos:ldos.o
	$(CLINKER) -Wall -g -o $@ ldos.o
projsimpoverlap:projsimpoverlap.o
	$(CLINKER) -Wall -g -o $@ projsimpoverlap.o
singleldos:singleldos.o
	$(CLINKER) -Wall -g -o $@ singleldos.o
ldosonly:ldosonly.o
	$(CLINKER) -Wall -g -o $@ ldosonly.o
filters:filters.o
	$(CLINKER) -Wall -g -o $@ filters.o
alpha:alpha.o
	$(CLINKER) -Wall -g -o $@ alpha.o
FOM:FOM.o
	$(CLINKER) -Wall -g -o $@ FOM.o
ldosconstraint:ldosconstraint.o
	$(CLINKER) -Wall -g -o $@ ldosconstraint.o
lens:lens.o
	$(CLINKER) -Wall -g -o $@ lens.o
eigsolver:eigsolver.o
	$(CLINKER) -Wall -g -o $@ eigsolver.o
