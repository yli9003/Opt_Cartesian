all: MoperatorGeneral.o MoperatorGeneralBloch_XdL.o SourceGeneration.o PML.o Eps.o MathTools.o MatVecMaker.o Output.o DifferentialOps.o mympisetup.o ldos.o projsimpoverlap.o singleldos.o ldosonly.o filters.o alpha.o FOM.o ldosconstraint.o lens.o c4v.o EP.o eigsolver.o

export PETSC_DIR=/etc/petsc-3.5.3
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/etc/slepc-3.5.4

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${SLEPC_DIR}/conf/slepc_common
CLEANFILES =  MoperatorGeneral.o MoperatorGeneralBloch_XdL.o SourceGeneration.o PML.o Eps.o MathTools.o MatVecMaker.o Output.o DifferentialOps.o mympisetup.o ldos.o projsimpoverlap.o singleldos.o ldosonly.o filters.o alpha.o FOM.o ldosconstraint.o lens.o c4v.o EP.o eigsolver.o

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
DifferentialOps:DifferentialOps.o
	$(CLINKER) -Wall -g -o $@ DifferentialOps.o
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
c4v:c4v.o
	$(CLINKER) -Wall -g -o $@ c4v.o
EP:EP.o
	$(CLINKER) -Wall -g -o $@ EP.o
eigsolver:eigsolver.o
	$(CLINKER) -Wall -g -o $@ eigsolver.o