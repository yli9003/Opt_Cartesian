all: MoperatorGeneral.o MoperatorGeneralBloch_XdL.o SourceGeneration.o PML.o Eps.o MathTools.o MatVecMaker.o Output.o DifferentialOps.o mympisetup.o ldos.o projsimpoverlap.o singleldos.o ldosonly.o filters.o alpha.o FOM.o thgfom_arbitraryPol.o ldosconstraint.o ldoskconstraint.o ldoskconstraintepsomega.o ldoskconstraintnofilter.o ldoskonly.o lens.o c4v.o EP.o eigsolver.o eigsolvertrans.o chiral.o chiral2.o ldoskmaxconstraint.o ldoskminconstraint.o sfg_arbitraryPol.o sfg_singleldos.o sfg_graphene.o c3v.o

export PETSC_DIR=/opt/local/petsc-3.6.3
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/opt/local/slepc-3.6.2

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
CLEANFILES =  MoperatorGeneral.o MoperatorGeneralBloch_XdL.o SourceGeneration.o PML.o Eps.o MathTools.o MatVecMaker.o Output.o DifferentialOps.o mympisetup.o ldos.o projsimpoverlap.o singleldos.o ldosonly.o filters.o alpha.o FOM.o thgfom_arbitraryPol.o ldosconstraint.o ldoskconstraint.o ldoskconstraintepsomega.o ldoskconstraintnofilter.o ldoskonly.o lens.o c4v.o EP.o eigsolver.o eigsolvertrans.o chiral.o chiral2.o ldoskmaxconstraint.o ldoskminconstraint.o sfg_arbitraryPol.o sfg_singleldos.o sfg_graphene.o c3v.o

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
thgfom_arbitraryPol:thgfom_arbitraryPol.o
	$(CLINKER) -Wall -g -o $@ thgfom_arbitraryPol.o
ldosconstraint:ldosconstraint.o
	$(CLINKER) -Wall -g -o $@ ldosconstraint.o
ldoskconstraint:ldoskconstraint.o
	$(CLINKER) -Wall -g -o $@ ldoskconstraint.o
ldoskconstraintepsomega:ldoskconstraintepsomega.o
	$(CLINKER) -Wall -g -o $@ ldoskconstraintepsomega.o
ldoskconstraintnofilter:ldoskconstraintnofilter.o
	$(CLINKER) -Wall -g -o $@ ldoskconstraintnofilter.o
ldoskonly:ldoskonly.o
	$(CLINKER) -Wall -g -o $@ ldoskonly.o
lens:lens.o
	$(CLINKER) -Wall -g -o $@ lens.o
c4v:c4v.o
	$(CLINKER) -Wall -g -o $@ c4v.o
EP:EP.o
	$(CLINKER) -Wall -g -o $@ EP.o
eigsolver:eigsolver.o
	$(CLINKER) -Wall -g -o $@ eigsolver.o
eigsolvertrans:eigsolvertrans.o
	$(CLINKER) -Wall -g -o $@ eigsolvertrans.o
chiral:chiral.o
	$(CLINKER) -Wall -g -o $@ chiral.o
chiral2:chiral2.o
	$(CLINKER) -Wall -g -o $@ chiral2.o
ldoskmaxconstraint:ldoskmaxconstraint.o
	$(CLINKER) -Wall -g -o $@ ldoskmaxconstraint.o
ldoskminconstraint:ldoskminconstraint.o
	$(CLINKER) -Wall -g -o $@ ldoskminconstraint.o
sfg_arbitraryPol:sfg_arbitraryPol.o
	$(CLINKER) -Wall -g -o $@ sfg_arbitraryPol.o
sfg_singleldos:sfg_singleldos.o
	$(CLINKER) -Wall -g -o $@ sfg_singleldos.o
sfg_graphene:sfg_graphene.o
	$(CLINKER) -Wall -g -o $@ sfg_graphene.o
c3v:c3v.o
	$(CLINKER) -Wall -g -o $@ c3v.o
