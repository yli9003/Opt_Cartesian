all: eigenCartesian_exec eigenCartesianEpscomplex_exec eigenCartesianTranspose_exec

export PETSC_DIR=/opt/local/petsc-3.6.3
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/opt/local/slepc-3.6.2

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
CLEANFILES = eigenCartesian.o eigenCartesianEpscomplex.o eigenCartesianTranspose.o

eigenCartesian_exec: eigenCartesian.o 
	$(CLINKER) eigenCartesian.o -o eigenCartesian_exec ../CodesForLibrary/libOPT.a $(PETSC_LIB) $(SLEPC_LIB)

eigenCartesian: eigenCartesian.o
	$(CLINKER) -Wall -g -o $@ eigenCartesian.o

eigenCartesianEpscomplex_exec: eigenCartesianEpscomplex.o 
	$(CLINKER) eigenCartesianEpscomplex.o -o eigenCartesianEpscomplex_exec ../CodesForLibrary/libOPT.a $(PETSC_LIB) $(SLEPC_LIB)

eigenCartesianEpscomplex: eigenCartesianEpscomplex.o
	$(CLINKER) -Wall -g -o $@ eigenCartesianEpscomplex.o

eigenCartesianTranspose_exec: eigenCartesianTranspose.o 
	$(CLINKER) eigenCartesianTranspose.o -o eigenCartesianTranspose_exec ../CodesForLibrary/libOPT.a $(PETSC_LIB) $(SLEPC_LIB)

eigenCartesianTranspose: eigenCartesianTranspose.o
	$(CLINKER) -Wall -g -o $@ eigenCartesianTranspose.o
