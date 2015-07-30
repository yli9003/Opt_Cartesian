all: eigenCartesian_exec

export PETSC_DIR=/home/zlin/MyLocal/petsc-3.6.0
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/home/zlin/MyLocal/slepc-3.6.0

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
CLEANFILES = eigenCartesian.o

eigenCartesian_exec: eigenCartesian.o 
	$(CLINKER) eigenCartesian.o -o eigenCartesian_exec ../CodesForLibrary/libOPT.a $(PETSC_LIB) $(SLEPC_LIB)

eigenCartesian: eigenCartesian.o
	$(CLINKER) -Wall -g -o $@ eigenCartesian.o
