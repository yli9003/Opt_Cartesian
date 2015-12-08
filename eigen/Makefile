all: eigenCartesian_exec

export PETSC_DIR=/etc/petsc-3.5.3
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/etc/slepc-3.5.4

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${SLEPC_DIR}/conf/slepc_common
CLEANFILES = eigenCartesian.o

eigenCartesian_exec: eigenCartesian.o 
	$(CLINKER) eigenCartesian.o -o eigenCartesian_exec ../CodesForLibrary/libOPT.a $(PETSC_LIB) $(SLEPC_LIB)

eigenCartesian: eigenCartesian.o
	$(CLINKER) -Wall -g -o $@ eigenCartesian.o
