all: optexec

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = opt.o 

optexec: opt.o 
	$(CLINKER) opt.o -o optexec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)

opt: opt.o
	$(CLINKER) -Wall -g -o $@ opt.o 
