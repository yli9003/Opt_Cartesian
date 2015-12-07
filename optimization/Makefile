all: optexec c4vopt_exec

export PETSC_DIR=/home/zlin/MyLocal/petsc-3.6.0
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/home/zlin/MyLocal/slepc-3.6.0
export CavityLibPath=../CodesForLibrary
export NloptLibPath=/usr/local/lib/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = opt.o c4vopt.o

optexec: opt.o 
	$(CLINKER) opt.o -o optexec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)

opt: opt.o
	$(CLINKER) -Wall -g -o $@ opt.o

c4vopt_exec: c4vopt.o
	$(CLINKER) c4vopt.o -o c4vopt_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
c4vopt: c4vopt.o
	$(CLINKER) -Wall -g -o $@ c4vopt.o
