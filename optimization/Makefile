all: optexec sixmodes_exec

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = opt.o sixldos.o

optexec: opt.o 
	$(CLINKER) opt.o -o optexec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)

opt: opt.o
	$(CLINKER) -Wall -g -o $@ opt.o 

sixmodes_exec: sixldos.o
	$(CLINKER) sixldos.o -o sixmodes_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)

sixldos: sixldos.o
	$(CLINKER) -Wall -g -o $@ sixldos.o 