all: optexec sixmodes_exec c4vopt_exec EPopt_exec solver_exec

export PETSC_DIR=/opt/local/petsc-3.6.3
export PETSC_ARCH=arch-opt
export CavityLibPath=../CodesForLibrary
export NloptLibPath=/usr/local/lib/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = opt.o sixldos.o c4vopt.o EPopt.o solver.o

optexec: opt.o 
	$(CLINKER) opt.o -o optexec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
opt: opt.o
	$(CLINKER) -Wall -g -o $@ opt.o 

sixmodes_exec: sixldos.o
	$(CLINKER) sixldos.o -o sixmodes_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
sixldos: sixldos.o
	$(CLINKER) -Wall -g -o $@ sixldos.o 

c4vopt_exec: c4vopt.o
	$(CLINKER) c4vopt.o -o c4vopt_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
c4vopt: c4vopt.o
	$(CLINKER) -Wall -g -o $@ c4vopt.o 

EPopt_exec: EPopt.o
	$(CLINKER) EPopt.o -o EPopt_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
EPopt: EPopt.o
	$(CLINKER) -Wall -g -o $@ EPopt.o 

solver_exec: solver.o
	$(CLINKER) solver.o -o solver_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
solver: solver.o
	$(CLINKER) -Wall -g -o $@ solver.o 
