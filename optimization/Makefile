all: optexec sixmodes_exec c4vopt_exec EPopt_exec solver_exec fiber_exec blochdirac_exec degen_exec c4voptvar_exec penalty_exec chiralopt_exec sweepldos_exec sweepfreq_exec

export PETSC_DIR=/opt/local/petsc-3.6.3
export PETSC_ARCH=arch-opt
export CavityLibPath=../CodesForLibrary
export NloptLibPath=/usr/local/lib/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = opt.o sixldos.o c4vopt.o EPopt.o solver.o fiber.o blochdirac.o degen.o c4voptvar.o penalty.o chiralopt.o sweepldos.o sweepfreq.o

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

fiber_exec: fiber.o
	$(CLINKER) fiber.o -o fiber_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
fiber: fiber.o
	$(CLINKER) -Wall -g -o $@ fiber.o 

blochdirac_exec: blochdirac.o
	$(CLINKER) blochdirac.o -o blochdirac_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
blochdirac: blochdirac.o
	$(CLINKER) -Wall -g -o $@ blochdirac.o 

degen_exec: degen.o
	$(CLINKER) degen.o -o degen_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
degen: degen.o
	$(CLINKER) -Wall -g -o $@ degen.o 

c4voptvar_exec: c4voptvar.o
	$(CLINKER) c4voptvar.o -o c4voptvar_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
c4voptvar: c4voptvar.o
	$(CLINKER) -Wall -g -o $@ c4voptvar.o 

penalty_exec: penalty.o
	$(CLINKER) penalty.o -o penalty_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
penalty: penalty.o
	$(CLINKER) -Wall -g -o $@ penalty.o 

chiralopt_exec: chiralopt.o
	$(CLINKER) chiralopt.o -o chiralopt_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
chiralopt: chiralopt.o
	$(CLINKER) -Wall -g -o $@ chiralopt.o 

sweepldos_exec: sweepldos.o
	$(CLINKER) sweepldos.o -o sweepldos_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
sweepldos: sweepldos.o
	$(CLINKER) -Wall -g -o $@ sweepldos.o 

sweepfreq_exec: sweepfreq.o
	$(CLINKER) sweepfreq.o -o sweepfreq_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
sweepfreq: sweepfreq.o
	$(CLINKER) -Wall -g -o $@ sweepfreq.o 
