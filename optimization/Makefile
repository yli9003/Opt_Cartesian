all: metasurfexec makegeomexec optexec sixmodes_exec c4vopt_exec EPopt_exec solver_exec fiber_exec blochdirac_exec degen_exec c4voptvar_exec penalty_exec penalty2_exec penalty3_exec penalty4_exec c3vopt_exec c3vsolver_exec chiralopt_exec chiraloptmaxmin_exec sweepldos_exec sweepfreq_exec sfg_exec sfgGeneral_exec graphene_exec test_exec sfgGenerallayered_exec

export PETSC_DIR=/opt/local/petsc-3.6.3
export PETSC_ARCH=arch-opt
export CavityLibPath=../CodesForLibrary
export NloptLibPath=/usr/local/lib/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = metasurf.o makegeom.o opt.o sixldos.o c4vopt.o EPopt.o solver.o fiber.o blochdirac.o degen.o c4voptvar.o penalty.o penalty2.o penalty3.o penalty4.o c3vopt.o c3vsolver.o chiralopt.o chiraloptmaxmin.o sweepldos.o sweepfreq.o sfg.o sfgGeneral.o graphene.o test.o sfgGenerallayered.o

metasurfexec: metasurf.o 
	$(CLINKER) metasurf.o -o metasurfexec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
metasurf: metasurf.o
	$(CLINKER) -Wall -g -o $@ metasurf.o 

makegeomexec: makegeom.o 
	$(CLINKER) makegeom.o -o makegeomexec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
makegeom: makegeom.o
	$(CLINKER) -Wall -g -o $@ makegeom.o 

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

penalty2_exec: penalty2.o
	$(CLINKER) penalty2.o -o penalty2_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
penalty2: penalty2.o
	$(CLINKER) -Wall -g -o $@ penalty2.o 

penalty3_exec: penalty3.o
	$(CLINKER) penalty3.o -o penalty3_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
penalty3: penalty3.o
	$(CLINKER) -Wall -g -o $@ penalty3.o 

penalty4_exec: penalty4.o
	$(CLINKER) penalty4.o -o penalty4_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
penalty4: penalty4.o
	$(CLINKER) -Wall -g -o $@ penalty4.o 

c3vopt_exec: c3vopt.o
	$(CLINKER) c3vopt.o -o c3vopt_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
c3vopt: c3vopt.o
	$(CLINKER) -Wall -g -o $@ c3vopt.o 

c3vsolver_exec: c3vsolver.o
	$(CLINKER) c3vsolver.o -o c3vsolver_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
c3vsolver: c3vsolver.o
	$(CLINKER) -Wall -g -o $@ c3vsolver.o 

chiralopt_exec: chiralopt.o
	$(CLINKER) chiralopt.o -o chiralopt_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
chiralopt: chiralopt.o
	$(CLINKER) -Wall -g -o $@ chiralopt.o 

chiraloptmaxmin_exec: chiraloptmaxmin.o
	$(CLINKER) chiraloptmaxmin.o -o chiraloptmaxmin_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
chiraloptmaxmin: chiraloptmaxmin.o
	$(CLINKER) -Wall -g -o $@ chiraloptmaxmin.o 

sweepldos_exec: sweepldos.o
	$(CLINKER) sweepldos.o -o sweepldos_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
sweepldos: sweepldos.o
	$(CLINKER) -Wall -g -o $@ sweepldos.o 

sweepfreq_exec: sweepfreq.o
	$(CLINKER) sweepfreq.o -o sweepfreq_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
sweepfreq: sweepfreq.o
	$(CLINKER) -Wall -g -o $@ sweepfreq.o 

sfg_exec: sfg.o
	$(CLINKER) sfg.o -o sfg_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
sfg: sfg.o
	$(CLINKER) -Wall -g -o $@ sfg.o 

sfgGeneral_exec: sfgGeneral.o
	$(CLINKER) sfgGeneral.o -o sfgGeneral_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
sfgGeneral: sfgGeneral.o
	$(CLINKER) -Wall -g -o $@ sfgGeneral.o 

graphene_exec: graphene.o
	$(CLINKER) graphene.o -o graphene_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
graphene: graphene.o
	$(CLINKER) -Wall -g -o $@ graphene.o 

test_exec: test.o
	$(CLINKER) test.o -o test_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
test: test.o
	$(CLINKER) -Wall -g -o $@ test.o 

sfgGenerallayered_exec: sfgGenerallayered.o
	$(CLINKER) sfgGenerallayered.o -o sfgGenerallayered_exec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)
sfgGenerallayered: sfgGenerallayered.o
	$(CLINKER) -Wall -g -o $@ sfgGenerallayered.o 


