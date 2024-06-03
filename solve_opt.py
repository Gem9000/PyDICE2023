'''
Authors: George Moraites, Jacob Wessel
'''

from pyomo.environ import *
from pyomo.opt import SolverFactory

import numpy as np
import time

from params import DiceParams
from dice_dynamics import simulateDynamics, dumpState

if __name__ == '__main__':
    num_times = 81
    tstep = 5.0

    t = np.arange(1, num_times+1)

    p = DiceParams(num_times, tstep)
    outputType = 1

    start_year = 2020
    final_year = start_year + p._tstep * num_times
    years = np.linspace(start_year, final_year, num_times+1, dtype=np.int32)

    scc_index = 0
    c_bump = 0.0
    e_bump = 0.0
    
    '''
    Fix the argsv in here. Original:
        argsv = [-1.0, outputType, num_times,
             p._tstep,
             p._al, p._l, p._sigma, p._cumetree, p._forcoth,
             p._cost1, p._etree, p._scale1, p._scale2,
             p._ml0, p._mu0, p._mat0, p._cca0,
             p._a1, p._a2, p._a3,
             p._c1, p._c3, p._c4,
             p._b11, p._b12, p._b21, p._b22, p._b32, p._b23, p._b33,
             p._fco22x, p._t2xco2,
             p._rr, p._gama,
             p._tocean0, p._tatm0, p._elasmu, p._prstp, p._expcost2,
             p._k0, p._dk, p._pbacktime,
             scc_index, e_bump, c_bump]
    '''
# p._fco22x, p._tatm0 in FAIRmodel
    argsv = [-1.0, outputType, num_times, p._tstep,
             p._al, p._l, p._sigma,
             p._cost1tot, p._eland, p._scale1, p._scale2,
             p._a1, p._a2base, p._init__a3,
             p._rr, p._gama,
             p._elasmu, p._prstp, p._expcost2,
             p._k0, p._dk, p._pbacktime,
             scc_index, e_bump, c_bump]

    args = tuple(argsv)

    ###########################################################################
    # Pyomo Model Definition
    ###########################################################################
    model = ConcreteModel()

    model.t = RangeSet(1, num_times)

    # Variables
    model.S = Var(model.t, bounds=(0.1, 0.9))
    model.MIU = Var(model.t, bounds=(0.01, 1.0))

    # Objective Function
    def obj_rule(model):
        return simulateDynamics([value(model.MIU[t]) for t in model.t], *args)
    model.obj = Objective(rule=obj_rule)

    # Constraints
    def miu_constraint_rule(model, t):
        return model.MIU[t] <= 1.0 if t == 1 else model.MIU[t] <= p._limmiu
    model.miu_constraint = Constraint(model.t, rule=miu_constraint_rule)

    def savings_constraint_rule(model, t):
        return (model.S[t] >= 0.1) if t > num_times - 10 else (model.S[t] >= 0.1 and model.S[t] <= 0.9)
    model.savings_constraint = Constraint(model.t, rule=savings_constraint_rule)

    # def capital_stock_constraint(model,t):
    #     K[i] <= (1.0 - dk)**tstep * K[i-1] + tstep * I[i]
    
    '''PARAMETERS
    L(t)           Level of population and labor
    aL(t)          Level of total factor productivity
    sigma(t)       CO2-emissions output ratio
    sigmatot(t)    GHG-output ratio
    gA(t)          Growth rate of productivity from
    gL(t)          Growth rate of labor and population
    gcost1         Growth of cost factor
    gsig(t)        Change in sigma (rate of decarbonization)
    eland(t)       Emissions from deforestation (GtCO2 per year)
    cost1tot(T)    Abatement cost adjusted for backstop and sigma
    pbacktime(t)   Backstop price 2019$ per ton CO2
    optlrsav       Optimal long-run savings rate used for transversality
    scc(t)         Social cost of carbon
    cpricebase(t)  Carbon price in base case
    ppm(t)         Atmospheric concentrations parts per million
    atfrac2020(t)  Atmospheric share since 2020
    atfrac1765(t)  Atmospheric fraction of emissions since 1765
    abaterat(t)    Abatement cost per net output
    miuup(t)       Upper bound on miu
    gbacktime(t)   Decline rate of backstop price
    '''

    ''' Precautionary dynamic parameters
    varpcc(t)         Variance of per capita consumption 
    rprecaut(t)       Precautionary rate of return
    RR(t)             STP with precautionary factor
    RR1(t)            STP factor without precautionary factor;
    '''


    '''
    EQUATIONS
    FORCE(t)        Radiative forcing equation
    RES0LOM(t)      Reservoir 0 law of motion
    RES1LOM(t)      Reservoir 1 law of motion
    RES2LOM(t)      Reservoir 2 law of motion
    RES3LOM(t)      Reservoir 3 law of motion
    MMAT(t)         Atmospheric concentration equation
    Cacceq(t)       Accumulated carbon in sinks equation
    TATMEQ(t)       Temperature-climate equation for atmosphere
    TBOX1EQ(t)      Temperature box 1 law of motion
    TBOX2EQ(t)      Temperature box 2 law of motion
    IRFeqLHS(t)     Left-hand side of IRF100 equation
    IRFeqRHS(t)     Right-hand side of IRF100 equation
    '''

    ###########################################################################
    # Optimization
    ###########################################################################
    opt = SolverFactory('gurobi')

    # Solve the model
    start = time.time()
    results = opt.solve(model, tee=True)
    end = time.time()
    print("Time Elapsed:", end - start)

    ###########################################################################
    # Post-Processing
    ###########################################################################
    outputType = 0
    argsv[1] = outputType
    args = tuple(argsv)

    print("Dumping graphs to file.")
    output = simulateDynamics([value(model.MIU[t]) for t in model.t], *args)
    dumpState(years, output, "./results/base_case_state_post_opt.csv")

    print("Completed.")


    '''
* Various control rate limits, initial and terminal conditions
miu.up(t)       = miuup(t);
K.LO(t)         = 1;
C.LO(t)         = 2;
CPC.LO(t)       = .01;
RFACTLONG.lo(t) =.0001;
*set lag10(t) ;
*lag10(t)                =  yes$(t.val gt card(t)-10);
*S.FX(lag10(t))          = optlrsav;
s.fx(t)$(t.val > 37)    =.28;
ccatot.fx(tfirst)       = CumEmiss0;
K.FX(tfirst)            = k0;
F_GHGabate.fx(tfirst)   = F_GHGabate2020;
RFACTLONG.fx(tfirst)    = 1000000;

**  Upper and lower bounds for stability
MAT.LO(t)       = 10;
TATM.UP(t)      = 20;
TATM.lo(t)      = .5;
alpha.up(t) = 100;
alpha.lo(t) = 0.1;

    '''