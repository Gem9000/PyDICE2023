'''
Authors: Jacob Wessel, George Moraites
'''

from pyomo.environ import *
from pyomo.opt import SolverFactory

import numpy as np
import time

from DICE_params import modelParams
from dice_dynamics import simulateDynamics, dumpState

if __name__ == '__main__':
    num_times = 82
    tstep = 5
    t = np.arange(1, num_times+1)

    params = modelParams(num_times, tstep)
    outputType = 1

    start_year = 2020
    final_year = start_year + tstep * num_times
    years = np.linspace(start_year, final_year, num_times+1, dtype=np.int32)

# params._fco22x, params._tatm0 in FAIRmodel
    argsv = [-1.0, outputType, num_times, params._tstep,
             params._al, params._l, params._sigma,
             params._cost1tot, params._eland, params._scale1, params._scale2,
             params._a1, params._a2base, params._a3,
             params._rr, params._gama,
             params._elasmu, params._prstp, params._expcost2,
             params._k0, params._dk, params._pbacktime]

    args = tuple(argsv)

    ###########################################################################
    # Pyomo Model Definition
    ###########################################################################
    model = ConcreteModel()
    model.t = RangeSet(1, num_times)


    # Variables
    model.S = Var(model.t, bounds=(0.1, 0.9))
    model.MIU = Var(model.t, bounds=(0.01, 1.0))






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
    -> .fx used to "fix" variables at certain values.

    **  Upper and lower bounds for stability
    MAT.LO(t)       = 10;
    TATM.UP(t)      = 20;
    TATM.lo(t)      = .5;
    alpha.up(t) = 100;
    alpha.lo(t) = 0.1;
    IRF equation (constrain to be zero at every time point)
    '''



    # Objective Function
    def obj_rule(model):
        return simulateDynamics([value(model.MIU[t]) for t in model.t], *args)
    model.obj = Objective(rule=obj_rule)



    # Constraints
    def miu_constraint_rule(model, t):
        return model.MIU[t] <= 1.0 if t == 1 else model.MIU[t] <= params._limmiu
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

    print("Dumping graphs to file.")
    output = simulateDynamics([value(model.MIU[t]) for t in model.t], *args)
    dumpState(years, output, "./results/base_case_state_post_opt.csv")

    # Put these below equations and dumpState in separate py file to call as functions
    # ### Post-Solution Parameter-Assignment ###
    # self._scc[i]        = -1000 * self._eco2[i] / (.00001 + self._C[i]) # NOTE: THESE (self._eco2[i] and self._C[i]) NEED TO BE MARGINAL VALUES, NOT THE SOLUTIONS THEMSELVES
    # self._ppm[i]        = self._mat[i] / 2.13
    # self._abaterat[i]   = self._abatecost[i] / self._Y[i]
    # self._atfrac2020[i] = (self._mat[i] - self.params._mat0) / (self._ccatot[i] + .00001 - self.params._CumEmiss0)
    # self._atfrac1765[i] = (self._mat[i] - self.params._mateq) / (.00001 + self._ccatot[i])
    # self._FORC_CO2[i]   = self.params._fco22x * ((math.log((self._mat[i] / self.params._mateq)) / math.log(2)))

    print("Completed.")