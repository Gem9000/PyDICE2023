from pyomo.environ import *
from pyomo.opt import SolverFactory

import numpy as np
import time

from params import DiceParams
#from dice_dynamics import simulateDynamics, dumpState

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
    Fix the argsv in here
    '''

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

    args = tuple(argsv)

    ###########################################################################
    # Pyomo Model Definition
    ###########################################################################
    model = ConcreteModel()

    model.t = RangeSet(1, num_times)

    # Variables
    #model.S = Var(model.t, bounds=(0.1, 0.9))
    #model.MIU = Var(model.t, bounds=(0.01, 1.0))

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

    ###########################################################################
    # Optimization
    ###########################################################################
    opt = SolverFactory('ipopt')

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