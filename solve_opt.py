'''
Authors: Jacob Wessel, George Moraites
'''

from pyomo.environ import *
from pyomo.opt import SolverFactory

import numpy as np
import pandas as pd
import time

from input_params import *
from write_outputs import *
from build_model import *



if __name__ == '__main__':
    
    start = time.time()
    
    ### INPUTS ###
    solver_name = 'ipopt'
    scenario_name = 'defaultScenario'
    write_csv = True
    generate_plots = False
    
    
    #get parameters
    p = getParamsfromfile(input_file = 'inputs/param_inputs.csv')     # get model parameters
    #p = defaultParams()         # use default parameters instead (harc-coded into DICE_params.py)
    simParams(paramObj = p)     # simulate time-varying parameters (pass getParamsfromfile or defaultParams object)

    # build model
    m = build_model(p)
    opt = SolverFactory(solver_name)
    
    # solve model
    sol = opt.solve(m, tee=True)
    

    # results and post-processing / diagnostics
    results = get_df(m)     # converts pyomo results to dataframe
    
    # extra results for dataframe
    # m.scc[t]            = -1000 * m.ECO2[t] / (.00001 + m.C[t]) # NOTE: THESE (m.eco2[t] and m.C[t]) NEED TO BE MARGINAL VALUES, NOT THE SOLUTIONS THEMSELVES
    results['ppm']           = results['MAT'] / 2.13
    results['abaterat']      = results['ABATECOST'] / results['Y']
    results['atfrac2020']    = (results['MAT'] - p._mat0) / (results['CCATOT'] + 0.00001 - p._CumEmiss0)
    results['atfrac1765']    = (results['MAT'] - p._mateq) / (results['CCATOT'] + 0.00001)
    results['FORC_CO2']      = p._fco22x * (np.log(results['MAT'] / p._mateq) / np.log(2))
    results['MCABATE']       = results['pbacktime'] * (results['MIU']**(p._expcost2-1))
    results['CPRICE']        = results['pbacktime'] * (results['MIU']**(p._expcost2-1))
    
    
    if write_csv == True:
        results.to_csv('outputs/{}.csv'.format(scenario_name), index=False)  # export csv of results
    
    
    if generate_plots == True:
        # code to generate diagnostic plot(s)
        pass


    end = time.time()
    print('Completed. Time Elapsed:', end - start)
    
    