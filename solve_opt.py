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
    
    #################################
    ### INPUTS AND CUSTOMIZATIONS ###
    #################################
    
    # solver name (only tested for IPOPT3.11.1, must be able to handle NLP)
    solver_name = 'ipopt'
    
    # name of scenario
    scenario_name = 'defaultScenario'
    
    # save csv of outputs, uses scenario_name to name file
    write_csv = True
    
    # if true, will run the generate_plots() function within write_outputs
    generate_plots = False
    
    # if true, saves each plot generated within a new folder named after scenario_name.
    # if false, prints each plot to stdout
    save_plots = False
    
    # get parameters
    p = getParamsfromfile(input_file = 'inputs/param_inputs.csv')
    #p = defaultParams()         # use default parameters instead (hard-coded into DICE_params.py)
    
    simParams(paramObj = p)     # simulate time-varying parameters (pass getParamsfromfile or defaultParams object)

    #################################
    ##### BUILD AND SOLVE MODEL #####
    #################################

    m = build_model(p)
    
    opt = SolverFactory(solver_name)

    sol = opt.solve(m, tee=True)
    
    #################################
    ##### RESULTS AND PROCESSING ####
    #################################

    results = get_df(m)     # converts pyomo results to dataframe
    
    # extra results for dataframe
    results['ppm']           = results['MAT'] / 2.13
    results['abaterat']      = results['ABATECOST'] / results['Y']
    results['atfrac2020']    = (results['MAT'] - p._mat0) / (results['CCATOT'] + 0.00001 - p._CumEmiss0)
    results['atfrac1765']    = (results['MAT'] - p._mateq) / (results['CCATOT'] + 0.00001)
    results['FORC_CO2']      = p._fco22x * (np.log(results['MAT'] / p._mateq) / np.log(2))
    results['MCABATE']       = results['pbacktime'] * (results['MIU']**(p._expcost2-1))
    results['CPRICE']        = results['pbacktime'] * (results['MIU']**(p._expcost2-1))
    
    # social cost of carbon
    # (m.eco2[t] and m.C[t] need to be marginal (dual) values based on their respective constraints, not primal variable values)
    eco2_dual = np.array([m.dual.get(m._ECO2EQ[i]) for i in m._ECO2EQ])
    c_dual = np.array([m.dual.get(m._CEQ[i]) for i in m._CEQ])
    results['scc']           = -1000 * eco2_dual / (.00001 + c_dual)  # see include/def-opt-b-4-3-10.gms



    if write_csv == True:
        results.to_csv('outputs/{}.csv'.format(scenario_name), index=False)  # export csv of results
    
    
    if generate_plots == True:
        plot_results(results, save=False, sc_name=scenario_name)
    # note: can generate a single plot by calling single_plot(results, col='column_to_plot', save=False, sc_name='unnamedScenario')

    end = time.time()
    print('Completed. Time Elapsed:', end - start)
    

 
    
'''
results discrepancies:
     scc[0] ~= 50 $/tCO2 in GAMS
     scc[0] = 9.218076
    
    possible stopping years: 2200, 2300, 
    
    
'''
    
    
    
    
    
 
    
 