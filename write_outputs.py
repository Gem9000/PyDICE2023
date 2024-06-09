'''
Authors: Jacob Wessel
'''

from pyomo.environ import *

import pandas as pd
import numpy as np

def get_df(model): #needs solved pyomo model passed to it
    
    model = model
    mvars = [i for i in model.component_objects(ctype=Var)]
    varnames = [i.name for i in mvars]
    params = [i for i in model.component_objects(ctype=Param)]
    paramnames = [i.name for i in params]

    # save results to dataframe
    df = pd.DataFrame()
    
    for p in range(len(params)):
        df[paramnames[p]] = [params[p][i] for i in params[p]]
        
    for v in range(len(mvars)):
        if varnames[v] == 'UTILITY': #utility is just a scalar, so only populate first row
            df[varnames[v]] = [model.UTILITY.value]+[np.nan for i in range(len(params[0])-1)]
        else:
            df[varnames[v]] = [mvars[v][i].value for i in mvars[v]]
    
    return df