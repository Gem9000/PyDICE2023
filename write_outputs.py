'''
Authors: Jacob Wessel
'''

from pyomo.environ import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

label_mapper = {'L':'Population [millions]',
'aL':'Productivity',
'sigma':'CO2-emissions output ratio',
'sigmatot':'GHG-output ratio',
'gA':'Productivity Growth Rate',
'gsig':'Change in sigma (rate of decarbonization)',
'eland':'Emissions from deforestation (GtCO2 per year)',
'cost1tot':'Abatement cost adj. for backstop and sigma',
'pbacktime':'Backstop price 2019$ per ton CO2',
'cpricebase':'Carbon price in base case',
'varpcc':'Variance of per capita consumption',
'rprecaut':'Precautionary rate of return',
'rr':'STP with precautionary factor',
'rr1':'STP without precautionary factor',
'CO2E_GHGabateB':'Abateable non-CO2 GHG emissions base',
'F_Misc':'Non-abateable forcings (GHG and other)',
'emissrat':'Ratio of CO2e to industrial emissions',
'MIU':'Emission Control Rate',
'C':'Consumption (Trill 2019 USD per year)',
'K':'Capital Stock (Trill 2019 USD)',
'CPC':'Per Capita Consumption',
'I':'Investment (Trill 2019 USD per year)',
'S':'Gross Savings Rate as Fraction of Gross World Product',
'Y':'Gross world product net of abatement and damages (Trill 2019 USD/yr)',
'YGROSS':'Gross world product GROSS of abatement and damages (Trill 2019 USD/yr)',
'YNET':'Output net of damages equation (trillions 2019 USD/year)',
'DAMAGES':'Damages (Trill 2019 USD/year)',
'DAMFRAC':'Damages as fraction of gross output',
'ABATECOST':'Cost of emissions reductions (Trill 2019 USD/yr)',
'MCABATE':'',
'CCATOT':'',
'PERIODU':'',
'CPRICE':'',
'TOTPERIODU':'',
'RFACTLONG':'',
'RSHORT':'',
'RLONG':'',
'FORC':'',
'TATM':'',
'TBOX1':'',
'TBOX2':'',
'RES0':'',
'RES1':'',
'RES2':'',
'RES3':'',
'MAT':'',
'CACC':'',
'IRFt':'',
'ECO2':'',
'ECO2E':'',
'EIND':'',
'F_GHGabate':'',
'alpha':''}


df = pd.read_csv('outputs/defaultScenario.csv')

fig, ax = plt.subplots()
plt.plot(df['year'],df['L'])
plt.xlabel('year')
plt.title('L: Population [millions]')
plt.show()