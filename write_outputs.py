'''
Authors: Jacob Wessel
'''

from pyomo.environ import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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


def plot_results(df, save=False, sc_name='unnamedScenario', end_yr = 2100):

    for i in range(1,len(df.columns)):
        if df.columns[i] != 'UTILITY':
            single_plot(df, col=df.columns[i], save=save, sc_name=sc_name, end_yr = end_yr)

    return None

def single_plot(df, col, save=False, sc_name='unnamedScenario', end_yr = 2100):
    
    idx=int((end_yr-df['year'][0])/(df['year'][1]-df['year'][0])) + 1
    
    fig, ax = plt.subplots()
    plt.plot(df['year'][0:idx],df[col][0:idx])
    plt.xlabel('year')
    plt.title(col + ': ' + label_mapper[col])

    if save == True:
        figpath = 'outputs/{}/figures/'.format(sc_name)
        os.makedirs(figpath, exist_ok=True)
        plt.savefig(figpath + col + '.png',dpi=300,bbox_inches='tight',facecolor='w')
    else:
        plt.show()

    return None


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
'MCABATE':'Marginal cost of abatement (2019$ per ton CO2)',
'CCATOT':'Total emissions (GtC)',
'PERIODU':'Single period utility function',
'CPRICE':'Carbon price (2019$/tonCO2)',
'TOTPERIODU':'Period Utility',
'RFACTLONG':'Long interest factor',
'RSHORT':'Real interest rate w/ precautionary (per annum year on year)',
'RLONG':'Real interest rate from year 0 to T',
'FORC':'Increase in radiative forcing (W/m2 from 1765)',
'TATM':'Temp increase in atm. (deg C from 1765)',
'TBOX1':'Increase temp. of box 1 (deg C from 1765)',
'TBOX2':'Increase temp. of box 2 (deg C from 1765)',
'RES0':'C conc. Reservoir 0 (GtC from 1765)',
'RES1':'C conc. Reservoir 1 (GtC from 1765)',
'RES2':'C conc. Reservoir 2 (GtC from 1765)',
'RES3':'C conc. Reservoir 3 (GtC from 1765)',
'MAT':'C conc. increase in atm. (GtC from 1765)',
'CACC':'Accumulated carbon in ocean and other sinks (GtC)',
'IRFt':'IRF100 at time t',
'ECO2':'Total CO2 emissions (GtCO2/yr)',
'ECO2E':'Total CO2e emissions incl. abateable nonCO2 GHG (GtCO2/yr)',
'EIND':'Industrial CO2 emissions (GtCO2/yr)',
'F_GHGabate':'Forcings of abatable nonCO2 GHG',
'alpha':'Carbon decay time scaling factor',
'scc':'Social Cost of Carbon',
'ppm':'Atmospheric Concentration in ppm',
'abaterat':'Cost of emis. reduction / Gross world product net of abatement and damages',
'atfrac2020':'',
'atfrac1765':'',
'FORC_CO2':'CO2 Forcings',
'MCABATE':'Marginal cost of abatement (2019$/tonCO2)',
'CPRICE':'Carbon price (2019$/tonCO2)'}

