'''
Authors: Jacob Wessel, George Moraites
'''

import numpy as np
import pandas as pd




class getParamsfromfile():
    '''
    reads in csv of specific format
    with param names in first column
    and parameter values in column "col"
    (default is second column, or first column of data)
    '''
    
    def __init__(self, input_file = 'inputs/param_inputs.csv', col=1):
        
        self._csv_data          = pd.read_csv(input_file, header=None,index_col=0)  #CSV containing parameters

        self._region            = str(self._csv_data.loc['region',col])               #World region (arbitrary)
        
        # numtimes default/max = 81, tstep = 5 years. Other values not tested.
        self._tstep             = int(self._csv_data.loc['tstep',col])                #Years in period
        self._numtimes          = int(self._csv_data.loc['numtimes',col])             #Number of future model periods
        self._yr0               = int(self._csv_data.loc['yr0',col])                  #Calendar year that corresponds to model year zero ##
        self._finalyr           = self._yr0 + self._tstep * (self._numtimes)          #Final calendar year (computed)

        self._emshare0          = float(self._csv_data.loc['emshare0',col])           #Carbon emissions share into Reservoir 0
        self._emshare1          = float(self._csv_data.loc['emshare1',col])           #Carbon emissions share into Reservoir 1
        self._emshare2          = float(self._csv_data.loc['emshare2',col])           #Carbon emissions share into Reservoir 2
        self._emshare3          = float(self._csv_data.loc['emshare3',col])           #Carbon emissions share into Reservoir 3
        self._tau0              = float(self._csv_data.loc['tau0',col])               #Decay time constant for R0  (year)
        self._tau1              = float(self._csv_data.loc['tau1',col])               #Decay time constant for R1  (year)
        self._tau2              = float(self._csv_data.loc['tau2',col])               #Decay time constant for R2  (year)
        self._tau3              = float(self._csv_data.loc['tau3',col])               #Decay time constant for R3  (year)
    
        self._teq1              = float(self._csv_data.loc['teq1',col])               #Thermal equilibration parameter for box 1 (m^2 per KW)
        self._teq2              = float(self._csv_data.loc['teq2',col])               #Thermal equilibration parameter for box 2 (m^2 per KW)
        self._d1                = float(self._csv_data.loc['d1',col])                 #Thermal response timescale for deep ocean (year)
        self._d2                = float(self._csv_data.loc['d2',col])                 #Thermal response timescale for upper ocean (year)

        self._irf0              = float(self._csv_data.loc['irf0',col])               #Pre-industrial IRF100 (year)
        self._irC               = float(self._csv_data.loc['irC',col])                #Increase in IRF100 with cumulative carbon uptake (years per GtC)
        self._irT               = float(self._csv_data.loc['irT',col])                #Increase in IRF100 with warming (years per degree K)   
        self._fco22x            = float(self._csv_data.loc['fco22x',col])             #Forcings of equilibrium CO2 doubling (Wm-2)

        # INITIAL CONDITIONS CALIBRATED TO HISTORY
        self._mat0              = float(self._csv_data.loc['mat0',col])               #Initial concentration in atmosphere in 2020 (GtC)
        self._res00             = float(self._csv_data.loc['res00',col])              #Initial concentration in Reservoir 0 in 2020
        self._res10             = float(self._csv_data.loc['res10',col])              #Initial concentration in Reservoir 1 in 2020
        self._res20             = float(self._csv_data.loc['res20',col])              #Initial concentration in Reservoir 2 in 2020
        self._res30             = float(self._csv_data.loc['res30',col])              #Initial concentration in Reservoir 3 in 2020

        self._mateq             = float(self._csv_data.loc['mateq',col])              #Equilibrium concentration atmosphere
        self._tbox10            = float(self._csv_data.loc['tbox10',col])             #Initial temperature box 1 change in 2020
        self._tbox20            = float(self._csv_data.loc['tbox20',col])             #Initial temperature box 2 change in 2020
        self._tatm0             = float(self._csv_data.loc['tatm0',col])              #Initial atmospheric temperature change in 2020 

        # Parameters for non-industrial emission
        # NOTE: Default abateable share of non-CO2 GHG is 65%
        # (pre-processed in GAMS version, changeable here)
        self._eland0            = float(self._csv_data.loc['eland0',col])             # Carbon emissions from land 2015 (GtCO2 per year)
        self._deland            = float(self._csv_data.loc['deland',col])             # Decline rate of land emissions (per period)
        self._F_Misc2020        = float(self._csv_data.loc['F_Misc2020',col])         # Non-abatable forcings 2020
        self._F_Misc2100        = float(self._csv_data.loc['F_Misc2100',col])         # Non-abatable forcings 2100
        self._F_GHGabate2020    = float(self._csv_data.loc['F_GHGabate2020',col])     # Forcings of abatable nonCO2 GHG
        self._F_GHGabate2100    = float(self._csv_data.loc['F_GHGabate2100',col])     # Forcings of abatable nonCO2 GHG
        self._nonco2frac        = float(self._csv_data.loc['nonco2frac',col])         # Abateable share of nonco2 GHGs
        self._ECO2eGHGB2020act  = float(self._csv_data.loc['ECO2eGHGB2020act',col])   # Emis of nonCO2 GHG GtCO2e 2020
        self._ECO2eGHGB2020     = self._ECO2eGHGB2020act * self._nonco2frac           # Emis of abatable nonCO2 GHG GtCO2e 2020
        self._ECO2eGHGB2100act  = float(self._csv_data.loc['ECO2eGHGB2100act',col])   # Emis of nonCO2 GHG GtCO2e 2100
        self._ECO2eGHGB2100     = self._ECO2eGHGB2100act * self._nonco2frac           # Emis of abatable nonCO2 GHG GtCO2e 2100
        self._Fcoef1            = float(self._csv_data.loc['Fcoef1',col])             # Coefficient of nonco2 abateable emissions
        self._Fcoef2            = float(self._csv_data.loc['Fcoef2',col])             # Coefficient of nonco2 abateable emissions

        #########################################################
        #Population and technology 
        #########################################################
        
        self._gama              = float(self._csv_data.loc['gama',1])               #Capital elasticity in production func.
        self._pop1              = float(self._csv_data.loc['pop1',col])               #Initial world population 2020 (millions)
        self._popadj            = float(self._csv_data.loc['popadj',col])             #Growth rate to calibrate to 2050 pop projection
        self._popasym           = float(self._csv_data.loc['popasym',col])            #Asymptotic population (millions)
        self._dk                = float(self._csv_data.loc['dk',col])                 #Deprication on capital (per year)
        self._q1                = float(self._csv_data.loc['q1',col])                 #Initial world output 2020 (trillion 2019 USD)
        self._AL1               = float(self._csv_data.loc['AL1',col])                #Initial level of total factor productivity
        self._gA1               = float(self._csv_data.loc['gA1',col])                #Initial growth rate for TFP per 5 yrs 
        self._delA              = float(self._csv_data.loc['delA',col])               #Decline rate of TFP per 5 yrs 

        ####################################################################
        #Emissions parameters and Non-CO2 GHG with sigma = emissions/output 
        ####################################################################

        self._gsigma1           = float(self._csv_data.loc['gsigma1',col])            #Initial growth of sigma (per year)  
        self._delgsig           = float(self._csv_data.loc['delgsig',col])            #Decline rate of gsigma per period
        self._asymgsig          = float(self._csv_data.loc['asymgsig',col])           #Asympototic gsigma  
        self._e1                = float(self._csv_data.loc['e1',col])                 #Industrial emissions 2020 (GtCO2 per year)   
        self._miu1              = float(self._csv_data.loc['miu1',col])               #Emissions control rate historical 2020
        self._miu2              = float(self._csv_data.loc['miu2',col])               #Second emission limit
        self._fosslim           = float(self._csv_data.loc['fosslim',col])            #Maximum cumulative extraction fossil fuels (GtC) ##
        self._CumEmiss0         = float(self._csv_data.loc['CumEmiss0',col])          #CumEmiss0 Cumulative emissions 2020 (GtC)
        self._emissrat2020      = float(self._csv_data.loc['emissrat2020',col])       #Ratio of CO2e to industrial CO2 2020
        self._emissrat2100      = float(self._csv_data.loc['emissrat2100',col])       #Ratio of CO2e to industrial CO2 2100

        ####################################################################
        #Climate damage parameter
        ####################################################################
        
        self._a1                = float(self._csv_data.loc['a1',col])                 #Damage intercept
        self._a2base            = float(self._csv_data.loc['a2base',col])             #Damage quadratic term rev 01-13-23
        self._a3                = float(self._csv_data.loc['a3',col])                 #Damage exponent

        ####################################################################
        #Abatement cost
        ####################################################################

        self._expcost2          = float(self._csv_data.loc['expcost2',col])           #Exponent of control cost function
        self._pback2050         = float(self._csv_data.loc['pback2050',col])          #Cost of backstop 2019$ per tCO2 2050
        self._cprice1           = float(self._csv_data.loc['cprice1',col])            #Carbon price 2020 2019$ per tCO2
        self._cpriceup          = float(self._csv_data.loc['cpriceup',col])           #Upper limit carbon price
        self._gcprice           = float(self._csv_data.loc['gcprice',col])            #Growth rate of base carbon price per year
        self._gback             = float(self._csv_data.loc['gback',col])              #Initial cost decline backstop cost per year ##

        ####################################################################
        #Limits on emissions controls, and other boundaries (lo and up are gams terminologies)
        ####################################################################

        self._limmiu2070        = float(self._csv_data.loc['limmiu2070',col])         #Emission control limit from 2070
        self._limmiu2120        = float(self._csv_data.loc['limmiu2120',col])         #Emission control limit from 2120
        self._limmiu2200        = float(self._csv_data.loc['limmiu2200',col])         #Emission control limit from 2220
        self._limmiu2300        = float(self._csv_data.loc['limmiu2300',col])         #Emission control limit from 2300
        self._delmiumax         = float(self._csv_data.loc['delmiumax',col])          #Emission control delta limit per period
        self._klo               = float(self._csv_data.loc['klo',col])                #Lower limit on K for all t
        self._clo               = float(self._csv_data.loc['clo',col])                #Lower limit on C for all t
        self._cpclo             = float(self._csv_data.loc['cpclo',col])              #Lower limit on CPC for all t
        self._rfactlong1        = float(self._csv_data.loc['rfactlong1',col])         #Initial RFACTLONG at tfirst
        self._rfactlonglo       = float(self._csv_data.loc['rfactlonglo',col])        #Lower limit on RFACTLONG for all t
        self._alphalo           = float(self._csv_data.loc['alphalo',col])            #Lower limit on alpha for all t
        self._alphaup           = float(self._csv_data.loc['alphaup',col])            #Upper limit on alpha for all t
        self._matlo             = float(self._csv_data.loc['matlo',col])              #Lower limit on MAT for all t
        self._tatmlo            = float(self._csv_data.loc['tatmlo',col])             #Lower limit on TATM for all t
        self._tatmup            = float(self._csv_data.loc['tatmup',col])             #Upper limit on TATM for all t
        self._sfx2200           = float(self._csv_data.loc['sfx2200',col])            #Fixed Gross savings rate after 2200

        ####################################################################
        #Preferences, growth uncertainty, and timing
        ####################################################################
    
        self._betaclim          = float(self._csv_data.loc['betaclim',col])           #Climate beta
        self._elasmu            = float(self._csv_data.loc['elasmu',col])             #Elasticity of marginal utility of consumption
        self._prstp             = float(self._csv_data.loc['prstp',col])              #Pure rate of social time preference
        self._pi                = float(self._csv_data.loc['pi',col])                 #Capital risk premium
        self._k0                = float(self._csv_data.loc['k0',col])                 #Initial capital stock calibrated (1012 2019 USD)
        self._siggc1            = float(self._csv_data.loc['siggc1',col])             #Annual standard deviation of consumption growth

        #####################################################################
        #Scaling so that MU(C(1)) = 1 and objective function = PV consumption
        #####################################################################

        self._SRF               = float(self._csv_data.loc['SRF',col])                #Scaling factor discounting
        self._scale1            = float(self._csv_data.loc['scale1',col])             #Multiplicative scaling coefficient
        self._scale2            = float(self._csv_data.loc['scale2',col])             #Additive scaling coefficient








class defaultParams():
    '''
    default parameters for DICE2023 with DFAIR
    '''

    def __init__(self):

        self._region    = 'global'
        
        # Num times should be 81, time increment should be 5 years
        self._tstep     = 5             #Years in period
        self._numtimes  = 81
        self._yr0       = 2020          #Calendar year that corresponds to model year zero ##
        self._finalyr   = self._yr0 + self._tstep * (self._numtimes)

        self._emshare0  = 0.2173        #Carbon emissions share into Reservoir 0
        self._emshare1  = 0.224         #Carbon emissions share into Reservoir 1
        self._emshare2  = 0.2824        #Carbon emissions share into Reservoir 2
        self._emshare3  = 0.2763        #Carbon emissions share into Reservoir 3
        self._tau0      = 1000000       #Decay time constant for R0  (year)
        self._tau1      = 394.4         #Decay time constant for R1  (year)
        self._tau2      = 36.53         #Decay time constant for R2  (year)
        self._tau3      = 4.304         #Decay time constant for R3  (year)
    
        self._teq1      = 0.324         #Thermal equilibration parameter for box 1 (m^2 per KW)
        self._teq2      = 0.44          #Thermal equilibration parameter for box 2 (m^2 per KW)
        self._d1        = 236           #Thermal response timescale for deep ocean (year)
        self._d2        = 4.07          #Thermal response timescale for upper ocean (year)

        self._irf0      = 32.4          #Pre-industrial IRF100 (year)
        self._irC       = 0.019         #Increase in IRF100 with cumulative carbon uptake (years per GtC)
        self._irT       = 4.165         #Increase in IRF100 with warming (years per degree K)   
        self._fco22x    = 3.93          #Forcings of equilibrium CO2 doubling (Wm-2)

        # INITIAL CONDITIONS CALIBRATED TO HISTORY
        self._mat0      = 886.5128014   #Initial concentration in atmosphere in 2020 (GtC)
        self._res00     = 150.093       #Initial concentration in Reservoir 0 in 2020
        self._res10     = 102.698       #Initial concentration in Reservoir 1 in 2020
        self._res20     = 39.534        #Initial concentration in Reservoir 2 in 2020
        self._res30     = 6.1865        #Initial concentration in Reservoir 3 in 2020

        self._mateq     = 588           #Equilibrium concentration atmosphere
        self._tbox10    = 0.1477        #Initial temperature box 1 change in 2020
        self._tbox20    = 1.099454      #Initial temperature box 2 change in 2020
        self._tatm0     = 1.24715       #Initial atmospheric temperature change in 2020 

        # Parameters for non-industrial emission (Assumes abateable share of non-CO2 GHG is 65%)
        self._eland0           = 5.9     # Carbon emissions from land 2015 (GtCO2 per year)
        self._deland           = 0.1     # Decline rate of land emissions (per period)
        self._F_Misc2020       = -0.054  # Non-abatable forcings 2020
        self._F_Misc2100       = 0.265   # Non-abatable forcings 2100
        self._F_GHGabate2020   = 0.518   # Forcings of abatable nonCO2 GHG
        self._F_GHGabate2100   = 0.957   # Forcings of abatable nonCO2 GHG
        self._ECO2eGHGB2020    = 9.96    # Emis of abatable nonCO2 GHG GtCO2e 2020
        self._ECO2eGHGB2100    = 15.5    # Emis of abatable nonCO2 GHG GtCO2e 2100
        self._nonco2frac       = 0.65                                       # Abateable share of nonco2 GHGs
        self._ECO2eGHGB2020act = 15.3230769230769                           # Emis of nonCO2 GHG GtCO2e 2020
        self._ECO2eGHGB2020    = self._ECO2eGHGB2020act * self._nonco2frac  # Emis of abatable nonCO2 GHG GtCO2e 2020
        self._ECO2eGHGB2100act = 23.8461538461538                           # Emis of nonCO2 GHG GtCO2e 2100
        self._ECO2eGHGB2100    = self._ECO2eGHGB2100act * self._nonco2frac  # Emis of abatable nonCO2 GHG GtCO2e 2100
        self._Fcoef1           = 0.00955 # Coefficient of nonco2 abateable emissions
        self._Fcoef2           = 0.861   # Coefficient of nonco2 abateable emissions

        #########################################################
        #Population and technology 
        #########################################################
        
        self._gama    = 0.300  #Capital elasticity in production func.
        self._pop1    = 7752.9 #Initial world population 2020 (millions)
        self._popadj  = 0.145  #Growth rate to calibrate to 2050 pop projection
        self._popasym = 10825  #Asymptotic population (millions)
        self._dk      = 0.100  #Deprication on capital (per year)
        self._q1      = 135.7  #Initial world output 2020 (trillion 2019 USD)
        self._AL1     = 5.84   #Initial level of total factor productivity
        self._gA1     = 0.066  #Initial growth rate for TFP per 5 yrs 
        self._delA    = 0.0015 #Decline rate of TFP per 5 yrs 

        ####################################################################
        #Emissions parameters and Non-CO2 GHG with sigma = emissions/output 
        ####################################################################

        self._gsigma1      = -0.015 #Initial growth of sigma (per year)  
        self._delgsig      = 0.96   #Decline rate of gsigma per period
        self._asymgsig     = -0.005 #Asympototic gsigma  
        self._e1           = 37.56  #Industrial emissions 2020 (GtCO2 per year)   
        self._miu1         = 0.05   #Emissions control rate historical 2020
        self._miu2         = 0.10   #Second emission limit
        self._fosslim      = 6000   #Maximum cumulative extraction fossil fuels (GtC) ##
        self._CumEmiss0    = 633.5  #CumEmiss0 Cumulative emissions 2020 (GtC)
        self._emissrat2020 = 1.40   #Ratio of CO2e to industrial CO2 2020
        self._emissrat2100 = 1.21   #Ratio of CO2e to industrial CO2 2100

        ####################################################################
        #Climate damage parameter
        ####################################################################
        
        self._a1           = 0        #Damage intercept
        self._a2base       = 0.003467 #Damage quadratic term rev 01-13-23
        self._a3           = 2.00     #Damage exponent

        ####################################################################
        #Abatement cost
        ####################################################################

        self._expcost2     = 2.6    #Exponent of control cost function
        self._pback2050    = 515.0  #Cost of backstop 2019$ per tCO2 2050
        self._cprice1      = 6      #Carbon price 2020 2019$ per tCO2
        self._cpriceup     = 1000   #Upper limit carbon price
        self._gcprice      = 0.025  #Growth rate of base carbon price per year
        self._gback        = -0.012 #Initial cost decline backstop cost per year ##

        ####################################################################
        #Limits on emissions controls, and other boundaries
        ####################################################################

        self._limmiu2070  = 1.0     #Emission control limit from 2070
        self._limmiu2120  = 1.1     #Emission control limit from 2120
        self._limmiu2200  = 1.05    #Emission control limit from 2220
        self._limmiu2300  = 1.0     #Emission control limit from 2300
        self._delmiumax   = 0.12    #Emission control delta limit per period
        self._klo         = 1       #Lower limit on K for all t
        self._clo         = 2       #Lower limit on C for all t
        self._cpclo       = 0.01    #Lower limit on CPC for all t
        self._rfactlong1  = 1000000 #Initial RFACTLONG at tfirst
        self._rfactlonglo = 0.0001  #Lower limit on RFACTLONG for all t
        self._alphalo     = 0.1     #Lower limit on alpha for all t
        self._alphaup     = 100     #Upper limit on alpha for all t
        self._matlo       = 10      #Lower limit on MAT for all t
        self._tatmlo      = 0.5     #Lower limit on TATM for all t
        self._tatmup      = 20      #Upper limit on TATM for all t
        self._sfx2200     = 0.28    #Fixed Gross savings rate after 2200

        ####################################################################
        #Preferences, growth uncertainty, and timing
        ####################################################################
    
        self._betaclim   = 0.5   #Climate beta
        self._elasmu     = 0.95  #Elasticity of marginal utility of consumption
        self._prstp      = 0.001 #Pure rate of social time preference
        self._pi         = 0.05  #Capital risk premium
        self._k0         = 295   #Initial capital stock calibrated (1012 2019 USD)
        self._siggc1     = 0.01  #Annual standard deviation of consumption growth

        #####################################################################
        #Scaling so that MU(C(1)) = 1 and objective function = PV consumption
        #####################################################################

        self._SRF        = 1000000    #Scaling factor discounting
        self._scale1     = 0.00891061 #Multiplicative scaling coefficient
        self._scale2     = -6275.91   #Additive scaling coefficient
        






def simParams(paramObj):
    
    '''
    simulate time-varying parameters.
    (move any to main model that are endogenized in future)
    keyword p: pass getParamsfromfile or defaultParams object
    '''
    p = paramObj
    
    p._L = [np.nan for i in range(p._numtimes+2)]                #
    p._aL = [np.nan for i in range(p._numtimes+2)]               #
    p._sigma = [np.nan for i in range(p._numtimes+2)]            #
    p._sigmatot = [np.nan for i in range(p._numtimes+2)]         #
    p._gA = [np.nan for i in range(p._numtimes+2)]               #
    p._gL = [np.nan for i in range(p._numtimes+2)]               #
    p._gsig = [np.nan for i in range(p._numtimes+2)]             #
    p._eland = [np.nan for i in range(p._numtimes+2)]            #
    p._cost1tot = [np.nan for i in range(p._numtimes+2)]         #
    p._pbacktime = [np.nan for i in range(p._numtimes+2)]        #
    p._cpricebase = [np.nan for i in range(p._numtimes+2)]       #
    p._gbacktime = [np.nan for i in range(p._numtimes+2)]        #
    p._varpcc = [np.nan for i in range(p._numtimes+2)]           #
    p._rprecaut = [np.nan for i in range(p._numtimes+2)]         #
    p._rr = [np.nan for i in range(p._numtimes+2)]               #
    p._rr1 = [np.nan for i in range(p._numtimes+2)]              #
    p._CO2E_GHGabateB = [np.nan for i in range(p._numtimes+2)]   #
    p._CO2E_GHGabateact = [np.nan for i in range(p._numtimes+2)] #
    p._F_Misc = [np.nan for i in range(p._numtimes+2)]           #
    p._emissrat = [np.nan for i in range(p._numtimes+2)]         #
    p._FORC_CO2 = [np.nan for i in range(p._numtimes+2)]         #

    for t in range(1,p._numtimes+2):
        
         #varpccEQ(m,t): # Variance of per capita consumption
        p._varpcc[t] = min(p._siggc1**2*5*(t-1), p._siggc1**2*5*47)
        
         #rprecautEQ(m,t): # Precautionary rate of return
        p._rprecaut[t] = (-0.5 * p._varpcc[t] * p._elasmu**2)
        
         #rr1EQ(m,t): # STP factor without precautionary factor
        rartp = np.exp(p._prstp + p._betaclim * p._pi)-1
        p._rr1[t] = 1/((1 + rartp)**(p._tstep * (t-1)))
        
         #rrEQ(m,t): # STP factor with precautionary factor
        p._rr[t] = p._rr1[t] * (1 + p._rprecaut[t])**(-p._tstep * (t-1))
        
         #LEQ(m,t): # Population adjustment over time (note initial condition)
        p._L[t] = p._L[t-1]*(p._popasym / p._L[t-1])**p._popadj if t > 1 else p._pop1
        
         #gAEQ(m,t): # Growth rate of productivity (note initial condition - can either enforce through bounds or this constraint)
        p._gA[t] = p._gA1 * np.exp(-p._delA * p._tstep * (t-1)) if t > 1 else p._gA1
        
         #aLEQ(m,t): # Level of total factor productivity (note initial condition)
        p._aL[t] = p._aL[t-1] /((1-p._gA[t-1])) if t > 1 else p._AL1

         #cpricebaseEQ(m,t): # Carbon price in base case of model
        p._cpricebase[t] = p._cprice1 * ((1 + p._gcprice)**(p._tstep * (t-1)))
        
         #pbacktimeEQ(m,t): # Backstop price 2019$ per ton CO2
        j = 0.01 if t <= 7 else 0.001
        p._pbacktime[t] = p._pback2050 * np.exp(-p._tstep * j * (t-7))
        
         #gsigEQ(m,t): # Change in rate of sigma (rate of decarbonization)
        p._gsig[t] = min(p._gsigma1 * (p._delgsig**(t-1)), p._asymgsig)
        
         #sigmaEQ(m,t): # CO2 emissions output ratio (note initial condition)
        p._sigma[t] = p._sigma[t-1]*np.exp(p._tstep*p._gsig[t-1]) if t > 1 else p._e1/(p._q1*(1-p._miu1))
        
         # elandEQ(m,t):
        p._eland[t] = p._eland0 * (1 - p._deland)**(t-1)
         
         # CO2E_GHGabateBEQ(m,t):
        p._CO2E_GHGabateB[t] = p._ECO2eGHGB2020 + ((p._ECO2eGHGB2100 - p._ECO2eGHGB2020) / 16) * (t-1) if t <= 16 else p._ECO2eGHGB2100
         
         # F_MiscEQ(m,t):
        p._F_Misc[t] = p._F_Misc2020 + ((p._F_Misc2100 - p._F_Misc2020) / 16) * (t-1) if t <= 16 else p._F_Misc2100
         
         # emissratEQ(m,t):
        p._emissrat[t] = p._emissrat2020 + ((p._emissrat2100 - p._emissrat2020) / 16) * (t-1) if t <= 16 else p._emissrat2100
         
         # sigmatotEQ(m,t):
        p._sigmatot[t] = p._sigma[t] * p._emissrat[t]
         
         # cost1totEQ(m,t):
        p._cost1tot[t] = p._pbacktime[t] * p._sigmatot[t] / p._expcost2 / 1000

    return None




