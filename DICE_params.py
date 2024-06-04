'''
Authors: Jacob Wessel, George Moraites
'''

import numpy as np

class modelParams():

    def __init__(self, num_times=82, tstep=5):

    #########################################################
    #Initial Parameters for DICE with DFAIR
    #########################################################

        # Num times should be 82, time increment should be 5 years
        self._tstep     = tstep         #Years in period
        self._num_times = num_times
        self._t         = np.arange(0,self._num_times+1)

        self._yr0       = 2020          #Calendar year that corresponds to model year zero ##
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
        self._MILLE      = 1000.0
        
    def runModel(self):
        pass    

print("Success")

fair_params = modelParams()

fair_params.runModel()

