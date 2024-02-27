'''
Created 02/23/2024 
Author: George Moraites
Adapted From: domokane dice_params.py
'''

import numpy as np
import csv

class DiceParams():

    '''
    This Class will hold the static runtime 
    paramerter imputs to the DICE model.
    '''

    def __init__(self, num_times, tstep):
        
        #Indicator variable. 1 Denotes optimized and zero is base
        self._ifopt = 0

        #########################################################
        #Population and technology 
        #########################################################
        
        self._gama = 0.300 #Capital elasticity in production func.
        self._pop1 = 7752.9 #Initial world population 2020 (millions)
        self._popadj = 0.145 # Growth rate to calibrate to 2050 pop projection
        self._popasym = 10825 #Asymptotic population (millions)
        self._dk = 0.100 #Deprication on capital (per year)
        self._q1 = 135.7 #Initial world output 2020 (trillion 2019 USD)
        self._AL1 = 5.84 #Initial level of total factor productivity
        self._gA1 = 0.066 #Initial growth rate for TFP per 5 yrs 
        self._delA = 0.0015 #Decline rate of TFP per 5 yrs 

        ####################################################################
        #Emissions parameters and Non-CO2 GHG with sigma = emissions/output 
        ####################################################################

        self._gsigma1 = -0.015 #Initial growth of sigma (per year)  
        self._delgsig = 0.96   #Decline rate of gsigma per period
        self._asymgsig = -0.005 #Asympototic gsigma  
        self._e1 = 37.56  #Industrial emissions 2020 (GtCO2 per year)   
        self._miu1 = 0.05  #Emissions control rate historical 2020 
        self._fosslim = 6000 #Maximum cumulative extraction fossil fuels (GtC)
        self._CumEmiss0 = 633.5 #CumEmiss0 Cumulative emissions 2020 (GtC) 

        ####################################################################
        #Climate damage parameter
        ####################################################################
        
        self._a1 = 0 #Damage intercept
        self._a2base = 0.003467 #Damage quadratic term rev 01-13-23
        self._init__a3 = 2.00 #Damage exponent

        ####################################################################
        #Abatement cost
        ####################################################################

        self._expcost2 = 2.6 #Exponent of control cost function                     
        self._pback2050 = 515.0 #Cost of backstop 2019$ per tCO2 2050                   
        self._gback  =  -0.012 #Initial cost decline backstop cost per year           
        self._cprice1 = 6 #Carbon price 2020 2019$ per tCO2                         
        self._gcprice  = 0.025 #Growth rate of base carbon price per year              

        ####################################################################
        #Limits on emissions controls
        ####################################################################

        self._limmiu2070 = 1.0   #Emission control limit from 2070          
        self._limmiu2120 = 1.1   #Emission control limit from 2120          
        self._limmiu2200 = 1.05  #Emission control limit from 2220          
        self._limmiu2300 = 1.0   #Emission control limit from 2300                
        self._delmiumax  =  0.12 #Emission control delta limit per period   

        ####################################################################
        #Preferences, growth uncertainty, and timing
        ####################################################################
    
        self._betaclim = 0.5 #Climate beta                                      
        self._elasmu   = 0.95 #Elasticity of marginal utility of consumption    
        self._prstp    = 0.001 #Pure rate of social time preference               
        self._pi       = 0.05  #Capital risk premium                              
        self._rartp    = 0    #Risk-adjusted rate of time preference (SET TO ZERO FOR NOW)
        self._k0       =  295  #Initial capital stock calibrated (1012 2019 USD)  
        self._siggc1   = 0.01  #Annual standard deviation of consumption growth   
        self._sig1     = 0 #Carbon intensity 2020 kgCO2-output 2020 (SET TO ZERO FOR NOW)

        ####################################################################
        #Scaling so that MU(C(1)) = 1 and objective function = PV consumption
        ####################################################################
        self._tstep = tstep #Years in period 
        self._SRF = 1000000 #Scaling factor discounting
        self._scale1 = 0.00891061 #Multiplicative scaling coefficient
        self._scale2 = -6275.91   #Additive scaling coefficient

        
        ##Legacy equations, may not be relevant for 2023 version
        #self.a20 = self.a2base
        #self.lam = self.q
        #self.

        ##################################################################
        #Functions
        ##################################################################

        #Num times should be 81
        #time increment should be 5 years
        self._num_times = num_times
        self._time_increment = np.arrange(0,self.num_times+1)

        #Create size arrays so we can index from 1 instead of 0
        self._l = np.zeros(num_times+1)
        self._al = np.zeros(num_times+1)
        self._sigma = np.zeros(num_times+1)
        self._sigmatot = np.zeros(num_times+1)
        self._gA = np.zeros(num_times+1)
        self._gL = np.zeros(num_times+1)
        self._gcost1 = np.zeros(num_times+1)
        self._gsig = np.zeros(num_times+1)
        self._eland = np.zeros(num_times+1)
        self._cost1tot = np.zeros(num_times+1)
        self._pbacktime = np.zeros(num_times+1)
        self._scc = np.zeros(num_times+1)
        self._cpricebase = np.zeros(num_times+1)
        self._ppm = np.zeros(num_times+1)
        self._atfrac2020 = np.zeros(num_times+1)
        self._atfrac1765 = np.zeros(num_times+1)
        self._abaterat = np.zeros(num_times+1)
        self._miuup = np.zeros(num_times+1)
        self._gbacktime = np.zeros(num_times+1)


        #Set relevant values using the paramteters above
        self._l[1] = self._pop1 #Population 
        self._gA[1] = self._gA1 #Growth rate
        self._al[1] = self._AL1 #Initial total factor productivity
        self._gsig[1] = self._gsigma1 #Initial growth of sigma
        self._sigma[1] = self._sig1 #sig1 did not have a value in the gms code
        
        


print("Success")
