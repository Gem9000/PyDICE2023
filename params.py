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
        self.ifopt = 0

        #########################################################
        #Population and technology 
        #########################################################
        
        self.gama = 0.300 #Capital elasticity in production func.
        self.pop1 = 7752.9 #Initial world population 2020 (millions)
        self.popadj = 0.145 # Growth rate to calibrate to 2050 pop projection
        self.popasym = 10825 #Asymptotic population (millions)
        self.dk = 0.100 #Deprication on capital (per year)
        self.q1 = 135.7 #Initial world output 2020 (trillion 2019 USD)
        self.AL1 = 5.84 #Initial level of total factor productivity
        self.gA1 = 0.066 #Initial growth rate for TFP per 5 yrs 
        self.delA = 0.0015 #Decline rate of TFP per 5 yrs 

        ####################################################################
        #Emissions parameters and Non-CO2 GHG with sigma = emissions/output 
        ####################################################################

        self.gsigma1 = -0.015 #Initial growth of sigma (per year)  
        self.delgsig = 0.96   #Decline rate of gsigma per period
        self.asymgsig = -0.005 #Asympototic gsigma  
        self.e1 = 37.56  #Industrial emissions 2020 (GtCO2 per year)   
        self.miu1 = 0.05  #Emissions control rate historical 2020 
        self.fosslim = 6000 #Maximum cumulative extraction fossil fuels (GtC)
        self.CumEmiss0 = 633.5 #CumEmiss0 Cumulative emissions 2020 (GtC) 

        ####################################################################
        #Climate damage parameter
        ####################################################################
        
        self.a1 = 0 #Damage intercept
        self.a2base = 0.003467 #Damage quadratic term rev 01-13-23
        self.a3 = 2.00 #Damage exponent

        ####################################################################
        #Abatement cost
        ####################################################################

        self.expcost2 = 2.6 #Exponent of control cost function                     
        self.pback2050 = 515.0 #Cost of backstop 2019$ per tCO2 2050                   
        self.gback  =  -0.012 #Initial cost decline backstop cost per year           
        self.cprice1 = 6 #Carbon price 2020 2019$ per tCO2                         
        self.gcprice  = 0.025 #Growth rate of base carbon price per year              

        ####################################################################
        #Limits on emissions controls
        ####################################################################

        self.limmiu2070 = 1.0   #Emission control limit from 2070          
        self.limmiu2120 = 1.1   #Emission control limit from 2120          
        self.limmiu2200 = 1.05  #Emission control limit from 2220          
        self.limmiu2300 = 1.0   #Emission control limit from 2300                
        self.delmiumax  =  0.12 #Emission control delta limit per period   

        ####################################################################
        #Preferences, growth uncertainty, and timing
        ####################################################################
    
        self.betaclim = 0.5 #Climate beta                                      
        self.elasmu   = 0.95 #Elasticity of marginal utility of consumption    
        self.prstp    = 0.001 #Pure rate of social time preference               
        self.pi       = 0.05  #Capital risk premium                              
        self.rartp    = 0    #Risk-adjusted rate of time preference (SET TO ZERO FOR NOW)
        self.k0       =  295  #Initial capital stock calibrated (1012 2019 USD)  
        self.siggc1   = 0.01  #Annual standard deviation of consumption growth   
        self.sig1     = 0 #Carbon intensity 2020 kgCO2-output 2020 (SET TO ZERO FOR NOW)

        ####################################################################
        #Scaling so that MU(C(1)) = 1 and objective function = PV consumption
        ####################################################################
        self.tstep = tstep #Years in period 
        self.SRF = 1000000 #Scaling factor discounting
        self.scale1 = 0.00891061 #Multiplicative scaling coefficient
        self.scale2 = -6275.91   #Additive scaling coefficient

        
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
        self.l = np.zeros(num_times+1)
        self.al = np.zeros(num_times+1)
        self.sigma = np.zeros(num_times+1)
        self.sigmatot = np.zeros(num_times+1)
        self.gA = np.zeros(num_times+1)
        self.gL = np.zeros(num_times+1)
        self.gcost1 = np.zeros(num_times+1)
        self.gsig = np.zeros(num_times+1)
        self.eland = np.zeros(num_times+1)
        self.cost1tot = np.zeros(num_times+1)


print("Success")
