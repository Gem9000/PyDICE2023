'''
Authors: George Moraites, Jacob Wessel
'''

import numpy as np
import math
import csv
from scipy.optimize import minimize_scalar

class modelParams():

    def __init__(self, num_times=82, tstep=5):

    #########################################################
    #Initial Parameters for the FAIR model
    #########################################################

        self._tstep    = tstep         #Years in period 

        self._yr0      = 2020          #Calendar year that corresponds to model year zero
        self._emshare0 = 0.2173        #Carbon emissions share into Reservoir 0
        self._emshare1 = 0.224         #Carbon emissions share into Reservoir 1
        self._emshare2 = 0.2824        #Carbon emissions share into Reservoir 2
        self._emshare3 = 0.2763        #Carbon emissions share into Reservoir 3
        self._tau0     = 1000000       #Decay time constant for R0  (year)
        self._tau1     = 394.4         #Decay time constant for R1  (year)
        self._tau2     = 36.53         #Decay time constant for R2  (year)
        self._tau3     = 4.304         #Decay time constant for R3  (year)
    
        self._teq1     = 0.324         #Thermal equilibration parameter for box 1 (m^2 per KW)
        self._teq2     = 0.44          #Thermal equilibration parameter for box 2 (m^2 per KW)
        self._d1       = 236           #Thermal response timescale for deep ocean (year)
        self._d2       = 4.07          #Thermal response timescale for upper ocean (year)

        self._irf0     = 32.4          #Pre-industrial IRF100 (year)
        self._irC      = 0.019         #Increase in IRF100 with cumulative carbon uptake (years per GtC)
        self._irT      = 4.165         #Increase in IRF100 with warming (years per degree K)   
        self._fco22x   = 3.93          #Forcings of equilibrium CO2 doubling (Wm-2)

        # INITIAL CONDITIONS CALIBRATED TO HISTORY
        self._mat0     = 886.5128014   #Initial concentration in atmosphere in 2020 (GtC)
        self._res00    = 150.093       #Initial concentration in Reservoir 0 in 2020
        self._res10    = 102.698       #Initial concentration in Reservoir 1 in 2020
        self._res20    = 39.534        #Initial concentration in Reservoir 2 in 2020
        self._res30    = 6.1865        #Initial concentration in Reservoir 3 in 2020

        self._mateq    = 588           #Equilibrium concentration atmosphere
        self._tbox10   = 0.1477        #Initial temperature box 1 change in 2020
        self._tbox20   = 1.099454      #Initial temperature box 2 change in 2020
        self._tatm0    = 1.24715       #Initial atmospheric temperature change in 2020 

        # Num times should be 82, time increment should be 5 years
        self._num_times = num_times
        self._t         = np.arange(0,self._num_times+1)

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
        self._fosslim      = 6000   #Maximum cumulative extraction fossil fuels (GtC)
        self._CumEmiss0    = 633.5  #CumEmiss0 Cumulative emissions 2020 (GtC)
        self._emissrat2020 = 1.40   #Ratio of CO2e to industrial CO2 2020
        self._emissrat2100 = 1.21   #Ratio of CO2e to industrial CO2 2100

        ####################################################################
        #Climate damage parameter
        ####################################################################
        
        self._a1        = 0        #Damage intercept
        self._a2base    = 0.003467 #Damage quadratic term rev 01-13-23
        self._a3  = 2.00     #Damage exponent

        ####################################################################
        #Abatement cost
        ####################################################################

        self._expcost2  = 2.6    #Exponent of control cost function                     
        self._pback2050 = 515.0  #Cost of backstop 2019$ per tCO2 2050                   
        self._gback     = -0.012 #Initial cost decline backstop cost per year           
        self._cprice1   = 6      #Carbon price 2020 2019$ per tCO2                         
        self._gcprice   = 0.025  #Growth rate of base carbon price per year              

        ####################################################################
        #Limits on emissions controls
        ####################################################################

        self._limmiu2070 = 1.0   #Emission control limit from 2070          
        self._limmiu2120 = 1.1   #Emission control limit from 2120          
        self._limmiu2200 = 1.05  #Emission control limit from 2220          
        self._limmiu2300 = 1.0   #Emission control limit from 2300                
        self._delmiumax  = 0.12  #Emission control delta limit per period   

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







        #Create size arrays so we can index from 1 instead of 0
        self._L = np.zeros(num_times+1)
        self._al = np.zeros(num_times+1)
        self._sigma = np.zeros(num_times+1)
        self._sigmatot = np.zeros(num_times+1)
        self._gA = np.zeros(num_times+1)
        self._gL = np.zeros(num_times+1)
        self._gcost1 = np.zeros(num_times+1)
        self._gsig = np.zeros(num_times+1)
        self._eland = np.zeros(num_times+1)
        self._emissrat = np.zeros(num_times+1)
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
        self._rr = np.zeros(num_times+1) 
        self._varpcc = np.zeros(num_times+1)
        self._rprecaut = np.zeros(num_times+1)
        self._RR1 = np.zeros(num_times+1)

        self._res0lom         = np.zeros(num_times+1)
        self._res1lom         = np.zeros(num_times+1)
        self._res2lom         = np.zeros(num_times+1)
        self._res3lom         = np.zeros(num_times+1)
        self._mmat            = np.zeros(num_times+1)
        self._cacc            = np.zeros(num_times+1)
        self._force           = np.zeros(num_times+1)
        self._tbox1           = np.zeros(num_times+1)
        self._tbox2           = np.zeros(num_times+1)
        self._tatm            = np.zeros(num_times+1)
        self._irfeqlhs        = np.zeros(num_times+1)
        self._irfeqrhs        = np.zeros(num_times+1)
        self._alpha         = np.zeros(num_times+1)
        self._calculated_mmat = np.zeros(num_times+1)

        # variables from nonco2 forcings include
        self._eco2            = np.zeros(num_times+1) #Total CO2 emissions (GtCO2 per year) XXX
        self._eco2e           = np.zeros(num_times+1) #Total CO2e emissions including abateable nonCO2 GHG (GtCO2 per year)
        self._F_GHGabate      = np.zeros(num_times+1) #Forcings abateable nonCO2 GHG
        self._eind            = np.zeros(num_times+1) #Industrial CO2 emissions (GtCO2 per yr)
        self._eland           = np.zeros(num_times+1) #Emissions from deforestation (GtCO2 per year)

        self._CO2E_GHGabateB   = np.zeros(num_times+1) #Abateable non-CO2 GHG emissions base
        self._CO2E_GHGabateact = np.zeros(num_times+1) #Abateable non-CO2 GHG emissions base (actual)
        self._F_Misc           = np.zeros(num_times+1) #Non-abateable forcings (GHG and other)
        self._sigmatot         = np.zeros(num_times+1) #Emissions output ratio for CO2e
        
        self._K = np.zeros(num_times+1)          #Capital stock (trillions 2019 US dollars)
        self._C = np.zeros(num_times+1)              #Consumption (trillions 2019 US dollars per year)
        self._CPC = np.zeros(num_times+1)            #Per capita consumption (thousands 2019 USD per year)
        self._I =  np.zeros(num_times+1)             #Investment (trillions 2019 USD per year)
        self._Y  = np.zeros(num_times+1)             #Gross world product net of abatement and damages (trillions 2019 USD per year)
        self._ygross = np.zeros(num_times+1)     #Gross world product of abatement and damages (trillions 2019 USD per year)
        self._ynet  = np.zeros(num_times+1)          #Output net of damages equation (trillions 2019 USD per year)
        self._damages = np.zeros(num_times+1)        #Damages (trillions 2019 USD per year)
        self._damfrac = np.zeros(num_times+1)        #Damages as fraction of gross output
        self._abatecost = np.zeros(num_times+1)      #Cost of emissions reductions  (trillions 2019 USD per year)
        self._mcabate  = np.zeros(num_times+1)       #Marginal cost of abatement (2019$ per ton CO2)
        self._ccatot  = np.zeros(num_times+1)        #Total carbon emissions (GtC)
        self._cprice   =  np.zeros(num_times+1)      #Carbon price (2019$ per ton of CO2)
        self._periodu  = np.zeros(num_times+1)       #One period utility function
        self._totperiodu = np.zeros(num_times+1)     #Period utility
        self._utility    = np.zeros(num_times+1)     #Welfare function
        self._rfactlong = np.zeros(num_times+1)      #Long interest factor
        self._rshort    = np.zeros(num_times+1)      #Short-run interest rate: Real interest rate with precautionary(per annum year on year)
        self._rlong   = np.zeros(num_times+1)        #Long-run interest rate: Real interest rate from year 0 to T

        self._rartp     = math.exp(self._prstp + self._betaclim * self._pi)-1 #Risk adjusted rate of time preference
        self._sig1      = (self._e1)/(self._q1*(1-self._miu1)) #Carbon intensity 2020 kgCO2-output 2020

##### Initial Conditions #####

        self._K[1]      = self._k0
        self._L[1]      = self._pop1    #Population 
        self._gA[1]     = self._gA1     #Growth rate
        self._al[1]     = self._AL1     #Initial total factor productivity
        self._gsig[1]   = self._gsigma1 #Initial growth of sigma
        self._rr[1]     = 1.0
        self._miuup[1]  = self._miu1
        self._miuup[2]  = self._miu2
        self._sigma[1]  = self._sig1
        self._ccatot[1] = self._CumEmiss0

        self._mmat[1]       = self._mat0
        self._tatm[1]     = self._tatm0
        self._res0lom[1]    = self._res00
        self._res1lom[1]    = self._res10
        self._res2lom[1]    = self._res20
        self._res3lom[1]    = self._res30
        self._tbox1[1]    = self._tbox10
        self._tbox2[1]    = self._tbox20
        self._eland[1]      = self._eland0
        self._F_GHGabate[1] = self._F_GHGabate2020
        self._rfactlong[1] = 1000000




    def simulateDynamics(self,x):
    
        # Set the optimization variables
        MIUopt = np.zeros(self._num_times+1)
        Sopt = np.zeros(self._num_times+1)
        for i in range(1, self._num_times+1):
            MIUopt[i] = x[i-1]                   #Optimal emissions control rate GHGs
            Sopt[i] = x[self._num_times + i-1]   #Gross savings rate as fraction of gross world product
    
###########################################################################
#Variables and nonnegative variables equations
###########################################################################

        #Control logic for the emissions control rate (piecewise pyomo)
        for i in range(3, self._num_times+1):
            if self._t[i]> 2:
                self._miuup[i] = (self._delmiumax*(self._t[i]-1))
            if self._t[i] > 8:
                self._miuup[i] = (0.85 + 0.05 * (self._t[i]-8))
            if self._t[i] > 11:
                self._miuup[i] = self._limmiu2070
            if self._t[i] > 20:
                self._miuup[i] = self._limmiu2120
            if self._t[i] > 37:
                self._miuup[i] = self._limmiu2200
            if self._t[i] > 57:
                self._miuup[i] = self._limmiu2300

        for i in range(2, self._num_times+1):

            #Depends on the t-1 time period
            self._ygross[i] = self._al[i] * ((self._L[i]/self._MILLE)**(1.0-self._gama)) * self._K[i]**self._gama  #Gross world product GROSS of abatement and damages (trillions 20i9 USD per year)
            self._eland[i] = self._eland0 * (1 - self._deland) ** (self._t[i]-1)
            self._eco2[i] = (self._sigma[i] * self._ygross[i] + self._eland[i]) * (1-MIUopt) #New
            self._eind[i] = self._sigma[i] * self._ygross[i] * (1.0 - MIUopt[i])
            self._eco2e[i] = (self._sigma[i] * self._ygross[i] + self._eland[i] + self._CO2E_GHGabateB[i]) * (1-MIUopt) #New
            self._ccatot[i] = self._ccatot[i] + self._eco2[i]*(5/3.666)
            self._damfrac[i] = (self._a1 * self._tatm[i] + self._a2base * self._tatm[i] ** self._a3)  
            self._damages[i] = self._ygross[i] * self._damfrac[i]
            self._abatecost[i] = self._ygross[i] * self._cost1tot[i] * MIUopt[i]**self._expcost2
            self._mcabate[i] = self._pbacktime[i] * MIUopt[i]**(self._expcost2-1)
            self._cprice[i] = self._pbacktime[i] * MIUopt[i]**(self._expcost2-1)

            ########################Economic##############################
            self._ynet[i] = self._ygross[i] * (1-self._damfrac[i])
            self._Y[i] = self._ynet[i] - self._abatecost[i]
            self._C[i] = self._Y[i] - self._I[i]
            self._CPC[i] = self._MILLE * self._C[i]/self._L[i]
            self._I[i] = Sopt[i] * self._Y[i]

            # this equation is a <= inequality and needs to be treated as such #
            self._K[i] = (1.0 - self._dk)**self._tstep * self._K[i-1] + self._tstep * self._I[i]

            self._rfactlong[i] = (self._SRF * (self._CPC[i]/self._CPC[i-1])**(-self._elasmu)*self._rr[i]) #Modified/New
            self._rlong[i] = -math.log(self._rfactlong[i]/self._SRF)/(5*(i-1)) #NEW
            self._rshort[i] = (-math.log(self._rfactlong[i]/self._rfactlong[i-1])/5) #NEW 

            #########################Welfare Functions###################
            self._periodu[i] = ((self._C[i]*self._MILLE/self._L[i])**(1.0-self._elasmu)-1.0) / (1.0 - self._elasmu) - 1.0
            self._totperiodu[i] = self._periodu[i] * self._L[i] * self._rr[i]

            self._varpcc[i]       = min(self._siggc1**2*5*(self._t[i]-1), self._siggc1**2*5*47) #Variance of per capita consumption
            self._rprecaut[i]     = -0.5 * self._varpcc[i-1]* self._elasmu**2 #Precautionary rate of return
            self._RR1[i]          = 1/((1+self._rartp)**(-self._tstep*(self._t[i]-1))) #STP factor without precautionary factor
            self._rr[i]           = self._RR1[i-1]*(1+ self._rprecaut[i-1]**(self._tstep*(self._t[i]-1)))  #STP factor with precautionary factor
            self._L[i]            = self._L[i-1]*(self._popasym / self._L[i-1])**self._popadj # Population adjustment over time
            self._gA[i]           = self._gA1 * np.exp(-self._delA * 5.0 * (self._t[i] - 1)) # Growth rate of productivity
            self._al[i]           = self._al[i-1] /((1-self._gA[i-1])) # Level of total factor productivity
            self._cpricebase[i]   = self._cprice1*(1+self._gcprice)**(5*(self._t[i]-1)) #Carbon price in base case of model
            self._pbacktime[i]    = self._pback2050 * math.exp(-5*(0.01 if self._t[i] <= 7 else 0.001)*(self._t[i]-7)) #Backstop price 2019$ per ton CO2. Incorporates 2023 condition
            self._gsig[i]         = min(self._gsigma1*self._delgsig **((self._t[i]-1)), self._asymgsig) #Change in rate of sigma (rate of decarbonization)
            self._sigma[i]        = self._sigma[i-1]*math.exp(5*self._gsig[i-1])
            if self._t[i] <= 16:
                self._emissrat[i] = self._emissrat2020 +((self._emissrat2100-self._emissrat2020)/16)*(self._t[i]-1)
            else:
                self._emissrat[i] = self._emissrat2100
            self._sigmatot[i]     = self._sigma[i]*self._emissrat[i]
            self._cost1tot[i]     = self._pbacktime[i]*self._sigmatot[i]/self._expcost2/1000

            #Solve for alpha(t) in each time period 
            self._alpha[i] = self.solve_alpha(i-1)

            self._res0lom[i] = (self._emshare0 * self._tau0 * self._alpha[i] * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau0 * self._alpha[i]))) + 
                                    self._res0lom[i-1] * math.exp(-self._tstep / (self._tau0 * self._alpha[i])))

            self._res1lom[i] = (self._emshare1 * self._tau1 * self._alpha[i] * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau1 * self._alpha[i]))) + 
                                    self._res1lom[i-1] * math.exp(-self._tstep / (self._tau1 * self._alpha[i])))

            self._res2lom[i] = (self._emshare2 * self._tau2 * self._alpha[i] * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau2 * self._alpha[i]))) + 
                                    self._res2lom[i-1] * math.exp(-self._tstep / (self._tau2 * self._alpha[i])))

            self._res3lom[i] = (self._emshare3 * self._tau3 * self._alpha[i] * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau3 * self._alpha[i]))) + 
                                    self._res3lom[i-1] * math.exp(-self._tstep / (self._tau3 * self._alpha[i])))

            self._calculated_mmat[i] = self._mateq + self._res0lom[i] + self._res1lom[i] + self._res2lom[i] + self._res3lom[i]
            if self._calculated_mmat[i] < 10:
                self._mmat[i] = 10
            else:
                self._mmat[i] = self._calculated_mmat[i]
                
            self._force[i] = (self._fco22x * ((math.log((self._mmat[i-1]+1e-9/self._mateq))/math.log(2)) 
                                + self._F_Misc[i-1] + self._F_GHGabate[i-1])) #Good

            self._tbox1[i] = (self._tbox1[i-1] *
                                    math.exp(self._tstep/self._d1) + self._teq1 *
                                    self._force[i] * (1-math.exp(self._tstep/self._d1))) #Good

            self._tbox2[i] = (self._tbox2[i-1] *
                                    math.exp(self._tstep/self._d2) + self._teq2 *
                                    self._force[i] * (1-math.exp(self._tstep/self._d2))) #Good

            self._tatm[i] = np.clip(self._tbox1[i-1] + self._tbox2[i-1], 0.5, 20) #Good

            self._cacc[i] = (self._ccatot[i-1] - (self._mmat[i-1] - self._mateq)) #Good

    #irfeqlhs(t)..    IRFt(t)   =E=  ((alpha(t)*emshare0*tau0*(1-exp(-100/(alpha(t)*tau0))))+(alpha(t)*emshare1*tau1*(1-exp(-100/(alpha(t)*tau1))))+(alpha(t)*emshare2*tau2*(1-exp(-100/(alpha(t)*tau2))))+(alpha(t)*emshare3*tau3*(1-exp(-100/(alpha(t)*tau3)))));
    #irfeqrhs(t)..    IRFt(t)   =E=  irf0+irC*Cacc(t)+irT*TATM(t);
    #force(t)..       FORC(t)    =E=  fco22x*((log((MAT(t)/mateq))/log(2))) + F_Misc(t)+F_GHGabate(t);

            if self._t[i-1] <= 16:
                self._CO2E_GHGabateB[i-1] = self._ECO2eGHGB2020 + ((self._ECO2eGHGB2100-self._ECO2eGHGB2020)/16)*(self._t[i-1]-1)
                self._F_Misc[i-1]=self._F_Misc2020 + ((self._F_Misc2100-self._F_Misc2020)/16)*(self._t[i-1]-1)
            else:
                self._CO2E_GHGabateB[i-1] = self._ECO2eGHGB2100
                self._F_Misc[i-1]=self._F_Misc2100
            
            self._F_GHGabate[i] = self._F_GHGabate2020*self._F_GHGabate[i-1] + self._F_GHGabate2100*self._CO2E_GHGabateB[i-1]*(1-MIUopt[i-1])

                    #This solves for the irfeqlhs and irfeqrhs equations
    def solve_alpha(self, t):
        # Define the equation for IRFt(t) using the given parameters and functions
        def equation(alpha, *args):
            return (
                (alpha * self._emshare0 * self._tau0 * (1 - np.exp(-100 / (alpha * self._tau0)))) +
                (alpha * self._emshare1 * self._tau1 * (1 - np.exp(-100 / (alpha * self._tau1)))) +
                (alpha * self._emshare2 * self._tau2 * (1 - np.exp(-100 / (alpha * self._tau2)))) +
                (alpha * self._emshare3 * self._tau3 * (1 - np.exp(-100 / (alpha * self._tau3)))) - 
                (self._irf0 + self._irC * self._cacc[t] + self._irT * self._tatm[t])
            )

        # Use root_scalar to solve for alpha
        alpha_initial_guess = 1.0
        alpha_solution = minimize_scalar(equation, alpha_initial_guess, bounds=(0.1, 100))
        return alpha_solution.x

    def runModel(self):
        pass    

print("Success")

fair_params = modelParams() #

fair_params.runModel()

