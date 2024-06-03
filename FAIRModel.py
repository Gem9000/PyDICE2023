'''
Created 02/23/2024 
Author: George Moraites
Adapted From: domokane dice_params.py
Edited by Jacob Wessel
'''

import numpy as np
import math
import csv

#from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
from scipy.optimize import fsolve

'''
** Equals old FAIR with recalibrated parameters for revised F2xco2 and Millar model.
** Deletes nonnegative reservoirs. EXPLANATION:
    Earlier implementations have reservoirs as non-negative.
    However, these are not physical but mathematical solutions.
    So, they need to be unconstrained so they can have negative emissions.
'''

class FAIRParams():

    def __init__(self, num_times, tstep):

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

        #Num times should be 82
        #time increment should be 5 years
        self._num_times      = num_times
        self._time_increment = np.arange(0,self._num_times+1)

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

        ### GEORGE
        # self._e1 = 37.56  #Industrial emissions 2020 (GtCO2 per year)  
        # self._miu1 = 0.05  #Emissions control rate historical 2020 
        # self._q1 = 135.7 #Initial world output 2020 (trillion 2019 USD)
        # self._gsigma1 = -0.015 #Initial growth of sigma (per year)  
        # self._AL1 = 5.84 #Initial level of total factor productivity
        # self._gama = 0.300 #Capital elasticity in production func.
        # self._asymgsig = -0.005 #Asympototic gsigma
        # self._pop1 = 7752.9 #Initial world population 2020 (millions)
        # self._popasym = 10825 #Asymptotic population (millions)
        # self._popadj = 0.145 # Growth rate to calibrate to 2050 pop projection
        # self._k0       =  295  #Initial capital stock calibrated (1012 2019 USD)
        # self._MILLE = 1000.0

        '''
        EQUATIONS
        FORCE(t)        Radiative forcing equation
        RES0LOM(t)      Reservoir 0 law of motion
        RES1LOM(t)      Reservoir 1 law of motion
        RES2LOM(t)      Reservoir 2 law of motion
        RES3LOM(t)      Reservoir 3 law of motion
        MMAT(t)         Atmospheric concentration equation
        Cacceq(t)       Accumulated carbon in sinks equation
        TATMEQ(t)       Temperature-climate equation for atmosphere
        TBOX1EQ(t)      Temperature box 1 law of motion
        TBOX2EQ(t)      Temperature box 2 law of motion
        IRFeqLHS(t)     Left-hand side of IRF100 equation
        IRFeqRHS(t)     Right-hand side of IRF100 equation
        '''
        
        #Creating Size arrays so we can index from 1 instead of zero 
        self._res0lom         = np.zeros(num_times+1)
        self._res1lom         = np.zeros(num_times+1)
        self._res2lom         = np.zeros(num_times+1)
        self._res3lom         = np.zeros(num_times+1)
        self._mmat            = np.zeros(num_times+1)
        self._cacceq          = np.zeros(num_times+1)
        self._force           = np.zeros(num_times+1)
        self._tbox1eq         = np.zeros(num_times+1)
        self._tbox2eq         = np.zeros(num_times+1)
        self._tatmeq          = np.zeros(num_times+1)
        self._irfeqlhs        = np.zeros(num_times+1)
        self._irfeqrhs        = np.zeros(num_times+1)
        self._alpha_t         = np.zeros(num_times+1)
        self._calculated_mmat = np.zeros(num_times+1)

        # variables from nonco2 forcings include
        self._eco2            = np.zeros(num_times+1) #Total CO2 emissions (GtCO2 per year)
        self._eco2e           = np.zeros(num_times+1) #Total CO2e emissions including abateable nonCO2 GHG (GtCO2 per year)
        self._F_GHGabate      = np.zeros(num_times+1) #Forcings abateable nonCO2 GHG
        self._eind            = np.zeros(num_times+1) #Industrial CO2 emissions (GtCO2 per yr)
        self._eland           = np.zeros(num_times+1) #Emissions from deforestation (GtCO2 per year)
        self._CCATOT          = np.zeros(num_times+1)

        self._CO2E_GHGabateB   = np.zeros(num_times+1) #Abateable non-CO2 GHG emissions base
        self._CO2E_GHGabateact = np.zeros(num_times+1) #Abateable non-CO2 GHG emissions base (actual)
        self._F_Misc           = np.zeros(num_times+1) #Non-abateable forcings (GHG and other)
        self._sigmatot         = np.zeros(num_times+1) #Emissions output ratio for CO2e
        self._FORC_CO2         = np.zeros(num_times+1) #CO2 Forcings

        #Won't be optimized until the Pyomo optimizer is used
        self._MIUopt = np.zeros(num_times+1)
        Sopt = np.zeros(num_times)
        x = np.zeros(2*num_times+1)

        #Timestep counter
        self._t = np.arange(0,num_times+1)

        #Initial conditions
        self._mmat[1]       = self._mat0
        self._tatmeq[1]     = self._tatm0
        self._res0lom[1]    = self._res00
        self._res1lom[1]    = self._res10
        self._res2lom[1]    = self._res20
        self._res3lom[1]    = self._res30
        self._tbox1eq[1]    = self._tbox10
        self._tbox2eq[1]    = self._tbox20
        self._eland[1]      = self._eland0
        self._F_GHGabate[1] = self._F_GHGabate2020

        ### GEORGE
        #Set relevant values using the parameters above
        # self._k[1] = self._k0
        # self._l[1] = self._pop1 #Population 
        # self._al[1] = self._AL1 #Initial total factor productivity
        # self._gsig[1] = self._gsigma1 #Initial growth of sigma
        # self._sig1 = (self._e1)/(self._q1*(1-self._miu1))
        # self._sigma[1] = self._sig1 #sig1 did not have a value in the gms code
        # self._ygross[1] = self._al[1] * ((self._l[1]/self._MILLE)**(1.0-self._gama)) * self._k[1]**self._gama  #Gross world product GROSS of abatement and damages (trillions 2019 USD per year)
        # self._eco2[1] = ((self._sigma[1] * self._ygross[1] + self._eland[1]) * (1-self._MIUopt[1])) #New

        #This code wasn't running for debugging,
        #But we don't need the optimal values right now to check 
        #Most of this
        '''
        #Initialize the equations
        for i in range(1, num_times+1):
            MIUopt[i] = x[i-1]          #Optimal emissions control rate GHGS
            Sopt[i] = x[num_times + i-1]   #Gross savings rate as fraction of gross world product
        '''

        #Changing this to match the format of the DICE Model 
        #If i+1 should be i
        #If i should be i-1

        for i in range(2, self._num_times + 1):

            #Must be computed since it's used in the eco2 equation
    #        self._l[i] = self._l[i-1]*(self._popasym / self._l[i-1])**self._popadj # Level of population and labor 
    #        self._gsig[i] = min(self._gsigma1*self._delgsig **((self._t[i]-1)), self._asymgsig) #Change in rate of sigma (represents rate of decarbonization)
    #        self._sigma[i] = self._sigma[i-1]*math.exp(5*self._gsig[i-1])

            self._eland[i-1] = self._eland0 * (1 - self._deland) ** (self._t[i-1]-1)

            #Depends on the t-1 time period
            self._CCATOT[i] = self._CCATOT[i-1] + self._eco2[i-1] * (5/3.666)
    #        self._ygross[i-1] = self._al[i-1] * ((self._l[i-1]/self._MILLE)**(1.0-self._gama)) * self._k[i-1]**self._gama  #Gross world product GROSS of abatement and damages (trillion 2019USD/yr)

    #        self._eco2[i-1] = (self._sigma[i-1] * self._ygross[i-1] + self._eland[i-1]) * (1-self._MIUopt[i-1]) #New
    #        self._eind[i-1] = self._sigma[i-1] * self._ygross[i-1] * (1.0 - self._MIUopt[i-1])
            
    #        self._eco2e[i-1] = (self._sigma[i-1] * self._ygross[i-1] + self._eland[i-1] + self._CO2E_GHGabateB[i-1]) * (1-self._MIUopt[i-1]) #New

            #Solve for alpha(t) in each time period 
            self._alpha_t[i] = self.solve_alpha(i-1)

            self._res0lom[i] = (self._emshare0 * self._tau0 * self.solve_alpha(i-1) * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau0 * self.solve_alpha(i-1)))) + 
                                    self._res0lom[i-1] * math.exp(-self._tstep / (self._tau0 * self.solve_alpha(i-1))))

            self._res1lom[i] = (self._emshare1 * self._tau1 * self.solve_alpha(i-1) * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau1 * self.solve_alpha(i-1)))) + 
                                    self._res1lom[i-1] * math.exp(-self._tstep / (self._tau1 * self.solve_alpha(i))))

            self._res2lom[i] = (self._emshare2 * self._tau2 * self.solve_alpha(i-1) * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau2 * self.solve_alpha(i-1)))) + 
                                    self._res2lom[i-1] * math.exp(-self._tstep / (self._tau2 * self.solve_alpha(i-1))))

            self._res3lom[i] = (self._emshare3 * self._tau3 * self.solve_alpha(i) * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau3 * self.solve_alpha(i-1)))) + 
                                    self._res3lom[i-1] * math.exp(-self._tstep / (self._tau3 * self.solve_alpha(i-1))))

            self._calculated_mmat[i] = self._mateq + self._res0lom[i] + self._res1lom[i] + self._res2lom[i] + self._res3lom[i]
            if self._calculated_mmat[i] < 10:
                self._mmat[i] = 10
            else:
                self._mmat[i] = self._calculated_mmat[i-1]
                
            self._force[i] = (self._fco22x * ((math.log((self._mmat[i-1]+1e-9/self._mateq))/math.log(2)) 
                                + self._F_Misc[i-1] + self._F_GHGabate[i-1])) #Good

            self._tbox1eq[i] = (self._tbox1eq[i-1] *
                                    math.exp(self._tstep/self._d1) + self._teq1 *
                                    self._force[i] * (1-math.exp(self._tstep/self._d1))) #Good

            self._tbox2eq[i] = (self._tbox2eq[i-1] *
                                    math.exp(self._tstep/self._d2) + self._teq2 *
                                    self._force[i] * (1-math.exp(self._tstep/self._d2))) #Good

            self._tatmeq[i] = np.clip(self._tbox1eq[i-1] + self._tbox2eq[i-1], 0.5, 20) #Good

            self._cacceq[i] = (self._CCATOT[i-1] - (self._mmat[i-1] - self._mateq)) #Good

            if self._t[i-1] <= 16:
                self._CO2E_GHGabateB[i-1] = self._ECO2eGHGB2020 + ((self._ECO2eGHGB2100-self._ECO2eGHGB2020)/16)*(self._t[i-1]-1)
                self._F_Misc[i-1]=self._F_Misc2020 + ((self._F_Misc2100-self._F_Misc2020)/16)*(self._t[i-1]-1)
            else:
                self._CO2E_GHGabateB[i-1] = self._ECO2eGHGB2100
                self._F_Misc[i-1]=self._F_Misc2100
            
            self._F_GHGabate[i] = self._F_GHGabate2020*self._F_GHGabate[i-1] + self._F_GHGabate2100*self._CO2E_GHGabateB[i-1]*(1-self._MIUopt[i-1])

        #Adding in additional code for creating a CSV with the fair model 
        #Equation values

        if 1==1:

            f = open("./results/FAIR_Model.csv" , mode = "w", newline='')
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                
            header = []
            header.append("Time Period")
            header.append("Alpha(t)")
            header.append("Reservoir 0")
            header.append("Reservoir 1")
            header.append("Reservoir 2")
            header.append("Reservoir 3")
            header.append("Atmospheric Concentration Equation")
            header.append("Accumulated Carbon in Sinks")
            header.append("Temp-Climate Equation for Atmosphere")
            header.append("Temp. box 1")
            header.append("Temp. box 2")
            header.append("Radiative Forcing")
            header.append("Abateable non-CO2 GHG emissions base")
            header.append("Non-abateable forcings (GHG and other)")
            header.append("Emissions from deforestation (GtCO2 per year)")
            writer.writerow(header)

            num_rows = self._num_times + 1

            for i in range(0, num_rows):
                row = []
                row.append(i)
                row.append(self._alpha_t[i])
                row.append(self._res0lom[i])
                row.append(self._res1lom[i])
                row.append(self._res2lom[i])
                row.append(self._res3lom[i])
                row.append(self._mmat[i])
                row.append(self._cacceq[i])
                row.append(self._tatmeq[i])
                row.append(self._tbox1eq[i])
                row.append(self._tbox2eq[i])
                row.append(self._force[i])
                row.append(self._CO2E_GHGabateB[i])
                row.append(self._F_Misc[i])
                row.append(self._eland[i])

                writer.writerow(row)

            f.close()

        elif 1==2: 
            print("Variance of per capita consumption:", self._alpha_t) # Need to check
            print("Precationary rate of return:",self._res0lom) # Need to check
            print("STP factor without precationary factor:",self._res1lom) # Need to check
            print("STP factor with precationary factor:",self._res2lom) # Need to check
            print("Labour:", self._res3lom) # Need to check
            print("Growth rate of productivity", self._mmat) # Need to check
            print("Productivity", self._cacceq) # Need to check
            print("Carbon price based case", self._tatmeq) # Need to check
            print("Backstop price", self._tbox1eq) # Need to check
            print("Change in sigma", self._tbox2eq) # Need to check
            print("CO2 output ratio:", self._force) # Need to check

        else:
            print("SOME CHECKING TO BE DONE")
    

                    #This solves for the irfeqlhs and irfeqrhs equations
    def solve_alpha(self, t):
        # Define the equation for IRFt(t) using the given parameters and functions
        def equation(alpha, *args):
            return (
                (alpha * self._emshare0 * self._tau0 * (1 - np.exp(-100 / (alpha * self._tau0)))) +
                (alpha * self._emshare1 * self._tau1 * (1 - np.exp(-100 / (alpha * self._tau1)))) +
                (alpha * self._emshare2 * self._tau2 * (1 - np.exp(-100 / (alpha * self._tau2)))) +
                (alpha * self._emshare3 * self._tau3 * (1 - np.exp(-100 / (alpha * self._tau3)))) - 
                (self._irf0 + self._irC * self._cacceq[t] + self._irT * self._tatmeq[t])
            )

        # Use root_scalar to solve for alpha
        alpha_initial_guess = 1.0
        alpha_solution = minimize_scalar(equation, alpha_initial_guess, bounds=(0.1, 100))
        return alpha_solution.x

    def runModel(self):
        pass    

print("Success")

fair_params = FAIRParams(num_times=82, tstep= 5)

fair_params.runModel()

