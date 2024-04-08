'''
Created 02/23/2024 
Author: George Moraites
Adapted From: domokane dice_params.py
'''

import numpy as np
import math
import csv

from scipy.optimize import root_scalar

'''
** Equals old FAIR with recalibrated parameters for revised F2xco2 and Millar model.
** Deletes nonnegative reservoirs. See explanation below

EXPLAINATION:
IMPORTANT PROGRAMMING NOTE. Earlier implementations has reservoirs as non-negative.
However, these are not physical but mathematical solutions.
So, they need to be unconstrained so that can have negative emissions.
'''

class FAIRParams():


    def __init__(self, num_times, tstep):

    #########################################################
    #Initial Parameters for the FAIR model
    #########################################################

        '''
        PARAMETERS
            yr0     Calendar year that corresponds to model year zero         /2020/
            emshare0 Carbon emissions share into Reservoir 0   /0.2173/
            emshare1 Carbon emissions share into Reservoir 1    /0.224/
            emshare2 Carbon emissions share into Reservoir 2    /0.2824/
            emshare3 Carbon emissions share into Reservoir 3    /0.2763/
            tau0    Decay time constant for R0  (year)                            /1000000/
            tau1    Decay time constant for R1  (year)                            /394.4/
            tau2    Decay time constant for R2  (year)       /36.53/
            tau3    Decay time constant for R3  (year) /4.304/

            teq1    Thermal equilibration parameter for box 1 (m^2 per KW)         /0.324/
            teq2    Thermal equilibration parameter for box 2 (m^2 per KW)        /0.44/
            d1      Thermal response timescale for deep ocean (year)               /236/
            d2      Thermal response timescale for upper ocean (year)              /4.07/

            irf0    Pre-industrial IRF100 (year)                                        /32.4/
            irC      Increase in IRF100 with cumulative carbon uptake (years per GtC)    /0.019/
            irT      Increase in IRF100 with warming (years per degree K)                /4.165/
            fco22x   Forcings of equilibrium CO2 doubling (Wm-2)                         /3.93/
                                                                     
            ** INITIAL CONDITIONS TO BE CALIBRATED TO HISTORY
            ** CALIBRATION
            mat0   Initial concentration in atmosphere in 2020 (GtC)       /886.5128014/

            res00  Initial concentration in Reservoir 0 in 2020 (GtC)      /150.093 /
            res10  Initial concentration in Reservior 1 in 2020 (GtC)      /102.698 /
            res20  Initial concentration in Reservoir 2 in 2020 (GtC)      /39.534  /
            res30  Initial concentration in Reservoir 3 in 2020 (GtC)      / 6.1865 /

            mateq      Equilibrium concentration atmosphere  (GtC)            /588   /
            tbox10    Initial temperature box 1 change in 2020 (C from 1765)  /0.1477  /
            tbox20    Initial temperature box 2 change in 2020 (C from 1765)  /1.099454/
            tatm0     Initial atmospheric temperature change in 2020          /1.24715 /     
        '''
        self._tstep = tstep #Years in period 

        self._yr0 = 2020            #Calendar year that corresponds to model year zero
        self._emshare0 = 0.2173     #Carbon emissions share into Reservoir 0
        self._emshare1 = 0.224      #Carbon emissions share into Reservoir 1
        self._emshare2 = 0.2824     #Carbon emissions share into Reservoir 2
        self._emshare3 = 0.2763     #Carbon emissions share into Reservoir 3
        self._tau0 = 1000000        #Decay time constant for R0  (year)
        self._tau1 = 394.4          #Decay time constant for R1  (year)
        self._tau2 = 36.53          #Decay time constant for R2  (year)
        self._tau3 = 4.304          #Decay time constant for R3  (year)
    
        self._teq1 = 0.324          #Thermal equilibration parameter for box 1 (m^2 per KW)
        self._teq2 = 0.44           #Thermal equilibration parameter for box 2 (m^2 per KW)
        self._d1 = 236              #Thermal response timescale for deep ocean (year)
        self._d2 = 4.07             #Thermal response timescale for upper ocean (year)

        self._irf0 = 32.4           #Pre-industrial IRF100 (year)
        self._irC  = 0.019          #Increase in IRF100 with cumulative carbon uptake (years per GtC)
        self._irT =  4.165          #Increase in IRF100 with warming (years per degree K)   
        self._fco22x = 3.93         #Forcings of equilibrium CO2 doubling (Wm-2)

        self._mat0 = 886.5128014    #Initial concentration in atmosphere in 2020 (GtC)
        self._res00 = 150.093       #Initial concentration in Reservoir 0 in 2020
        self._res10 = 102.698       #Initial concentration in Reservoir 1 in 2020
        self._res20 = 39.534        #Initial concentration in Reservoir 2 in 2020
        self._res30 = 6.1865        #Initial concentration in Reservoir 3 in 2020

        self._mateq = 588           #Equilibrium concentration atmosphere
        self._tbox10 = 0.1477       #Initial temperature box 1 change in 2020
        self._tbox20 = 1.099454     #Initial temperature box 2 change in 2020
        self._tatm0 =  1.24715      #Initial atmospheric temperature change in 2020 

        #Num times should be 81
        #time increment should be 5 years
        self._num_times = num_times
        self._time_increment = np.arange(0,self._num_times+1)

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
        self._res0lom = np.zeros(num_times+2)
        self._res1lom =np.zeros(num_times+2)
        self._res2lom = np.zeros(num_times+2)
        self._res3lom = np.zeros(num_times+2)
        self._mmat = np.zeros(num_times+2)
        self._cacceq = np.zeros(num_times+2)
        self._force = np.zeros(num_times+2)
        self._tbox1eq = np.zeros(num_times+2)
        self._tbox2eq = np.zeros(num_times+2)
        self._tatmeq = np.zeros(num_times+2)
        self._irfeqlhs = np.zeros(num_times+2)
        self._irfeqrhs = np.zeros(num_times+2)
        self._alpha_t = np.zeros(num_times+2)

        #Filler for the model
        self._F_Misc = np.zeros(num_times+2)

        #Fillers for equations that are in the DICE model equation
        self._eco2 = np.zeros(num_times+2)
        self._F_GHGabate = np.zeros(num_times+2)
        self._CCATOT = np.zeros(num_times+2)
        
        #Timestep counter
        self._t = np.arange(0,num_times+2)

        #Initial conditions
        self._mmat[1] = self._mat0
        self._tatmeq[1] = self._tatm0
        self._res0lom[1] = self._res00
        self._res1lom[1] = self._res10
        self._res2lom[1] = self._res20
        self._res3lom[1] = self._res30
        self._tbox1eq[1] = self._tbox10
        self._tbox2eq[1] = self._tbox20
    
        for i in range(1, self._num_times + 1):

            #Solve for alpha(t) in each time period 
            self._alpha_t[i+1] = self.solve_alpha(i)

            self._res0lom[i+1] = (self._emshare0 * self._tau0 * self.solve_alpha(i+1) * 
                                    (self._eco2[i+1] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau0 * self.solve_alpha(i+1)))) + 
                                    self._res0lom[i] * math.exp(-self._tstep / (self._tau0 * self.solve_alpha(i+1))))

            self._res1lom[i+1] = (self._emshare1 * self._tau1 * self.solve_alpha(i+1) * 
                                    (self._eco2[i+1] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau1 * self.solve_alpha(i+1)))) + 
                                    self._res1lom[i] * math.exp(-self._tstep / (self._tau1 * self.solve_alpha(i+1))))

            self._res2lom[i+1] = (self._emshare2 * self._tau2 * self.solve_alpha(i+1) * 
                                    (self._eco2[i+1] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau2 * self.solve_alpha(i+1)))) + 
                                    self._res2lom[i] * math.exp(-self._tstep / (self._tau2 * self.solve_alpha(i+1))))

            self._res3lom[i+1] = (self._emshare3 * self._tau3 * self.solve_alpha(i+1) * 
                                    (self._eco2[i+1] / 3.667) * 
                                    (1 - math.exp(-self._tstep / (self._tau3 * self.solve_alpha(i+1)))) + 
                                    self._res3lom[i] * math.exp(-self._tstep / (self._tau3 * self.solve_alpha(i+1))))

            calculated_mmat = self._mateq + self._res0lom[i+1] + self._res1lom[i+1] + self._res2lom[i+1] + self._res3lom[i+1]
            if calculated_mmat < 20:
                self._mmat[i+1] = 20
            else:
                self._mmat[i+1] = calculated_mmat
                
            self._force[i] = (self._fco22x * ((math.log((self._mmat[i]+1e-9/self._mateq))/math.log(2)) 
                                + self._F_Misc[i] + self._F_GHGabate[i]))

            self._tbox1eq[i+1] = (self._tbox1eq[i] *
                                    math.exp(self._tstep/self._d1) + self._teq1 *
                                    self._force[i+1] * (1-math.exp(self._tstep/self._d1)))  

            self._tbox2eq[i+1] = (self._tbox2eq[i] *
                                    math.exp(self._tstep/self._d2) + self._teq2 *
                                    self._force[i+1] * (1-math.exp(self._tstep/self._d2)))

            self._tatmeq[i+1] = np.clip(self._tbox1eq[i+1] + self._tbox2eq[i+1], 0.5, 20)

            self._cacceq[i] = (self._CCATOT[i] - (self._mmat[i] - self._mateq))

        #Adding in additional code for creating a CSV with the fair model 
        #Equation values

        if 1==1:

            f = open("./results/FAIR_Model.csv" , mode = "w", newline='')
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                
            header = []
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
        alpha_solution = root_scalar(equation, alpha_initial_guess, bracket=(0.1, 100),method= 'brenth')
        return alpha_solution.root

    def runModel(self):
        pass    

print("Success")

fair_params = FAIRParams(num_times=81, tstep= 5)

fair_params.runModel()

