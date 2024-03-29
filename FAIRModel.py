'''
Created 02/23/2024 
Author: George Moraites
Adapted From: domokane dice_params.py
'''

import numpy as np
import math
import csv

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
        self._time_increment = np.arrange(0,self.num_times+1)

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
        self._res0lom = np.zeros(num_times+1)
        self._res1lom =np.zeros(num_times+1)
        self._res2lom = np.zeros(num_times+1)
        self._res3lom = np.zeros(num_times+1)
        self._mmat = np.zeros(num_times+1)
        self._cacceq = np.zeros(num_times+1)
        self._force = np.zeros(num_times+1)
        self._tbox1eq = np.zeros(num_times+1)
        self._tbox2eq = np.zeros(num_times+1)
        self._tatmeq = np.zeros(num_times+1)
        self._irfeqlhs = np.zeros(num_times+1)
        self._irfeqrhs = np.zeros(num_times+1)


        #Initial conditions
        self._mateq[1] = self._mat0
        self._tatmeq[1] = self._tatm0
        self._res0lom[1] = self._res00
        self._res1lom[1] = self._res10
        self._res2lom[1] = self._res20
        self._res3lom[1] = self._res30
        self._tbox1eq[1] = self._tbox10
        self._tbox2eq[1] = self._tbox2eq

        for i in range (2, self._num_times+1):
            print("Success")


        
    def runModel(self):
        pass

print("Success")