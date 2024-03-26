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


    def __init__(self, tfirst, tlast):

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
            fco22x   Forcings of equilibrium CO2 doubling (Wm-2)                        /3.93/       
        '''
        self._yr0 = 2020            #Calendar year that corresponds to model year zero
        self._emshare0 = 0.2173     #Carbon emissions share into Reservoir 0
        self._emshare1 = 0.224      #Carbon emissions share into Reservoir 1
        self._emshare2 = 0.2824     #Carbon emissions share into Reservoir 2
        self._emshare3 = 0.2763     #Carbon emissions share into Reservoir 3
        self._tau0 = 1000000        #Decay time constant for R0  (year)
        self._tau1 = 394.4          #Decay time constant for R1  (year)
        self._tau2 = 36.53          #Decay time constant for R2  (year)
        self._tau3 = 4.304          #Decay time constant for R3  (year)
    
    def runModel(self):
        pass


print("Success")