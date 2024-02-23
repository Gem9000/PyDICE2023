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

    def __init__(self, numtimes, tstep):
        
        #Indicator variable. 1 Denotes optimized and zero is base
        self.ifopt = 0

        #Population and technology 
        
        self.gama = 0.300 #Capital elasticity in production func.
        self.pop1 = 7752.9 #Initial world population 2020 (millions)
        self.popadj = 0.145 # Growth rate to calibrate to 2050 pop projection
        self.popasym = 10825 #Asymptotic population (millions)
        self.dk = 0.100 #Deprication on capital (per year)
        self.q1 = 135.7 #Initial world output 2020 (trillion 2019 USD)
        self.AL1 = 5.84 #Initial level of total factor productivity
        self.gA1 = 0.066 #Initial growth rate for TFP per 5 yrs 
        self.delA = 0.0015 #Decline rate of TFP per 5 yrs 

print("Success")
