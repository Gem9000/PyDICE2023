'''
Author: George Moraites
Edited by Jacob Wessel
'''

import csv
import numpy as np
from numba import njit
from matplotlib import pyplot
import seaborn
import math 
seaborn.set_theme(style='ticks')

from FAIRModel import FAIRParams

###############################################################################
# All arrays have been shifted to have length numtimes + 1 and to start at 1
###############################################################################
# The optimizer calls objFn() which returns a float for the optimizer. However,
# internally this calls simulateDynamics(), which can return either the utility
# to be maximized or the state information.
###############################################################################

@njit(cache=True, fastmath=True)
def objFn(x, *args):
    """ Pass-through function returning a single float value of the objective function """
    out = simulateDynamics(x, *args)
    return out[0, 0]

###############################################################################

def simulateDynamics(self, SRF, x, sign, outputType, num_times,
                         tstep, al, ll, sigma,
                         cost1, etree,
                         scale1, scale2,
                         ml0, mu0, mat0, cca0,
                         a1, a2, a3,
                         c1, c3, c4,
                         b11, b12, b21, b22, b32, b23, b33,
                         fco22x, t2xco2, rr, gama,
                         tocean0, tatm0, elasmu, prstp, expcost2,
                         k0, dk, pbacktime, CumEmiss0):
        """ Simulation of DICE 2023 model dynamics """
    
        LOG2 = np.log(2)
        L = ll  # renamed to uppercase in equations
        MILLE = 1000 # conversion factor
    
        # Ensure indexing starts at 1 to allow comparison with matlab
        MIUopt = np.zeros(num_times+1)
        Sopt = np.zeros(num_times+1)
    
###############################################################################
# Set the optimization variables
###############################################################################

        for i in range(1, num_times+1):
            MIUopt[i] = x[i-1]          #Optimal emissions control rate GHGS
            Sopt[i] = x[num_times + i-1]   #Gross savings rate as fraction of gross world product
    
###########################################################################
#Variables and nonnegative variables equations
###########################################################################
        
        #Make an instance of the FAIR PARAMETERS class
        instance = FAIRParams()

        #These are already initilized to zero in the FAIR class 

        FORCING = instance._force              #Radiative forcing equation
        TATM = instance._tatmeq                #Initial atmospheric temperature change in 2020
        TBOX1 = instance._tbox1eq              #Temperaute box 1 law of motion
        TBOX2 = instance._tbox2eq              #Temperaute box 1 law of motion
        RES0 = instance._res0lom               #Reservoir 0 law of motion
        RES1 = instance._res1lom               #Reservoir 1 law of motion
        RES2 = instance._res2lom               #Reservoir 2 law of motion
        RES3 = instance._res3lom               #Reservoir 3 law of motion
        MAT = instance._mmat                   #Atmospheric concentration equation
        CACC = instance._cacceq                #Accumulated carbon in sinks equation
        
        ECO2 = instance._eco2                     #Total CO2 emissions (GtCO2 per year)
        ECO2E = instance._eco2e                   #Total CO2e emissions including abateable nonCO2 GHG (GtCO2 per year)
        EIND = instance._eind                     #Industrial Emissions (GtCO2 per year)
        CO2E_GHGabateB = instance._CO2E_GHGabateB #Abateable non-CO2 GHG emissions base
        F_GHGabate = instance._F_GHGabate         #Forcings abateable nonCO2 GHG
        ELAND = instance._eland                   #Emissions from deforestation (GtCO2 per year)
        
        C = np.zeros(num_times+1)              #Consumption (trillions 2019 US dollars per year)
        K = np.zeros(num_times+1)              #Capital stock (trillions 2019 US dollars)
        CPC = np.zeros(num_times+1)            #Per capita consumption (thousands 2019 USD per year)
        I =  np.zeros(num_times+1)             #Investment (trillions 2019 USD per year)
        Y  = np.zeros(num_times+1)             #Gross world product net of abatement and damages (trillions 2019 USD per year)
        YGROSS = np.zeros(num_times+1)         #Gross world product GROSS of abatement and damages (trillions 2019 USD per year)
        YNET  = np.zeros(num_times+1)          #Output net of damages equation (trillions 2019 USD per year)
        DAMAGES = np.zeros(num_times+1)        #Damages (trillions 2019 USD per year)
        DAMFRAC = np.zeros(num_times+1)        #Damages as fraction of gross output
        ABATECOST = np.zeros(num_times+1)      #Cost of emissions reductions  (trillions 2019 USD per year)
        MCABATE  = np.zeros(num_times+1)       #Marginal cost of abatement (2019$ per ton CO2)
        CCATOT  = np.zeros(num_times+1)        #Total carbon emissions (GtC)
        PERIODU  = np.zeros(num_times+1)       #One period utility function
        CPRICE   =  np.zeros(num_times+1)      #Carbon price (2019$ per ton of CO2)
        TOTPERIODU = np.zeros(num_times+1)     #Period utility
        UTILITY    = np.zeros(num_times+1)     #Welfare function
        RFACTLONG = np.zeros(num_times+1)
        RSHORT    = np.zeros(num_times+1)      #Real interest rate with precautionary (per year on year)
        RLONG   = np.zeros(num_times+1)        #Real interest rate from year 0 to T
        EIND = np.zeros(num_times+1)           #Industrial Emissions (GtCO2 per year)

#Emissions and Damages
        CCATOTEQ = np.zeros(num_times+1)       #Cumulative total carbon emissions
        DAMFRACEQ = np.zeros(num_times+1)      #Equation for damage fraction
        DAMEQ = np.zeros(num_times+1)          #Damage equation
        ABATEEQ = np.zeros(num_times+1)        #Cost of emissions reductions equation
        MCABATEEQ = np.zeros(num_times+1)      #Equation for MC abatement
        CARBPRICEEQ = np.zeros(num_times+1)    #Carbon price equation from abatement
#Economic variables
        YGROSSEQ = np.zeros(num_times+1)       #Output gross equation
        YNETEQ = np.zeros(num_times+1)         #Output net of damages equation
        YY = np.zeros(num_times+1)             #Output net equation
        CC = np.zeros(num_times+1)             #Consumption equation
        CPCE = np.zeros(num_times+1)           #Per capita consumption definition
        SEQ = np.zeros(num_times+1)            #Savings rate equation
        KK =  np.zeros(num_times+1)            #Capital balance equation
        RSHORTEQ = np.zeros(num_times+1)       #Short-run interest rate equation
        RLONGEQ = np.zeros(num_times+1)        #Long-run interest rate equation
        RFACTLONGEQ = np.zeros(num_times+1)    #Long interest factor
#Utility
        TOTPERIODUEQ = np.zeros(num_times+1)   #Period utility
        PERIODUEQ = np.zeros(num_times+1)      #Instantaneous utility function equation
        #UTILEQ =  np.zeros(num_times+1)        #Objective function


###################################Initializing Equations#################################
        K[1] = k0
        CCATOTEQ[1] = CumEmiss0
        
        YGROSS[1] = al[1] * ((L[1]/MILLE)**(1.0-gama)) * K[1]**gama  #Gross world product GROSS of abatement and damages (trillions 2019 USD per year)

        ECO2[1] = (sigma[1] * YGROSS[1] + etree[1]) * (1-MIUopt) #New
        EIND[1] = sigma[1] * YGROSS[1] * (1.0 - MIUopt[1])
        
        ECO2E[1] = (sigma[1] * YGROSS[1] + etree[1] + CO2E_GHGabateB[1]) * (1-MIUopt) #New
        
        CCATOT[1] = CCATOT[1] + ECO2[1]*(5/3.666)
        DAMFRAC[1] = (a1 * TATM[1] + a2 * TATM[1] ** a3)  
        DAMAGES[1] = YGROSS[1] * DAMFRAC[1]

        DAMFRAC[1] = a1*TATM[1] + a2*TATM[1]**a3
        DAMAGES[1] = YGROSS[1] * DAMFRAC[1]
        ABATECOST[1] = YGROSS[1] * cost1[1] * MIUopt[1]**expcost2
        MCABATE[1] = pbacktime[1] * MIUopt[1]**(expcost2-1)
        CPRICE[1] = pbacktime[1] * (MIUopt[1])**(expcost2-1)

        CACC[1] = cca0  # DOES NOT START TILL PERIOD 2

        ########################Economic Initializations##############
        YNET[1] = YGROSS[1] * (1-DAMFRAC[1])
        Y[1] = YNET[1] - ABATECOST[1]
        C[1] = Y[1] - I[1]
        CPC[1] = MILLE * C[1]/L[1]
        I[1] = Sopt[1] * Y[1]

        RFACTLONG[0] = 1000000
        RFACTLONG[1] = 1000000
        RLONG[1] = (-math.log(RFACTLONG[1]/SRF)/5) #NEW
        RSHORT[1] = (-math.log(RFACTLONG[1]/RFACTLONG[0])/5) #NEW 

        #########################Welfare Functions###################
        PERIODU[1] = ((C[1]*MILLE/L[1])**(1.0-elasmu)-1.0) / (1.0 - elasmu) - 1.0
        TOTPERIODU[1] = PERIODU[1] * L[1] * rr[1]

        #Many of the equations in the module have been put into a separate DFAIR class
        # The equations that the DFAIR modules rely on should be calculated and back checked 
        for i in range(2, num_times+1):

            #Depends on the t-1 time period
            CCATOT[i] = CCATOT[i-1] + ECO2[i-1] * (5/3.666)
            YGROSS[i] = al[i] * ((L[i]/MILLE)**(1.0-gama)) * K[i]**gama  #Gross world product GROSS of abatement and damages (trillions 20i9 USD per year)

            ECO2[i] = (sigma[i] * YGROSS[i] + ELAND[i]) * (1-MIUopt) #New
            EIND[i] = sigma[i] * YGROSS[i] * (1.0 - MIUopt[i])
            
            ECO2E[i] = (sigma[i] * YGROSS[i] + ELAND[i] + CO2E_GHGabateB[i]) * (1-MIUopt) #New
            
            CCATOT[i] = CCATOT[i] + ECO2[i]*(5/3.666)
            DAMFRAC = (a1 * TATM[i] + a2 * TATM[i] ** a3)  
            DAMAGES = YGROSS[i] * DAMFRAC[i]

            DAMFRAC[i] = a1*TATM[i] + a2*TATM[i]**a3
            DAMAGES[i] = YGROSS[i] * DAMFRAC[i]
            ABATECOST[i] = YGROSS[i] * cost1[i] * MIUopt[i]**expcost2
            MCABATE[i] = pbacktime[i] * MIUopt[i]**(expcost2-1)
            CPRICE[i] = pbacktime[i] * (MIUopt[i])**(expcost2-1)

            ########################Economic##############################
            YNET[i] = YGROSS[i] * (i-DAMFRAC[i])
            Y[i] = YNET[i] - ABATECOST[i]
            C[i] = Y[i] - I[i]
            CPC[i] = MILLE * C[i]/L[i]
            I[i] = Sopt[i] * Y[i]

            # this equation is a <= inequality and needs to be treated as such
            #K[i] = (1.0 - dk)**tstep * K[i-1] + tstep * I[i]

            RFACTLONG[i] = (SRF * (CPC[i]/CPC[i-1])**(-elasmu)*rr[i]) #Modified/New
            RLONG[i] = -math.log(RFACTLONG[i]/SRF)/(5*(i-1)) #NEW
            RSHORT[i] = (-math.log(RFACTLONG[i]/RFACTLONG[i-1])/5) #NEW 

            #########################Welfare Functions###################
            PERIODU[i] = ((C[i]*MILLE/L[i])**(1.0-elasmu)-1.0) / (1.0 - elasmu) - 1.0
            TOTPERIODU[i] = PERIODU[i] * L[i] * rr[i]

        output = np.zeros((num_times,50))
        
        #Defining some control logic for the model
        if outputType == 0:
            
            #Find the Total utility
            resultUtility = tstep  * scale1 * np.sum(TOTPERIODU) + scale2
            resultUtility *= sign 
            output[0,0] = resultUtility

        elif outputType == 1:

            #Implemented in DICE2016 but might be depricated. Needs checked
            """
               # EXTRA VALUES COMPUTED LATER
            CO2PPM = np.zeros(num_times+1)
            for i in range(1, num_times):
                CO2PPM[i] = MAT[i] / 2.13

            SOCCC = np.zeros(num_times+1)
            for i in range(1, num_times):
                SOCCC[i] = -999.0
            """

            for iTime in range(1, num_times+1):

                col = 0
                jTime = iTime - 1
                output[jTime, col] = EIND[iTime]
                col += 1  # 0
                output[jTime, col] = ECO2[iTime]
                col += 1  # 1
                output[jTime, col] = ECO2E[iTime]
                col += 1  # 2
                output[jTime, col] = TATM[iTime]
                col += 1  # 3
                output[jTime, col] = Y[iTime]
                col += 1  # 4
                output[jTime, col] = DAMFRAC[iTime]
                col += 1  # 5
                output[jTime, col] = CPC[iTime]
                col += 1  # 6
                output[jTime, col] = CPRICE[iTime]
                col += 1  # 7
                output[jTime, col] = MIUopt[iTime]
                col += 1  # 8
                output[jTime, col] = rr[iTime]
                col += 1  # 9
                output[jTime, col] = ll[iTime]
                col += 1  # 10
                output[jTime, col] = al[iTime]
                col += 1  # 11
                output[jTime, col] = YGROSS[iTime]
                col += 1  # 12
                output[jTime, col] = K[iTime]
                col += 1  # 13
                output[jTime, col] = Sopt[iTime]
                col += 1  # 14
                output[jTime, col] = I[iTime]
                col += 1  # 15
                output[jTime, col] = YNET[iTime]
                col += 1  # 16
                output[jTime, col] = CCATOT[iTime]
                col += 1  # 17
                output[jTime, col] = DAMAGES[iTime]
                col += 1  # 18
                output[jTime, col] = ABATECOST[iTime]
                col += 1  # 19
                output[jTime, col] = MCABATE[iTime]
                col += 1  # 20
                output[jTime, col] = C[iTime]
                col += 1  # 21
                output[jTime, col] = PERIODU[iTime]
                col += 1  # 22
                output[jTime, col] = TOTPERIODU[iTime]
                col += 1  # 23
                output[jTime, col] = RFACTLONG[iTime]
                col += 1  # 24
                output[jTime, col] = RLONG[iTime]
                col += 1  # 25
                output[jTime, col] = RSHORT[iTime]
                col += 1  # 26
                output[jTime,col] = ELAND[iTime]
                col += 1 # 27
            return output

        else:
            raise Exception("Unknown output type.")

        return output
    

def dumpState(years, output, filename):

    f = open(filename, mode="w", newline='')
    writer = csv.writer(f, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)

    header = []
    header.append("EIND")
    header.append("ECO2")
    header.append("ECO2E")
    header.append("TATM")
    header.append("Y")
    header.append("DAMFRAC")
    header.append("CPC")
    header.append("CPRICE")
    header.append("MIUopt")
    header.append("rr")
    
    header.append("L")
    header.append("AL")
    header.append("YGROSS")

    header.append("K")
    header.append("Sopt")
    header.append("I")
    header.append("YNET")

    header.append("CCATOT")
    header.append("DAMAGES")
    header.append("ABATECOST")
    header.append("MCABATE")
    header.append("C")
    header.append("PERIODU")
    header.append("TOTPERIODU")
    header.append("RFACTLONG")
    header.append("RLONG")
    header.append("RSHORT")
    header.append("ELAND")

    if 1 == 0:
        num_cols = output.shape[0]
        num_rows = len(header)

        row = ["INDEX"]
        for iCol in range(0, num_cols):
            row.append(iCol+1)
        writer.writerow(row)

        for iRow in range(1, num_rows):
            row = [header[iRow-1]]
            for iCol in range(0, num_cols):
                row.append(output[iCol, iRow-1])
            writer.writerow(row)
    else:
        num_rows = output.shape[0]
        num_cols = len(header)

        row = ['IPERIOD']
        for iCol in range(0, num_cols):
            row.append(header[iCol])
        writer.writerow(row)

        for iRow in range(1, num_rows):
            row = [iRow]
            for iCol in range(1, num_cols):
                row.append(output[iRow, iCol-1])
            writer.writerow(row)

    f.close()

###############################################################################   

print("Success")