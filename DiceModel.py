'''
Created 02/23/2024 
Author: George Moraites
Adapted From: domokane DICEModel.py
'''

import numpy as np
import math
import csv

from FAIRModel import FAIRParams

'''
Adding in new imports for the DICE Model Portion
'''
from numba import njit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
seaborn.set_theme(style='ticks')

###############################################################################

@njit(cache=True, fastmath=True)
def objFn(x, *args):
    """ This is the pass-through function that returns a single float value of
    the objective function for the benefit of the optimisation algorithm. """

    out = simulateDynamics(x, *args)
    return out[0, 0]

###############################################################################

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


        ##Legacy constants for the emission parameter. They were't present in 2023
        ##But were referenced in one of the equations so basing this off 2016 version##
        self._eland0 = 2.6 # Carbon emissions from land 2015 (GtC02 per year)
        self._deland = 0.115 # Decline rate of land emissions (per period)

        ##Emissions Limits########################

        self._miu2 = 0.10 #Second emission limit 

        ###########################################

        ##################################################################
        #Functions
        ##################################################################

        #Num times should be 81
        #time increment should be 5 years
        self._num_times = num_times
        self._t = np.arange(0,num_times+1)

        #Create size arrays so we can index from 1 instead of 0
        self._rartp = np.zeros(num_times+1)
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
        self._rr = np.zerors(num_times+1) 
        self._varpcc = np.zeros(num_times+1)
        self._rprecaut = np.zeros(num_times+1)
        self._RR1 = np.zeros(num_times+1)
        self._etree  = np.zeros(num_times+1)

        '''PARAMETERS
        L(t)           Level of population and labor
        aL(t)          Level of total factor productivity
        sigma(t)       CO2-emissions output ratio
        sigmatot(t)    GHG-output ratio
        gA(t)          Growth rate of productivity from
        gL(t)          Growth rate of labor and population
        gcost1         Growth of cost factor
        gsig(t)        Change in sigma (rate of decarbonization)
        eland(t)       Emissions from deforestation (GtCO2 per year)
        cost1tot(T)    Abatement cost adjusted for backstop and sigma
        pbacktime(t)   Backstop price 2019$ per ton CO2
        optlrsav       Optimal long-run savings rate used for transversality
        scc(t)         Social cost of carbon
        cpricebase(t)  Carbon price in base case
        ppm(t)         Atmospheric concentrations parts per million
        atfrac2020(t)  Atmospheric share since 2020
        atfrac1765(t)  Atmospheric fraction of emissions since 1765
        abaterat(t)    Abatement cost per net output
        miuup(t)       Upper bound on miu
        gbacktime(t)   Decline rate of backstop price
        '''

        ''' Precautionary dynamic parameters
        varpcc(t)         Variance of per capita consumption 
        rprecaut(t)       Precautionary rate of return
        RR(t)             STP with precautionary factor
        RR1(t)            STP factor without precautionary factor;
        '''

        #Set relevant values using the paramteters above
        self._l[1] = self._pop1 #Population 
        self._gA[1] = self._gA1 #Growth rate
        self._al[1] = self._AL1 #Initial total factor productivity
        self._gsig[1] = self._gsigma1 #Initial growth of sigma
        self._rr[1] = 1.0
        self._miuup[1] = self._miu1
        self._miuup[2] = self._miu2
        self._etree[1] = self._eland0
        #varpcc(t)       =  min(Siggc1**2*5*(t.val-1),Siggc1**2*5*47);

        self._rartp = math.exp(self._prstp + self._betaclim * self._pi)-1 #Risk adjusted rate of time preference 
        
        #NEED TO ASK ABOUT THIS ONE
        self._sig1 = (self._e1)/(self._q1*(1-self._miu1))
        self._sigma[1] = self._sig1 #sig1 did not have a value in the gms code

        for i in range(2, self._num_times+1):
            self._varpcc[i] = min(self._siggc1**2*5*(self._t[i]-1), self._siggc1**2*5*47) #Variance of per capita consumption
            self._rprecaut[i] = -0.5 * self._varpcc[i-1]* self._elasmu**2 #Precautionary rate of return
            self._RR1[i] = 1/((1+self._rartp)**(-self._tstep*(self._t[i]-1))) #STP factor without precautionary factor
            self._rr[i] =   self._RR1[i-1]*(1+ self._rprecaut[i-1]**(self._tstep*(self._t[i] -1)))          #STP factor with precautionary factor 
            self._l[i] = self._l[i-1]*(self._popasym / self._l[i-1])**self._popadj # Population adjustment over time 
            self._gA[i] = self._gA1 * np.exp(-self._delA * 5.0 * (self._t[i] - 1)) #NEED TO FIGURE OUT WHAT THIS IS 
            self._al[i] = self._al[i-1] /((1-self._gA[i-1])) #Degradation of productivity??
            self._cpricebase[i] = self._cprice1*(1+self._gcprice)**(5*(self._t[i]-1)) #Carbon price in base case of model
            self._pbacktime[i] = self._pback2050 * math.exp(-5*(0.01 if self.t[i] <= 7 else 0.001)*(self._t[i]-7)) #Backstop price 2019$ per ton CO2. Incorporates the condition found in the 2023 version
            self._gsig[i] = min(self._gsigma1*self._delgsig **((self._t[i]-1)), self._asymgsig) #Change in rate of sigma (represents rate of decarbonization)
            self._sigma[i] = self._sigma[i-1]*math.exp(5*self._gsig[i-1])

            self._etree[i] = self._eland0 * (1.0 - self._deland) ** (self._t[i]-1) #Not explicitly defined in the 2023 version, but needs to be included

        #Control logic for the emissions control rate
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
             
        if 1==1:

            f = open("./results/parameters.csv" , mode = "w", newline='')
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
            header = []
            header.append("PERIOD")
            header.append("VARPCC")
            header.append("RPRECAUT")
            header.append("RR1")
            header.append("RR")
            header.append("L")
            header.append("GA")
            header.append("AL")
            header.append("CPRICEBASE")
            header.append("PBACKTIME")
            header.append("GSIG")
            header.append("SIGMA")
            writer.writerow(header)
            
            num_rows = self._num_times + 1
        
            for i in range(0, num_rows):
                row = []
                row.append(i)
                row.append(self._varpcc[i])
                row.append(self._rprecaut[i])
                row.append(self._RR1[i])
                row.append(self._rr[i])
                row.append(self._l[i])
                row.append(self._gA[i])
                row.append(self._al[i])
                row.append(self._cpricebase[i])
                row.append(self._pbacktime[i])
                row.append(self._gsig[i])
                row.append(self._sigma[i])

                writer.writerow(row)

            f.close()

        elif 1==2:
            print("Variance of per capita consumption:", self._varpcc)
            print("Precationary rate of return:",self._rprecaut)
            print("STP factor without precationary factor:",self._RR1)
            print("STP factor with precationary factor:",self._rr)
            print("Labour:", self._l) # CHECKED OK
            print("Growth rate of productivity", self._gA) # CHECKED OK
            print("Productivity", self._al) # CHECKED OK
            print("Carbon price based case", self._cpricebase) # AGREES WITH NORDHAUS UNTIL 2235 ??
            print("Backstop price", self._pbacktime) # CHECKED OK
            print("Change in sigma", self._gsig) # CHECKED OK
            print("CO2 output ratio:", self._sigma) # CHECKED OK
            print("Long run savings rate", self._optlrsav) # CHECKED OK

        else:
            print("SOME CHECKING TO BE DONE")



   #    @njit(cache=True, fastmath=True)
    def simulateDynamics(self, SRF, x, sign, outputType, num_times,
                         tstep, al, ll, sigma, forcoth,
                         cost1, etree,
                         scale1, scale2,
                         ml0, mu0, mat0, cca0,
                         a1, a2, a3,
                         c1, c3, c4,
                         b11, b12, b21, b22, b32, b23, b33,
                         fco22x, t2xco2, rr, gama,
                         tocean0, tatm0, elasmu, prstp, expcost2,
                         k0, dk, pbacktime, CumEmiss0):
        """ This is the simulation of the DICE 2023 model dynamics. It is optimised
        for speed. For this reason I have avoided the use of classes. """
    
        LOG2 = np.log(2)
        L = ll  # NORDHAUS RENAMES IT TO UPPER CASE IN EQUATIONS
        MILLE = 1000.0  ######NEED TO LOOK AT THIS##############
    
        # We take care to ensure that the indexing starts at 1 to allow comparison
        # with matlab
        MIUopt = np.zeros(num_times+1)
        Sopt = np.zeros(num_times+1)
    
###############################################################################
# Set the optimisation variables
###############################################################################

        for i in range(1, num_times+1):
            MIUopt[i] = x[i-1]          #Optimal emissions control rate GHGS
            Sopt[i] = x[num_times + i-1]   #Gross savings rate as fraction of gross world product
    
###########################################################################
#Variabls and nonnegative variables equations
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
        RSHORT    = np.zeros(num_times+1)      #Real interest rate with precautionary(per annum year on year)
        RLONG   = np.zeros(num_times+1)        #Real interest rate from year 0 to T
        EIND = np.zeros(num_times+1)           #Industrial Emissions (GtCO2 per year)
        
        #New
        ECO2 = np.zeros(num_times+1)           
        ECO2E = np.zeros(num_times+1)
        CO2E_GHGabateB = np.zeros(num_times+1)
        F_GHGabate = np.zeros(num_times+1)
        #Emissions from deforestation



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

#Fixed initial values(depricated in the 2023 version included for now)
    #ML[1] = ml0
    
    

###################################Initilizing Equations#################################
        CCATOTEQ[1] = CumEmiss0
        ##F_GHGabate[1] F_GHGabate2020     #Need to find this value
        
        YGROSS[1] = al[1] * ((L[1]/MILLE)**(1.0-gama)) * K[1]**gama  #Gross world product GROSS of abatement and damages (trillions 2019 USD per year)

        ECO2[1] = (sigma[1] * YGROSS[1] + etree[1]) * (1-MIUopt) #New
        EIND[1] = sigma[1] * YGROSS[1] * (1.0 - MIUopt[1])
        
        ECO2E[1] = (sigma[1] * YGROSS[1] + etree[1] + CO2E_GHGabateB[1]) * (1-MIUopt) #New
        
        CCATOT[1] = CCATOT[1] + ECO2[1]*(5/3.666)
        DAMFRAC = (a1 * TATM[1] + a2 * TATM[1] ** a3)  
        DAMAGES = YGROSS[1] * DAMFRAC[1]
        
        ABATECOST = YGROSS[1] * cost1[1] * (MIUopt[1] ** expcost2) #NEEDS TO BE CHECKED
        MCABATE = pbacktime[1] * MIUopt[1] ** (expcost2-1)
        CPRICE = pbacktime[1] * (MIUopt[1]) **(expcost2-1)

        #FORC[1] = fco22x * np.log(MAT[1]/588.000)/LOG2 + forcoth[1]
        DAMFRAC[1] = a1*TATM[1] + a2*TATM[1]**a3
        DAMAGES[1] = YGROSS[1] * DAMFRAC[1]
        ABATECOST[1] = YGROSS[1] * cost1[1] * MIUopt[1]**expcost2
        MCABATE[1] = pbacktime[1] * MIUopt[1]**(expcost2-1)
        CPRICE[1] = pbacktime[1] * (MIUopt[1])**(expcost2-1)

        CACC[1] = cca0  # DOES NOT START TILL PERIOD 2
        t = 1           #Needed to define this just for one of the
                        #functions that needed it

        ########################Economic Initilizations##############
        YNET[1] = YGROSS[1] * (1-DAMFRAC[1])
        Y[1] = YNET[1] - ABATECOST[1]
        C[1] = Y[1] - I[1]
        CPC[1] = MILLE * C[1]/L[1]
        I[1] = Sopt[1] * Y[1]
        K[1] = k0

        RFACTLONG[1] = 1000000
        RLONG[1] = (-math.log(RFACTLONG[1]/SRF)/(5*t)) #NEW
        RSHORT[1] = (-math.log(RFACTLONG[1]/RFACTLONG[0])/(5)) #NEW 

        #########################Welfare Functions###################
        PERIODU[1] = ((C[1]*MILLE/L[1])**(1.0-elasmu)-1.0) / (1.0 - elasmu) - 1.0
        TOTPERIODU[1] = PERIODU[1] * L[1] * rr[1]


        #Many of the equations in the module have been put into
        #The separate DFAIR class. The equations that the DFAIR modules rely
        #On should be calculated and then back checked 
        for i in range(2, num_times+1):

            #Depends on the t-1 time period
            CCATOT[i] = CCATOT[i-1] + ECO2[i-1] * (5/3.666)
            YGROSS[i] = al[i] * ((L[i]/MILLE)**(1.0-gama)) * K[i]**gama  #Gross world product GROSS of abatement and damages (trillions 20i9 USD per year)

            ECO2[i] = (sigma[i] * YGROSS[i] + etree[i]) * (1-MIUopt) #New
            EIND[i] = sigma[i] * YGROSS[i] * (1.0 - MIUopt[i])
            
            ECO2E[i] = (sigma[i] * YGROSS[i] + etree[i] + CO2E_GHGabateB[i]) * (1-MIUopt) #New
            
            CCATOT[i] = CCATOT[i] + ECO2[i]*(5/3.666)
            DAMFRAC = (a1 * TATM[i] + a2 * TATM[i] ** a3)  
            DAMAGES = YGROSS[i] * DAMFRAC[i]
            
            ABATECOST = YGROSS[i] * cost1[i] * (MIUopt[i] ** expcost2) #NEEDS TO BE CHECKED
            MCABATE = pbacktime[i] * MIUopt[i] ** (expcost2-1)
            CPRICE = pbacktime[i] * (MIUopt[i]) **(expcost2-1)

            DAMFRAC[i] = a1*TATM[i] + a2*TATM[i]**a3
            DAMAGES[i] = YGROSS[i] * DAMFRAC[i]
            ABATECOST[i] = YGROSS[i] * cost1[i] * MIUopt[i]**expcost2
            MCABATE[i] = pbacktime[i] * MIUopt[i]**(expcost2-i)
            CPRICE[i] = pbacktime[i] * (MIUopt[i])**(expcost2-i)

            ########################Economic##############################
            YNET[i] = YGROSS[i] * (i-DAMFRAC[i])
            Y[i] = YNET[i] - ABATECOST[i]
            C[i] = Y[i] - I[i]
            CPC[i] = MILLE * C[i]/L[i]
            I[i] = Sopt[i] * Y[i]
            K[i] = (1.0 -dk)**tstep * K[i-1] + tstep * I[i]

            RFACTLONG[i] = (SRF * (CPC[i-1]/CPC[i])**(-elasmu)*rr[i-1]) #Modified/New
            RLONG[i] = (-math.log(RFACTLONG[i-1]/SRF)/(5*t)) #NEW
            RSHORT[i] = (-math.log(RFACTLONG[i-1]/RFACTLONG[i])/(5)) #NEW 

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

            #Implemented in the originial DICE 2016 implementation
            #However, might be depricated. Needs to be checked
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
                output[jTime,col] = etree[iTime]
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
    header.append("ETREE")

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

def runModel(self):
        pass


print("Success")

'''

###############################################################################


def plotFigure(x, y, xlabel, ylabel, title):

    x = x[:-1]
    y = y[:-1]

#    mpl.style.use('ggplot')
    fig = plt.figure(figsize=(8, 6), dpi=72, facecolor="white")
#    axes = plt.subplot(111)
    plt.plot(x, y)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=14)
    seaborn.despine()
    fig.tight_layout()
    return fig

###############################################################################


def plotStateToFile(fileName, years, output, x):

    num_times = output.shape[0]
    output = np.transpose(output)

    pp = PdfPages(fileName)

    TATM = output[3]
    title = 'Change in Atmosphere Temperature (TATM)'
    xlabel = 'Years'
    ylabel = 'Degrees C from 1900'
    fig = plotFigure(years, TATM, xlabel, ylabel, title)
    pp.savefig(fig)

    TOCEAN = output[23]
    xlabel = 'Years'
    title = 'Change in Ocean Temperature (TOCEAN)'
    ylabel = 'Degrees C from 1900'
    fig = plotFigure(years, TOCEAN, xlabel, ylabel, title)
    pp.savefig(fig)

    MU = output[21]
    xlabel = 'Years'
    title = 'Change in Carbon Concentration Increase in Upper Oceans (MU)'
    ylabel = 'GtC from 1750'
    fig = plotFigure(years, MU, xlabel, ylabel, title)
    pp.savefig(fig)

    ML = output[20]
    xlabel = 'Years'
    title = 'Change in Carbon Concentration in Lower Oceans (ML)'
    ylabel = 'GtC from 1750'
    fig = plotFigure(years, ML, xlabel, ylabel, title)
    pp.savefig(fig)

    DAMAGES = output[24]
    xlabel = 'Years'
    title = 'Damages (DAMAGES)'
    ylabel = 'USD (2010) Trillions per Year'
    fig = plotFigure(years, DAMAGES, xlabel, ylabel, title)
    pp.savefig(fig)

    DAMFRAC = output[5]
    xlabel = 'Years'
    title = 'Damages as a Fraction of Gross Output (DAMFRAC)'
    ylabel = 'Damages Output Ratio'
    fig = plotFigure(years, DAMFRAC, xlabel, ylabel, title)
    pp.savefig(fig)

    ABATECOST = output[25]
    xlabel = 'Years'
    title = 'Cost of Emissions Reductions (ABATECOST)'
    ylabel = 'USD (2010) Trillions per Year'
    fig = plotFigure(years, ABATECOST, xlabel, ylabel, title)
    pp.savefig(fig)

    MCABATE = output[26]
    xlabel = 'Years'
    title = 'Marginal abatement cost(MCABATE)'
    ylabel = '2010 USD per Ton of CO2'
    fig = plotFigure(years, MCABATE, xlabel, ylabel, title)
    pp.savefig(fig)

    E = output[1]
    xlabel = 'Years'
    title = 'Total CO2 emission (E)'
    ylabel = 'GtCO2 per year'
    fig = plotFigure(years, E, xlabel, ylabel, title)
    pp.savefig(fig)

    EIND = output[0]
    xlabel = 'Years'
    title = 'Total Industrial CO2 emissions (EIND)'
    ylabel = 'GtCO2 per year'
    fig = plotFigure(years, EIND, xlabel, ylabel, title)
    pp.savefig(fig)

    MAT = output[30]
    xlabel = 'Years'
    title = 'Change in Carbon Concentration in Atmosphere (MAT)'
    ylabel = 'GtC from 1750'
    fig = plotFigure(years, MAT, xlabel, ylabel, title)
    pp.savefig(fig)

    FORC = output[22]
    xlabel = 'Years'
    title = 'Increase in Radiative Forcing (FORC)'
    ylabel = 'Watts per M2 from 1900'
    fig = plotFigure(years, FORC, xlabel, ylabel, title)
    pp.savefig(fig)

    RI = output[9]
    xlabel = 'Years'
    title = 'Real Interest Rate (RI)'
    ylabel = 'Rate per annum'
    fig = plotFigure(years, RI, xlabel, ylabel, title)
    pp.savefig(fig)

    C = output[27]
    xlabel = 'Years'
    title = 'Consumption (C)'
    ylabel = 'USD (2010) Trillion per Year'
    fig = plotFigure(years, C, xlabel, ylabel, title)
    pp.savefig(fig)

    Y = output[4]
    xlabel = 'Years'
    title = 'Gross Product Net of Abatement and Damages (Y)'
    ylabel = 'USD (2010) Trillion per Year'
    fig = plotFigure(years, Y, xlabel, ylabel, title)
    pp.savefig(fig)

    YGROSS = output[13]
    xlabel = 'Years'
    title = 'World Gross Product (YGROSS)'
    ylabel = 'USD (2010) Trillion per Year'
    fig = plotFigure(years, YGROSS, xlabel, ylabel, title)
    pp.savefig(fig)

    II = output[16]
    xlabel = 'Years'
    title = 'Investment (I)'
    ylabel = 'USD (2010) Trillion per Year'
    fig = plotFigure(years, II, xlabel, ylabel, title)
    pp.savefig(fig)

    num_times = len(II)

    S = x[num_times:(2*num_times)]
    xlabel = 'Years'
    title = 'Optimised: Saving Rates (S)'
    ylabel = 'Rate'
    fig = plotFigure(years, S, xlabel, ylabel, title)
    pp.savefig(fig)

    MIU = x[0:num_times]
    title = 'Optimised: Carbon Emission Control Rate (MIU)'
    xlabel = 'Years'
    ylabel = 'Rate'
    fig = plotFigure(years, MIU, xlabel, ylabel, title)
    pp.savefig(fig)

    pp.close()

#################################################################
'''
