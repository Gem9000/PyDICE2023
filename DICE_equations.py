'''
Authors: Jacob Wessel, George Moraites
'''

import numpy as np
import math
from DICE_params import modelParams

class modelEqns():

    def __init__(self):

        self.params = modelParams()
        num_times = self.params._num_times
        tstep = self.params._tstep

        #Create size arrays so we can index from 1 instead of 0
        self._L = np.zeros(num_times+1)
        self._al = np.zeros(num_times+1)
        self._sigma = np.zeros(num_times+1)
        self._sigmatot = np.zeros(num_times+1)
        self._gA = np.zeros(num_times+1)
        self._gsig = np.zeros(num_times+1)
        self._eland = np.zeros(num_times+1)
        self._emissrat = np.zeros(num_times+1)
        self._cost1tot = np.zeros(num_times+1)
        self._pbacktime = np.zeros(num_times+1)
        self._miuup = np.zeros(num_times+1)
        self._rr = np.zeros(num_times+1) 
        self._varpcc = np.zeros(num_times+1)
        self._rprecaut = np.zeros(num_times+1)
        self._RR1 = np.zeros(num_times+1)
        self._gL = np.zeros(num_times+1) ##
        self._gcost1 = np.zeros(num_times+1) ##
        self._scc = np.zeros(num_times+1) ##
        self._cpricebase = np.zeros(num_times+1) ##
        self._ppm = np.zeros(num_times+1) ##
        self._atfrac2020 = np.zeros(num_times+1) ##
        self._atfrac1765 = np.zeros(num_times+1) ##
        self._abaterat = np.zeros(num_times+1) ##
        self._FORC_CO2 = np.zeros(num_times+1) ##
        self._gbacktime = np.zeros(num_times+1) ##

        self._res0lom          = np.zeros(num_times+1)
        self._res1lom          = np.zeros(num_times+1)
        self._res2lom          = np.zeros(num_times+1)
        self._res3lom          = np.zeros(num_times+1)
        self._mat              = np.zeros(num_times+1)
        self._cacc             = np.zeros(num_times+1)
        self._force            = np.zeros(num_times+1)
        self._tbox1            = np.zeros(num_times+1)
        self._tbox2            = np.zeros(num_times+1)
        self._tatm             = np.zeros(num_times+1)
        self._irfeqlhs         = np.zeros(num_times+1)
        self._irfeqrhs         = np.zeros(num_times+1)
        self._alpha            = np.zeros(num_times+1)
        self._calculated_mmat  = np.zeros(num_times+1)

        # variables from nonco2 forcings include
        self._eco2             = np.zeros(num_times+1) #Total CO2 emissions (GtCO2 per year)
        self._eco2e            = np.zeros(num_times+1) #Total CO2e emissions including abateable nonCO2 GHG (GtCO2 per year)
        self._F_GHGabate       = np.zeros(num_times+1) #Forcings abateable nonCO2 GHG
        self._eind             = np.zeros(num_times+1) #Industrial CO2 emissions (GtCO2 per yr)
        self._eland            = np.zeros(num_times+1) #Emissions from deforestation (GtCO2 per year)

        self._CO2E_GHGabateB   = np.zeros(num_times+1) #Abateable non-CO2 GHG emissions base
        self._CO2E_GHGabateact = np.zeros(num_times+1) #Abateable non-CO2 GHG emissions base (actual)
        self._F_Misc           = np.zeros(num_times+1) #Non-abateable forcings (GHG and other)
        self._sigmatot         = np.zeros(num_times+1) #Emissions output ratio for CO2e
        
        self._K          = np.zeros(num_times+1) #Capital stock (trillions 2019 US dollars)
        self._C          = np.zeros(num_times+1) #Consumption (trillions 2019 US dollars per year)
        self._CPC        = np.zeros(num_times+1) #Per capita consumption (thousands 2019 USD per year)
        self._I          = np.zeros(num_times+1) #Investment (trillions 2019 USD per year)
        self._Y          = np.zeros(num_times+1) #Gross world product net of abatement and damages (trillions 2019 USD per year)
        self._ygross     = np.zeros(num_times+1) #Gross world product of abatement and damages (trillions 2019 USD per year)
        self._ynet       = np.zeros(num_times+1) #Output net of damages equation (trillions 2019 USD per year)
        self._damages    = np.zeros(num_times+1) #Damages (trillions 2019 USD per year)
        self._damfrac    = np.zeros(num_times+1) #Damages as fraction of gross output
        self._abatecost  = np.zeros(num_times+1) #Cost of emissions reductions  (trillions 2019 USD per year)
        self._mcabate    = np.zeros(num_times+1) #Marginal cost of abatement (2019$ per ton CO2)
        self._ccatot     = np.zeros(num_times+1) #Total carbon emissions (GtC)
        self._cprice     = np.zeros(num_times+1) #Carbon price (2019$ per ton of CO2)
        self._periodu    = np.zeros(num_times+1) #One period utility function
        self._totperiodu = np.zeros(num_times+1) #Period utility
        self._utility    = np.zeros(num_times+1) #Welfare function
        self._rfactlong  = np.zeros(num_times+1) #Long interest factor
        self._rshort     = np.zeros(num_times+1) #Short-run interest rate: Real interest rate with precautionary(per annum year on year)
        self._rlong      = np.zeros(num_times+1) #Long-run interest rate: Real interest rate from year 0 to T

        self._irfeqlhs   = np.zeros(num_times+1) # IRF100 left hand side (for calculating alpha)
        self._irfeqrhs   = np.zeros(num_times+1) # IRF100 right hand side (for calculating alpha)
        self.IRFt        = np.zeros(num_times+1) # IRF equation (constrain to be zero at every time point)

##### Initial Conditions #####

        self._K[1]          = self.params._k0
        self._L[1]          = self.params._pop1    #Population 
        self._gA[1]         = self.params._gA1     #Growth rate
        self._al[1]         = self.params._AL1     #Initial total factor productivity
        self._gsig[1]       = self.params._gsigma1 #Initial growth of sigma
        self._rr[1]         = 1.0
        self._miuup[1]      = self.params._miu1
        self._miuup[2]      = self.params._miu2
        self._sigma[1]      = (self.params._e1)/(self.params._q1*(1-self.params._miu1)) #Carbon intensity 2020 kgCO2-output 2020
        self._ccatot[1]     = self.params._CumEmiss0

        self._mat[1]        = self.params._mat0
        self._tatm[1]       = self.params._tatm0
        self._res0lom[1]    = self.params._res00
        self._res1lom[1]    = self.params._res10
        self._res2lom[1]    = self.params._res20
        self._res3lom[1]    = self.params._res30
        self._tbox1[1]      = self.params._tbox10
        self._tbox2[1]      = self.params._tbox20
        self._eland[1]      = self.params._eland0
        self._F_GHGabate[1] = self.params._F_GHGabate2020
        self._rfactlong[1]  = 1000000




    def simulateDynamics(self,x):
    
        # Set the optimization variables
        MIUopt = np.zeros(self.params._num_times+1)
        Sopt = np.zeros(self.params._num_times+1)
        for i in range(1, self.params._num_times+1):
            MIUopt[i] = x[i-1]                   #Optimal emissions control rate GHGs
            Sopt[i] = x[self.params._num_times + i-1]   #Gross savings rate as fraction of gross world product

        #Control logic for the emissions control rate (piecewise pyomo)
        for i in range(3, self.params._num_times+1):
            if self.params._t[i] > 2:
                self._miuup[i] = (self.params._delmiumax*(self.params._t[i]-1))
            if self.params._t[i] > 8:
                self._miuup[i] = (0.85 + 0.05 * (self.params._t[i]-8))
            if self.params._t[i] > 11:
                self._miuup[i] = self.params._limmiu2070
            if self.params._t[i] > 20:
                self._miuup[i] = self.params._limmiu2120
            if self.params._t[i] > 37:
                self._miuup[i] = self.params._limmiu2200
            if self.params._t[i] > 57:
                self._miuup[i] = self.params._limmiu2300

        for i in range(2, self.params._num_times+1):

            #Depends on the t-1 time period
            self._ygross[i] = self._al[i] * ((self._L[i]/self.params._MILLE)**(1.0-self.params._gama)) * self._K[i]**self.params._gama  #Gross world product GROSS of abatement and damages (trillions 20i9 USD per year)
            self._eland[i] = self.params._eland0 * (1 - self.params._deland) ** (self._t[i]-1)
            self._eco2[i] = (self._sigma[i] * self._ygross[i] + self._eland[i]) * (1-MIUopt) #New
            self._eind[i] = self._sigma[i] * self._ygross[i] * (1.0 - MIUopt[i])
            self._eco2e[i] = (self._sigma[i] * self._ygross[i] + self._eland[i] + self._CO2E_GHGabateB[i]) * (1-MIUopt) #New
            self._ccatot[i] = self._ccatot[i] + self._eco2[i]*(5/3.666)
            self._damfrac[i] = (self.params._a1 * self._tatm[i] + self.params._a2base * self._tatm[i] ** self.params._a3)  
            self._damages[i] = self._ygross[i] * self._damfrac[i]
            self._abatecost[i] = self._ygross[i] * self._cost1tot[i] * MIUopt[i]**self.params._expcost2
            self._mcabate[i] = self._pbacktime[i] * MIUopt[i]**(self.params._expcost2-1)
            self._cprice[i] = self._pbacktime[i] * MIUopt[i]**(self.params._expcost2-1)

            ########################Economic##############################
            self._ynet[i]         = self._ygross[i] * (1-self._damfrac[i])
            self._Y[i]            = self._ynet[i] - self._abatecost[i]
            self._C[i]            = self._Y[i] - self._I[i]
            self._CPC[i]          = self.params._MILLE * self._C[i]/self._L[i]
            self._I[i]            = Sopt[i] * self._Y[i]

            # this equation is a <= inequality and needs to be treated as such #
            self._K[i]            = (1.0 - self.params._dk)**self.params._tstep * self._K[i-1] + self.params._tstep * self._I[i]

            self._rfactlong[i]    = (self.params._SRF * (self._CPC[i]/self._CPC[i-1])**(-self.params._elasmu)*self._rr[i]) #Modified/New
            self._rlong[i]        = -math.log(self._rfactlong[i]/self.params._SRF)/(5*(i-1)) #NEW
            self._rshort[i]       = (-math.log(self._rfactlong[i]/self._rfactlong[i-1])/5) #NEW 

            #########################Welfare Functions###################
            self._periodu[i]      = ((self._C[i]*self.params._MILLE/self._L[i])**(1.0-self.params._elasmu)-1.0) / (1.0 - self.params._elasmu) - 1.0
            self._totperiodu[i]   = self._periodu[i] * self._L[i] * self._rr[i]

            self._varpcc[i]       = min(self.params._siggc1**2*5*(self.params._t[i]-1), self.params._siggc1**2*5*47) #Variance of per capita consumption
            self._rprecaut[i]     = -0.5 * self._varpcc[i-1]* self.params._elasmu**2 #Precautionary rate of return
            self._RR1[i]          = 1/((1+math.exp(self.params._prstp + self.params._betaclim * self.params._pi)-1)**(-self.params._tstep*(self.params._t[i]-1))) #STP factor without precautionary factor
            self._rr[i]           = self._RR1[i]*(1+ self._rprecaut[i]**(-self.params._tstep*(self.params._t[i]-1)))  #STP factor with precautionary factor
            self._L[i]            = self._L[i-1]*(self.params._popasym / self._L[i-1])**self.params._popadj # Population adjustment over time
            self._gA[i]           = self.params._gA1 * np.exp(-self.params._delA * 5.0 * (self.params._t[i] - 1)) # Growth rate of productivity
            self._al[i]           = self._al[i-1] /((1-self._gA[i-1])) # Level of total factor productivity
            self._cpricebase[i]   = self.params._cprice1*(1+self.params._gcprice)**(5*(self.params._t[i]-1)) #Carbon price in base case of model
            self._pbacktime[i]    = self.params._pback2050 * math.exp(-5*(0.01 if self.params._t[i] <= 7 else 0.001)*(self._t[i]-7)) #Backstop price 2019$ per ton CO2. Incorporates 2023 condition
            self._gsig[i]         = min(self.params._gsigma1*self.params._delgsig **((self.params._t[i]-1)), self.params._asymgsig) #Change in rate of sigma (rate of decarbonization)
            self._sigma[i]        = self._sigma[i-1]*math.exp(5*self._gsig[i-1])
            if self._t[i] <= 16:
                self._emissrat[i] = self.params._emissrat2020 +((self.params._emissrat2100-self.params._emissrat2020)/16)*(self._t[i]-1)
            else:
                self._emissrat[i] = self.params._emissrat2100
            self._sigmatot[i]     = self._sigma[i]*self._emissrat[i]
            self._cost1tot[i]     = self._pbacktime[i]*self._sigmatot[i]/self.params._expcost2/1000

            self._irfeqlhs[i]   =  ((self._alpha[i] * self.params._emshare0 * self.params._tau0 * (1 - math.exp(-100 / (self._alpha[i] * self.params._tau0)))) +
                                    (self._alpha[i] * self.params._emshare1 * self.params._tau1 * (1 - math.exp(-100 / (self._alpha[i] * self.params._tau1)))) +
                                    (self._alpha[i] * self.params._emshare2 * self.params._tau2 * (1 - math.exp(-100 / (self._alpha[i] * self.params._tau2)))) +
                                    (self._alpha[i] * self.params._emshare3 * self.params._tau3 * (1 - math.exp(-100 / (self._alpha[i] * self.params._tau3)))))
            self._irfeqrhs[i]   =  self.params._irf0 + self.params._irC * self._cacc[i] + self.params._irT * self._tatm[i]
            self.IRFt[i] = self._irfeqlhs[i] - self._irfeqrhs[i]

            self._res0lom[i] = (self.params._emshare0 * self.params._tau0 * self._alpha[i] * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self.params._tstep / (self.params._tau0 * self._alpha[i]))) + 
                                    self._res0lom[i-1] * math.exp(-self.params._tstep / (self.params._tau0 * self._alpha[i])))

            self._res1lom[i] = (self.params._emshare1 * self.params._tau1 * self._alpha[i] * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self.params._tstep / (self.params._tau1 * self._alpha[i]))) + 
                                    self._res1lom[i-1] * math.exp(-self.params._tstep / (self.params._tau1 * self._alpha[i])))

            self._res2lom[i] = (self.params._emshare2 * self.params._tau2 * self._alpha[i] * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self.params._tstep / (self.params._tau2 * self._alpha[i]))) + 
                                    self._res2lom[i-1] * math.exp(-self.params._tstep / (self.params._tau2 * self._alpha[i])))

            self._res3lom[i] = (self.params._emshare3 * self.params._tau3 * self._alpha[i] * 
                                    (self._eco2[i] / 3.667) * 
                                    (1 - math.exp(-self.params._tstep / (self.params._tau3 * self._alpha[i]))) + 
                                    self._res3lom[i-1] * math.exp(-self.params._tstep / (self.params._tau3 * self._alpha[i])))

            self._calculated_mmat[i] = self.params._mateq + self._res0lom[i] + self._res1lom[i] + self._res2lom[i] + self._res3lom[i]
            if self._calculated_mmat[i] < 10:
                self._mat[i] = 10
            else:
                self._mat[i] = self._calculated_mmat[i]

            if self.params._t[i-1] <= 16:
                self._CO2E_GHGabateB[i-1] = self.params._ECO2eGHGB2020 + ((self.params._ECO2eGHGB2100-self.params._ECO2eGHGB2020)/16)*(self.params._t[i-1]-1)
                self._F_Misc[i-1]=self.params._F_Misc2020 + ((self.params._F_Misc2100-self.params._F_Misc2020)/16)*(self.params._t[i-1]-1)
            else:
                self._CO2E_GHGabateB[i-1] = self.params._ECO2eGHGB2100
                self._F_Misc[i-1]=self.params._F_Misc2100
            
            self._F_GHGabate[i] = self.params._F_GHGabate2020*self._F_GHGabate[i-1] + self.params._F_GHGabate2100*self._CO2E_GHGabateB[i-1]*(1-MIUopt[i-1])

            self._force[i] = (self.params._fco22x * ((math.log((self._mat[i-1]+1e-9/self.params._mateq))/math.log(2)) 
                                + self._F_Misc[i-1] + self._F_GHGabate[i-1])) #Good

            self._tbox1[i] = (self._tbox1[i-1] *
                                    math.exp(self.params._tstep/self.params._d1) + self.params._teq1 *
                                    self._force[i] * (1-math.exp(self.params._tstep/self.params._d1))) #Good

            self._tbox2[i] = (self._tbox2[i-1] *
                                    math.exp(self.params._tstep/self.params._d2) + self.params._teq2 *
                                    self._force[i] * (1-math.exp(self.params._tstep/self.params._d2))) #Good

            self._tatm[i] = np.clip(self._tbox1[i-1] + self._tbox2[i-1], 0.5, 20) #Good

            self._cacc[i] = (self._ccatot[i-1] - (self._mat[i-1] - self.params._mateq)) #Good



            # ### Post-Solution Parameter-Assignment ###
            # self._scc[i]        = -1000 * self._eco2[i] / (.00001 + self._C[i]) # NOTE: THESE (self._eco2[i] and self._C[i]) NEED TO BE MARGINAL VALUES, NOT THE SOLUTIONS THEMSELVES
            # self._ppm[i]        = self._mat[i] / 2.13
            # self._abaterat[i]   = self._abatecost[i] / self._Y[i]
            # self._atfrac2020[i] = (self._mat[i] - self.params._mat0) / (self._ccatot[i] + .00001 - self.params._CumEmiss0)
            # self._atfrac1765[i] = (self._mat[i] - self.params._mateq) / (.00001 + self._ccatot[i])
            # self._FORC_CO2[i]   = self.params._fco22x * ((math.log((self._mat[i] / self.params._mateq)) / math.log(2)))

    def runModel(self):
        pass    

print("Success")

dice_eqns = modelEqns()

dice_eqns.runModel()

