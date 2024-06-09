'''
Authors: Jacob Wessel, George Moraites
'''

import numpy as np
import pandas as pd
import os

class getParams():
    '''
    writes a .dat file with the second column
    of parameters in the specified input file
    (assuming first column is param names)
    '''
    
    def __init__(self, input_file = 'param_inputs.csv', scenario_name = 'unnamedscenario'):
        
        self._csv_data = pd.read_csv(input_file, header=None,index_col=0)
        self._datapath = 'temp_data/'
        os.makedirs(self._datapath, exist_ok=True)
        self._outfile = self._datapath + scenario_name + '.dat'
        self._numtimes = int(self._csv_data.loc['numtimes',1])
        
        # run simulations for time-varying parameters (move to main model as Vars if endogeneity added)
        
        self._L = [np.nan for i in range(self._numtimes+2)]
        self._aL = [np.nan for i in range(self._numtimes+2)]
        self._sigma = [np.nan for i in range(self._numtimes+2)]
        self._sigmatot = [np.nan for i in range(self._numtimes+2)]
        self._gA = [np.nan for i in range(self._numtimes+2)]
        #self._gL = [np.nan for i in range(self._numtimes+2)]
        self._gsig = [np.nan for i in range(self._numtimes+2)]
        self._eland = [np.nan for i in range(self._numtimes+2)]
        self._cost1tot = [np.nan for i in range(self._numtimes+2)]
        self._pbacktime = [np.nan for i in range(self._numtimes+2)]
        self._cpricebase = [np.nan for i in range(self._numtimes+2)]
        #self._gbacktime = [np.nan for i in range(self._numtimes+2)]
        self._varpcc = [np.nan for i in range(self._numtimes+2)]
        self._rprecaut = [np.nan for i in range(self._numtimes+2)]
        self._rr = [np.nan for i in range(self._numtimes+2)]
        self._rr1 = [np.nan for i in range(self._numtimes+2)]
        self._CO2E_GHGabateB = [np.nan for i in range(self._numtimes+2)]
        #self._CO2E_GHGabateact = [np.nan for i in range(self._numtimes+2)]
        self._F_Misc = [np.nan for i in range(self._numtimes+2)]
        self._emissrat = [np.nan for i in range(self._numtimes+2)]
        #self._FORC_CO2 = [np.nan for i in range(self._numtimes+2)]
        self._tvp_list = [self._L, self._aL, self._sigma, self._sigmatot, self._gA, self._gsig, self._eland, self._cost1tot, self._pbacktime,
                          self._cpricebase, self._varpcc, self._rprecaut, self._rr, self._rr1, self._CO2E_GHGabateB, self._F_Misc, self._emissrat]
        self._names = ['L','aL','sigma','sigmatot','gA','gsig','eland','cost1tot','pbacktime','cpricebase',
                       'varpcc','rprecaut','rr','rr1','CO2E_GHGabateB','F_Misc','emissrat']
        

        for t in range(1,self._numtimes+2):
            
             #varpccEQ(m,t): # Variance of per capita consumption
            self._varpcc[t] = min(float(self._csv_data.loc['siggc1',1])**2*5*(t-1),
                                  float(self._csv_data.loc['siggc1',1])**2*5*47)
            
             #rprecautEQ(m,t): # Precautionary rate of return
            self._rprecaut[t] = (-0.5 * self._varpcc[t] * float(self._csv_data.loc['elasmu',1])**2)
            
             #rr1EQ(m,t): # STP factor without precautionary factor
            rartp = np.exp(float(self._csv_data.loc['prstp',1]) + \
                           float(self._csv_data.loc['betaclim',1]) * float(self._csv_data.loc['pi',1]))-1
            self._rr1[t] = 1/((1 + rartp)**(int(self._csv_data.loc['tstep',1]) * (t-1)))
            
             #rrEQ(m,t): # STP factor with precautionary factor
            self._rr[t] = self._rr1[t] * (1 + self._rprecaut[t])**(-1*int(self._csv_data.loc['tstep',1]) * (t-1))
            
             #LEQ(m,t): # Population adjustment over time (note initial condition)
            self._L[t] = self._L[t-1]*(float(self._csv_data.loc['popasym',1]) / \
                         self._L[t-1])**float(self._csv_data.loc['popadj',1]) if t > 1 else float(self._csv_data.loc['pop1',1])
            
             #gAEQ(m,t): # Growth rate of productivity (note initial condition)
            self._gA[t] = float(self._csv_data.loc['gA1',1]) * \
                          np.exp(-1*float(self._csv_data.loc['delA',1]) * \
                          int(self._csv_data.loc['tstep',1]) * (t-1)) if t > 1 else float(self._csv_data.loc['gA1',1])
            
             #aLEQ(m,t): # Level of total factor productivity (note initial condition)
            self._aL[t] = self._aL[t-1] /((1-self._gA[t-1])) if t > 1 else float(self._csv_data.loc['AL1',1])

             #cpricebaseEQ(m,t): # Carbon price in base case of model
            self._cpricebase[t] = float(self._csv_data.loc['cprice1',1]) * \
                                  ((1 + float(self._csv_data.loc['gcprice',1]))**(int(self._csv_data.loc['tstep',1]) * (t-1)))
            
             #pbacktimeEQ(m,t): # Backstop price 2019$ per ton CO2
            j = 0.01 if t <= 7 else 0.001
            self._pbacktime[t] = float(self._csv_data.loc['pback2050',1]) * np.exp(-1*int(self._csv_data.loc['tstep',1]) * j * (t-7))
            
             #gsigEQ(m,t): # Change in rate of sigma (rate of decarbonization)
            self._gsig[t] = min(float(self._csv_data.loc['gsigma1',1]) * (float(self._csv_data.loc['delgsig',1])**(t-1)),
                                float(self._csv_data.loc['asymgsig',1]))
            
             #sigmaEQ(m,t): # CO2 emissions output ratio (note initial condition)
            self._sigma[t] = self._sigma[t-1]*np.exp(int(self._csv_data.loc['tstep',1]) * self._gsig[t-1]) if t > 1 else \
                             float(self._csv_data.loc['e1',1])/(float(self._csv_data.loc['q1',1])*(1-float(self._csv_data.loc['miu1',1])))
            
             # elandEQ(m,t):
            self._eland[t] =  float(self._csv_data.loc['eland0',1]) * (1 -  float(self._csv_data.loc['deland',1]))**(t-1)
             
             # CO2E_GHGabateBEQ(m,t):
            self._CO2E_GHGabateB[t] = float(self._csv_data.loc['ECO2eGHGB2020',1]) + \
                                    ((float(self._csv_data.loc['ECO2eGHGB2100',1]) - float(self._csv_data.loc['ECO2eGHGB2020',1])) / 16) * (t-1) \
                                    if t <= 16 else float(self._csv_data.loc['ECO2eGHGB2100',1])
             
             # F_MiscEQ(m,t):
            self._F_Misc[t] = float(self._csv_data.loc['F_Misc2020',1]) + \
                            ((float(self._csv_data.loc['F_Misc2100',1]) - float(self._csv_data.loc['F_Misc2020',1])) / 16) * (t-1) \
                            if t <= 16 else float(self._csv_data.loc['F_Misc2100',1])
             
             # emissratEQ(m,t):
            self._emissrat[t] = float(self._csv_data.loc['emissrat2020',1]) + \
                              ((float(self._csv_data.loc['emissrat2100',1]) - float(self._csv_data.loc['emissrat2020',1])) / 16) * (t-1) \
                              if t <= 16 else float(self._csv_data.loc['emissrat2100',1])
             
             # sigmatotEQ(m,t):
            self._sigmatot[t] = self._sigma[t] * self._emissrat[t]
             
             # cost1totEQ(m,t):
            self._cost1tot[t] = self._pbacktime[t] * self._sigmatot[t] / float(self._csv_data.loc['expcost2',1]) / 1000
            
            with open(self._outfile, 'w') as f:
                # create parameter matrix for global params
                for c in self._csv_data.iloc[1:,:].iterrows():
                    f.write('param {} := {};\n'.format(c[0],c[1].values[0]))                
                
                f.write('\n\nparam:' + '\t')
                
                # write time-varying params to file
                for j in range(len(self._names)):
                    f.write(self._names[j] + '\t')
                f.write(':=\n\n')
                for i in range(self._numtimes):
                    f.write(str(i+1) + '\t')
                    for j in range(len(self._names)):
                        f.write(str(self._tvp_list[j][i+1]) + '\t')
                    f.write('\n')
                f.write(';')
                    
                f.close()
        
        return None
    
    def removeTempData(self):
        os.remove(self._outfile)
        return None
    
    

class getParamsRegional(): #unfinished
    
    def __init__(self, input_file = 'param_inputs.csv', scenario = 'unnamed_scenario'):
        
        csv_data = pd.read_csv(input_file, header=None,index_col=0)
        param_mapping = pd.read_csv('param_mapping.csv', index_col=0)
        datapath = 'temp_data/'
        os.makedirs(datapath, exist_ok=True)
        outfile = datapath + scenario + '.dat'
        
        with open(outfile, 'w') as f:

            f.write('set Regions := ')
            regions = csv_data.loc['region',:].astype(str).tolist()
            for r in regions:
                f.write(r + ' ')
            f.write(';\n\n')

            csv_data_div = csv_data.merge(param_mapping[['coverage']], left_index=True, right_index=True)

            # create parameter matrix for global params
            for c in csv_data_div[csv_data_div.coverage=='gl'].iterrows():
                f.write('param {} := {};\n'.format(c[0],c[1].values[0]))
            f.write('\n')
            
            f.write('param:' + '\t')
            # create parameter matrix for (potentially) regional params
            for c in csv_data_div[csv_data_div.coverage=='reg'].iterrows():
                f.write(c[0] + '\t')
            f.write(':=\n\n')
            for r in range(len(regions)):
                f.write(regions[r] + '\t')
                for c in csv_data_div[csv_data_div.coverage=='reg'].iterrows():
                    f.write(c[1].values[r] + '\t')
            f.write(' ;\n\n')
            
            f.close()


class defaultParams():

    def __init__(self):

    #########################################################
    #Initial Parameters for DICE with DFAIR
    #########################################################
        
        self._region    = 'global'
        
        # Num times should be 81, time increment should be 5 years
        self._tstep     = 5         #Years in period
        self._numtimes  = 81
        self._yr0       = 2020          #Calendar year that corresponds to model year zero ##
        self._finalyr   = self._yr0 + self._tstep * self._numtimes
        self._t         = np.arange(0,self._numtimes+1)

        self._emshare0  = 0.2173        #Carbon emissions share into Reservoir 0
        self._emshare1  = 0.224         #Carbon emissions share into Reservoir 1
        self._emshare2  = 0.2824        #Carbon emissions share into Reservoir 2
        self._emshare3  = 0.2763        #Carbon emissions share into Reservoir 3
        self._tau0      = 1000000       #Decay time constant for R0  (year)
        self._tau1      = 394.4         #Decay time constant for R1  (year)
        self._tau2      = 36.53         #Decay time constant for R2  (year)
        self._tau3      = 4.304         #Decay time constant for R3  (year)
    
        self._teq1      = 0.324         #Thermal equilibration parameter for box 1 (m^2 per KW)
        self._teq2      = 0.44          #Thermal equilibration parameter for box 2 (m^2 per KW)
        self._d1        = 236           #Thermal response timescale for deep ocean (year)
        self._d2        = 4.07          #Thermal response timescale for upper ocean (year)

        self._irf0      = 32.4          #Pre-industrial IRF100 (year)
        self._irC       = 0.019         #Increase in IRF100 with cumulative carbon uptake (years per GtC)
        self._irT       = 4.165         #Increase in IRF100 with warming (years per degree K)   
        self._fco22x    = 3.93          #Forcings of equilibrium CO2 doubling (Wm-2)

        # INITIAL CONDITIONS CALIBRATED TO HISTORY
        self._mat0      = 886.5128014   #Initial concentration in atmosphere in 2020 (GtC)
        self._res00     = 150.093       #Initial concentration in Reservoir 0 in 2020
        self._res10     = 102.698       #Initial concentration in Reservoir 1 in 2020
        self._res20     = 39.534        #Initial concentration in Reservoir 2 in 2020
        self._res30     = 6.1865        #Initial concentration in Reservoir 3 in 2020

        self._mateq     = 588           #Equilibrium concentration atmosphere
        self._tbox10    = 0.1477        #Initial temperature box 1 change in 2020
        self._tbox20    = 1.099454      #Initial temperature box 2 change in 2020
        self._tatm0     = 1.24715       #Initial atmospheric temperature change in 2020 

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
        self._fosslim      = 6000   #Maximum cumulative extraction fossil fuels (GtC) ##
        self._CumEmiss0    = 633.5  #CumEmiss0 Cumulative emissions 2020 (GtC)
        self._emissrat2020 = 1.40   #Ratio of CO2e to industrial CO2 2020
        self._emissrat2100 = 1.21   #Ratio of CO2e to industrial CO2 2100

        ####################################################################
        #Climate damage parameter
        ####################################################################
        
        self._a1           = 0        #Damage intercept
        self._a2base       = 0.003467 #Damage quadratic term rev 01-13-23
        self._a3           = 2.00     #Damage exponent

        ####################################################################
        #Abatement cost
        ####################################################################

        self._expcost2     = 2.6    #Exponent of control cost function
        self._pback2050    = 515.0  #Cost of backstop 2019$ per tCO2 2050
        self._cprice1      = 6      #Carbon price 2020 2019$ per tCO2
        self._gcprice      = 0.025  #Growth rate of base carbon price per year
        self._gback        = -0.012 #Initial cost decline backstop cost per year ##

        ####################################################################
        #Limits on emissions controls, and other boundaries
        ####################################################################

        self._limmiu2070  = 1.0     #Emission control limit from 2070
        self._limmiu2120  = 1.1     #Emission control limit from 2120
        self._limmiu2200  = 1.05    #Emission control limit from 2220
        self._limmiu2300  = 1.0     #Emission control limit from 2300
        self._delmiumax   = 0.12    #Emission control delta limit per period
        self._klo         = 1       #Lower limit on K for all t
        self._clo         = 2       #Lower limit on C for all t
        self._cpclo       = 0.01    #Lower limit on CPC for all t
        self._rfactlong1  = 1000000 #Initial RFACTLONG at tfirst
        self._rfactlonglo = 0.0001  #Lower limit on RFACTLONG for all t
        self._alphalo     = 0.1     #Lower limit on alpha for all t
        self._alphaup     = 100     #Upper limit on alpha for all t
        self._matlo       = 10      #Lower limit on MAT for all t
        self._tatmlo      = 0.5     #Lower limit on TATM for all t
        self._tatmup      = 20      #Upper limit on TATM for all t
        self._sfx2200     = 0.28    #Fixed Gross savings rate after 2200

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

    def sim(self):
        
        self._L = [np.nan for i in range(self._numtimes+2)]                #
        self._aL = [np.nan for i in range(self._numtimes+2)]               #
        self._sigma = [np.nan for i in range(self._numtimes+2)]            #
        self._sigmatot = [np.nan for i in range(self._numtimes+2)]         #
        self._gA = [np.nan for i in range(self._numtimes+2)]               #
        self._gL = [np.nan for i in range(self._numtimes+2)]               #
        self._gsig = [np.nan for i in range(self._numtimes+2)]             #
        self._eland = [np.nan for i in range(self._numtimes+2)]            #
        self._cost1tot = [np.nan for i in range(self._numtimes+2)]         #
        self._pbacktime = [np.nan for i in range(self._numtimes+2)]        #
        self._cpricebase = [np.nan for i in range(self._numtimes+2)]       #
        self._gbacktime = [np.nan for i in range(self._numtimes+2)]        #
        self._varpcc = [np.nan for i in range(self._numtimes+2)]           #
        self._rprecaut = [np.nan for i in range(self._numtimes+2)]         #
        self._rr = [np.nan for i in range(self._numtimes+2)]               #
        self._rr1 = [np.nan for i in range(self._numtimes+2)]              #
        self._CO2E_GHGabateB = [np.nan for i in range(self._numtimes+2)]   #
        self._CO2E_GHGabateact = [np.nan for i in range(self._numtimes+2)] #
        self._F_Misc = [np.nan for i in range(self._numtimes+2)]           #
        self._emissrat = [np.nan for i in range(self._numtimes+2)]         #
        self._FORC_CO2 = [np.nan for i in range(self._numtimes+2)]         #

        for t in range(1,self._numtimes+2):
            
             #varpccEQ(m,t): # Variance of per capita consumption
            self._varpcc[t] = min(self._siggc1**2*5*(t-1), self._siggc1**2*5*47)
            
             #rprecautEQ(m,t): # Precautionary rate of return
            self._rprecaut[t] = (-0.5 * self._varpcc[t] * self._elasmu**2)
            
             #rr1EQ(m,t): # STP factor without precautionary factor
            rartp = np.exp(self._prstp + self._betaclim * self._pi)-1
            self._rr1[t] = 1/((1 + rartp)**(self._tstep * (t-1)))
            
             #rrEQ(m,t): # STP factor with precautionary factor
            self._rr[t] = self._rr1[t] * (1 + self._rprecaut[t])**(-self._tstep * (t-1))
            
             #LEQ(m,t): # Population adjustment over time (note initial condition)
            self._L[t] = self._L[t-1]*(self._popasym / self._L[t-1])**self._popadj if t > 1 else self._pop1
            
             #gAEQ(m,t): # Growth rate of productivity (note initial condition - can either enforce through bounds or this constraint)
            self._gA[t] = self._gA1 * np.exp(-self._delA * self._tstep * (t-1)) if t > 1 else self._gA1
            
             #aLEQ(m,t): # Level of total factor productivity (note initial condition)
            self._aL[t] = self._aL[t-1] /((1-self._gA[t-1])) if t > 1 else self._AL1

             #cpricebaseEQ(m,t): # Carbon price in base case of model
            self._cpricebase[t] = self._cprice1 * ((1 + self._gcprice)**(self._tstep * (t-1)))
            
             #pbacktimeEQ(m,t): # Backstop price 2019$ per ton CO2
            j = 0.01 if t <= 7 else 0.001
            self._pbacktime[t] = self._pback2050 * np.exp(-self._tstep * j * (t-7))
            
             #gsigEQ(m,t): # Change in rate of sigma (rate of decarbonization)
            self._gsig[t] = min(self._gsigma1 * (self._delgsig**(t-1)), self._asymgsig)
            
             #sigmaEQ(m,t): # CO2 emissions output ratio (note initial condition)
            self._sigma[t] = self._sigma[t-1]*np.exp(self._tstep*self._gsig[t-1]) if t > 1 else self._e1/(self._q1*(1-self._miu1))
            
             # elandEQ(m,t):
            self._eland[t] = self._eland0 * (1 - self._deland)**(t-1)
             
             # CO2E_GHGabateBEQ(m,t):
            self._CO2E_GHGabateB[t] = self._ECO2eGHGB2020 + ((self._ECO2eGHGB2100 - self._ECO2eGHGB2020) / 16) * (t-1) if t <= 16 else self._ECO2eGHGB2100
             
             # F_MiscEQ(m,t):
            self._F_Misc[t] = self._F_Misc2020 + ((self._F_Misc2100 - self._F_Misc2020) / 16) * (t-1) if t <= 16 else self._F_Misc2100
             
             # emissratEQ(m,t):
            self._emissrat[t] = self._emissrat2020 + ((self._emissrat2100 - self._emissrat2020) / 16) * (t-1) if t <= 16 else self._emissrat2100
             
             # sigmatotEQ(m,t):
            self._sigmatot[t] = self._sigma[t] * self._emissrat[t]
             
             # cost1totEQ(m,t):
            self._cost1tot[t] = self._pbacktime[t] * self._sigmatot[t] / self._expcost2 / 1000










