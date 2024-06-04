'''
Authors: Jacob Wessel, George Moraites
'''

from pyomo.environ import Var, ConcreteModel, RangeSet, Objective, value, Reals, NonNegativeReals, Constraint
from pyomo.environ import *
from pyomo.opt import SolverFactory

import numpy as np
import time

from DICE_params import modelParams
from dice_dynamics import simulateDynamics, dumpState

if __name__ == '__main__':
    num_times = 82
    tstep = 5

    params = modelParams(num_times, tstep)
    outputType = 1

    start_year = 2020
    final_year = start_year + tstep * num_times
    years = np.linspace(start_year, final_year, num_times+1, dtype=np.int32)

# params._fco22x, params._tatm0 in FAIRmodel
    argsv = [-1.0, outputType, num_times, params._tstep,
             params._al, params._l, params._sigma,
             params._cost1tot, params._eland, params._scale1, params._scale2,
             params._a1, params._a2base, params._a3,
             params._rr, params._gama,
             params._elasmu, params._prstp, params._expcost2,
             params._k0, params._dk, params._pbacktime]

    args = tuple(argsv)

    ###########################################################################
    # Pyomo Model Definition
    ###########################################################################
    model = ConcreteModel()
    model.time_periods = RangeSet(1, num_times)

    ###########################################################################
    # Boundaries and Initial Conditions
    ###########################################################################
    
    def MIUBounds(model,t):
        if t == 1:
            return (0, params._miu1) # initial value
        elif t == 2:
            return (0, params._miu2) # 2nd initial value
        elif (t>2)&(t<=8):
            return (0, params._delmiumax*(t-1))
        elif (t>8)&(t<=11):
            return (0, 0.85 + 0.05*(t-8))
        elif (t>11)&(t<=20):
            return (0,params._limmiu2070)
        elif (t>20)&(t<=37):
            return (0,params._limmiu2120)
        elif (t>37)&(t<=57):
            return (0,params._limmiu2200)
        elif t > 57:
            return (0,params._limmiu2300)

    def SBounds(model,t):
        if (t<=37)&(t<=num_times-10):
            return (-np.inf,np.inf)
        elif (t>37)&(t<=num_times-10):
            return (params._sfx2200, params._sfx2200) # 2nd initial value
        elif t>num_times-10:
            optlrsav = (params._dk+.004)/(params._dk+.004*params._elasmu+
                        np.exp(params._prstp+params._betaclim*params._pi)-1)*params._gama
            return (optlrsav, optlrsav)

    def KBounds(model,t):
        return (params._k0, params._k0) if t==1 else (params._klo, np.inf)

    def LBounds(model,t):
        return (params._pop1, params._pop1) if t==1 else (0, np.inf)

    def gABounds(model,t):
        return (params._gA1, params._gA1) if t==1 else (0, np.inf)

    def aLBounds(model,t):
        return (params._AL1, params._AL1) if t==1 else (0, np.inf)

    def gsigBounds(model,t):
        return (params._gsigma1, params._gsigma1) if t==1 else (-np.inf, np.inf)

    def sigmaBounds(model,t):
        sig1 = (params._e1)/(params._q1*(1-params._miu1))
        return (sig1, sig1) if t==1 else (0, np.inf)

    def CCATOTBounds(model,t):
        return (params._CumEmiss0, params._CumEmiss0) if t==1 else (-np.inf, np.inf)

    def MATBounds(model,t):
        return (params._mat0, params._mat0) if t==1 else (params._matlo, np.inf)

    def TATMBounds(model,t):
        return (params._tatm0, params._tatm0) if t==1 else (params._tatmlo, params._tatmup)

    def RES0Bounds(model,t):
        return (params._res00, params._res00) if t==1 else (-np.inf, np.inf)

    def RES1Bounds(model,t):
        return (params._res10, params._res00) if t==1 else (-np.inf, np.inf)

    def RES2Bounds(model,t):
        return (params._res20, params._res00) if t==1 else (-np.inf, np.inf)

    def RES3Bounds(model,t):
        return (params._res30, params._res00) if t==1 else (-np.inf, np.inf)

    def TBOX1Bounds(model,t):
        return (params._tbox10, params._tbox10) if t==1 else (-np.inf, np.inf)

    def TBOX2Bounds(model,t):
        return (params._tbox20, params._tbox20) if t==1 else (-np.inf, np.inf)

    def elandBounds(model,t):
        return (params._eland0, params._eland0) if t==1 else (-np.inf, np.inf)

    def F_GHGabateBounds(model,t):
        return (params._F_GHGabate2020, params._F_GHGabate2020) if t==1 else (-np.inf, np.inf)

    def RFACTLONGBounds(model,t):
        return (params._rfactlong1, params._rfactlong1) if t==1 else (params._rfactlonglo, np.inf)

    def alphaBounds(model,t):
        return (params._rfactlong1, params._rfactlong1) if t==1 else (params._rfactlonglo, np.inf)

    ###########################################################################
    # Time-varying parameters (defined as variables so they can be endogenized in the future, lowercase naming convention except for L)
    ###########################################################################
    
    model.L          = Var(model.time_periods, domain=NonNegativeReals, bounds=LBounds)                  #level of population and labor
    model.aL         = Var(model.time_periods, domain=NonNegativeReals, bounds=aLBounds)                 #level of productivity
    model.sigma      = Var(model.time_periods, domain=NonNegativeReals, bounds=sigmaBounds)              #CO2-emissions output ratio
    model.sigmatot   = Var(model.time_periods, domain=NonNegativeReals)                                  #GHG-output ratio
    model.gA         = Var(model.time_periods, domain=NonNegativeReals, bounds=gABounds)                 #productivity growth rate
    model.gL         = Var(model.time_periods, domain=NonNegativeReals)                                  #labor growth rate
    model.gsig       = Var(model.time_periods, domain=Reals, bounds=gsigBounds)                          #Change in sigma (rate of decarbonization)
    model.eland      = Var(model.time_periods, domain=NonNegativeReals, bounds=elandBounds)              #Emissions from deforestation (GtCO2 per year)
    model.cost1tot   = Var(model.time_periods, domain=NonNegativeReals)                                  #Abatement cost adjusted for backstop and sigma
    model.pbacktime  = Var(model.time_periods, domain=NonNegativeReals)                                  #Backstop price 2019$ per ton CO2
    model.cpricebase = Var(model.time_periods, domain=NonNegativeReals)                                  #Carbon price in base case
    model.gbacktime  = Var(model.time_periods, domain=NonNegativeReals)                                  #Decline rate of backstop price
    
    # precautionary dynamic parameters
    model.varpcc     = Var(model.time_periods, domain=NonNegativeReals)                                  #Variance of per capita consumption
    model.rprecaut   = Var(model.time_periods, domain=NonNegativeReals)                                  #Precautionary rate of return
    model.rr         = Var(model.time_periods, domain=NonNegativeReals)                                  #STP with precautionary factor
    model.rr1        = Var(model.time_periods, domain=NonNegativeReals)                                  #STP without precautionary factor
    
    # for post-processing:
    model.scc        = Var(model.time_periods, domain=NonNegativeReals)                                  #Social cost of carbon
    model.ppm        = Var(model.time_periods, domain=NonNegativeReals)                                  #Atmospheric concentrations parts per million
    model.atfrac2020 = Var(model.time_periods, domain=NonNegativeReals)                                  #Atmospheric share since 2020
    model.atfrac1765 = Var(model.time_periods, domain=NonNegativeReals)                                  #Atmospheric fraction of emissions since 1765
    model.abaterat   = Var(model.time_periods, domain=NonNegativeReals)                                  #Abatement cost per net output

    # Time-varying parameters in FAIR and Nonco2 modules (defined as variables so they can be endogenized in the future)
    model.CO2E_GHGabateB = Var(model.time_periods, domain=NonNegativeReals)                              #Abateable non-CO2 GHG emissions base
    model.CO2E_GHGabateB = Var(model.time_periods, domain=NonNegativeReals)                              #Abateable non-CO2 GHG emissions base (actual)
    model.F_Misc         = Var(model.time_periods, domain=Reals)                                         #Non-abateable forcings (GHG and other)
    model.emissrat       = Var(model.time_periods, domain=NonNegativeReals)                              #Ratio of CO2e to industrial emissions
    model.FORC_CO2       = Var(model.time_periods, domain=Reals)                                         #CO2 Forcings

    ###########################################################################
    # Variables (capitalized naming convention except for alpha)
    ###########################################################################
    # Nonnegative variables: MIU, TATM, MAT, Y, YNET, YGROSS, C, K, I, RFACTLONG, IRFt, alpha)
    
    model.UTILITY    = Var(domain=Reals)                                                      #utility function to maximize in objective
    model.MIU        = Var(model.time_periods, domain=NonNegativeReals, bounds=MIUBounds)                #emission control rate
    model.C          = Var(model.time_periods, domain=NonNegativeReals, bounds=(params._clo,np.inf))     #consumption (trillions 2019 US dollars per year)
    model.K          = Var(model.time_periods, domain=NonNegativeReals, bounds=KBounds)                  #capital stock (trillions 2019 US dollars)
    model.CPC        = Var(model.time_periods, domain=NonNegativeReals, bounds=(params._cpclo,np.inf))   #per capita consumption
    model.I          = Var(model.time_periods, domain=NonNegativeReals)                                  #Investment (trillions 2019 USD per year)
    model.S          = Var(model.time_periods, domain=Reals, bounds=SBounds)                             #gross savings rate as fraction of gross world product
    model.Y          = Var(model.time_periods, domain=NonNegativeReals)                                  #Gross world product net of abatement and damages (trillions 2019 USD per year)
    model.YGROSS     = Var(model.time_periods, domain=NonNegativeReals)                                  #Gross world product GROSS of abatement and damages (trillions 2019 USD per year)
    model.YNET       = Var(model.time_periods, domain=Reals)                                             #Output net of damages equation (trillions 2019 USD per year)
    model.DAMAGES    = Var(model.time_periods, domain=Reals)                                             #Damages (trillions 2019 USD per year)
    model.DAMFRAC    = Var(model.time_periods, domain=Reals)                                             #Damages as fraction of gross output
    model.ABATECOST  = Var(model.time_periods, domain=Reals)                                             #Cost of emissions reductions (trillions 2019 USD per year)
    model.MCABATE    = Var(model.time_periods, domain=Reals)                                             #Marginal cost of abatement (2019$ per ton CO2)
    model.CCATOT     = Var(model.time_periods, domain=Reals, bounds=CCATOTBounds)                        #Total emissions (GtC)
    model.PERIODU    = Var(model.time_periods, domain=Reals)                                             #One period utility function
    model.CPRICE     = Var(model.time_periods, domain=Reals)                                             #Carbon price (2019$ per ton of CO2)
    model.TOTPERIODU = Var(model.time_periods, domain=Reals)                                             #Period utility
    model.RFACTLONG  = Var(model.time_periods, domain=NonNegativeReals, bounds=RFACTLONGBounds)          #Long interest factor
    model.RSHORT     = Var(model.time_periods, domain=Reals)                                             #Long interest factor
    model.RLONG      = Var(model.time_periods, domain=Reals)                                             #Long interest factor

    # Variables in FAIR and Nonco2 modules
    model.FORC       = Var(model.time_periods, domain=Reals)                                                      #Increase in radiative forcing (watts per m2 from 1765)
    model.TATM       = Var(model.time_periods, domain=NonNegativeReals, bounds=TATMBounds)                        #temperature increase in atmosphere (degrees C from 1765)
    model.TBOX1      = Var(model.time_periods, domain=Reals, bounds=TBOX1Bounds)                                  #Increase temperature of box 1 (degrees C from 1765)
    model.TBOX2      = Var(model.time_periods, domain=Reals, bounds=TBOX2Bounds)                                  #Increase temperature of box 2 (degrees C from 1765)
    model.RES0       = Var(model.time_periods, domain=Reals, bounds=RES0Bounds)                                   #carbon concentration reservoir 0 (GtC from 1765)
    model.RES1       = Var(model.time_periods, domain=Reals, bounds=RES1Bounds)                                   #carbon concentration reservoir 1 (GtC from 1765)
    model.RES2       = Var(model.time_periods, domain=Reals, bounds=RES2Bounds)                                   #carbon concentration reservoir 2 (GtC from 1765)
    model.RES3       = Var(model.time_periods, domain=Reals, bounds=RES3Bounds)                                   #carbon concentration reservoir 3 (GtC from 1765)
    model.MAT        = Var(model.time_periods, domain=NonNegativeReals, bounds=MATBounds)                         #carbon concentration increase in atmosphere (GtC from 1765)
    model.CACC       = Var(model.time_periods, domain=Reals)                                                      #Accumulated carbon in ocean and other sinks (GtC)
    model.IRFt       = Var(model.time_periods, domain=NonNegativeReals)                                           #IRF100 at time t
    model.alpha      = Var(model.time_periods, domain=NonNegativeReals, bounds=(params._alphalo,params._alphaup)) #Carbon decay time scaling factor
    model.ECO2       = Var(model.time_periods, domain=Reals)                                                      #Total CO2 emissions (GtCO2 per year)
    model.ECO2E      = Var(model.time_periods, domain=Reals)                                                      #Total CO2e emissions including abateable nonCO2 GHG (GtCO2 per year)
    model.EIND       = Var(model.time_periods, domain=Reals)                                                      #Industrial CO2 emissions (GtCO2 per yr)
    model.F_GHGabate = Var(model.time_periods, domain=Reals, bounds=F_GHGabateBounds)                             #Forcings of abatable nonCO2 GHG
    


    ###########################################################################
    # Constraints (Equations)
    ###########################################################################
    
    ## Parameters
    def varpccEQ(model,t): # Variance of per capita consumption
        return model.varpcc[t] == min(params._siggc1**2*5*(t-1), params._siggc1**2*5*47)
    model._varpccEQ = Constraint(model.time_periods, rule=varpccEQ)
    
    def rprecautEQ(model,t): # Precautionary rate of return
        return model.rprecaut[t] == -0.5 * model.varpcc[t] * params._elasmu**2
    model._rprecautEQ = Constraint(model.time_periods, rule=rprecautEQ)
    
    def rr1EQ(model,t): # STP factor without precautionary factor
        rartp = np.exp(params._prstp + params._betaclim * params._pi)-1
        return model.rr1[t] == 1/((1 + rartp)**(params._tstep * (t-1)))
    model._rr1EQ = Constraint(model.time_periods, rule=rr1EQ)
    
    def rrEQ(model,t): # STP factor with precautionary factor
        return model.rr[t] == model.rr1[t] * (1 + model.rprecaut[t]**(-params._tstep * (t-1)))
    model._rrEQ = Constraint(model.time_periods, rule=rrEQ)
    
    def LEQ(model,t): # Population adjustment over time (note initial condition)
        return model.L[t] == model.L[t-1]*(params._popasym / model.L[t-1])**params._popadj if t > 1 else Constraint.Skip
    model._LEQ = Constraint(model.time_periods, rule=LEQ)
    
    def gAEQ(model,t): # Growth rate of productivity (note initial condition - can either enforce through bounds or this constraint)
        return model.gA[t] == params._gA1 * np.exp(-params._delA * 5.0 * (t-1)) if t > 1 else Constraint.Skip
    model._gAEQ = Constraint(model.time_periods, rule=gAEQ)
    
    def aLEQ(model,t): # Level of total factor productivity (note initial condition)
        return model.aL[t] == model.aL[t-1] /((1-model.gA[t-1])) if t > 1 else Constraint.Skip
    model._aLEQ = Constraint(model.time_periods, rule=aLEQ)
    
    ### CONTINUE HERE
    
    model.cpricebase[i]   = params._cprice1*(1+params._gcprice)**(5*(params._t[i]-1)) #Carbon price in base case of model
    model.pbacktime[i]    = params._pback2050 * np.exp(-5*(0.01 if params._t[i] <= 7 else 0.001)*(model.t[i]-7)) #Backstop price 2019$ per ton CO2. Incorporates 2023 condition
    model.gsig[i]         = min(params._gsigma1*params._delgsig **((params._t[i]-1)), params._asymgsig) #Change in rate of sigma (rate of decarbonization)
    model.sigma[i]        = model.sigma[i-1]*np.exp(5*model.gsig[i-1])
    
    ## Main DICE Equations
    
    # Emissions and Damages
    model.eco2[i] = (model.sigma[i] * model.ygross[i] + model.eland[i]) * (1-MIUopt) #New
    model.eind[i] = model.sigma[i] * model.ygross[i] * (1.0 - MIUopt[i])
    model.eco2e[i] = (model.sigma[i] * model.ygross[i] + model.eland[i] + model.CO2E_GHGabateB[i]) * (1-MIUopt) #New
    model.F_GHGabate[i] = params._F_GHGabate2020*model.F_GHGabate[i-1] + params._F_GHGabate2100*model.CO2E_GHGabateB[i-1]*(1-MIUopt[i-1])
    model.ccatot[i] = model.ccatot[i] + model.eco2[i]*(5/3.666)
    model.damfrac[i] = (params._a1 * model.tatm[i] + params._a2base * model.tatm[i] ** params._a3)  
    model.damages[i] = model.ygross[i] * model.damfrac[i]
    model.abatecost[i] = model.ygross[i] * model.cost1tot[i] * MIUopt[i]**params._expcost2
    model.mcabate[i] = model.pbacktime[i] * MIUopt[i]**(params._expcost2-1)
    model.cprice[i] = model.pbacktime[i] * MIUopt[i]**(params._expcost2-1)
    
    # Economic Variables
    model.ygross[i] = model.aL[i] * ((model.L[i]/params._MILLE)**(1.0-params._gama)) * model.K[i]**params._gama  #Gross world product GROSS of abatement and damages (trillions 20i9 USD per year)
    model.ynet[i]         = model.ygross[i] * (1-model.damfrac[i])
    model.Y[i]            = model.ynet[i] - model.abatecost[i]
    model.C[i]            = model.Y[i] - model.I[i]
    model.CPC[i]          = params._MILLE * model.C[i]/model.L[i]
    model.I[i]            = Sopt[i] * model.Y[i]
    model.K[i]           <= (1.0 - params._dk)**params._tstep * model.K[i-1] + params._tstep * model.I[i]
    model.rfactlong[i]    = (params._SRF * (model.CPC[i]/model.CPC[i-1])**(-params._elasmu)*model.rr[i]) #Modified/New
    model.rlong[i]        = -math.log(model.rfactlong[i]/params._SRF)/(5*(i-1)) #NEW
    model.rshort[i]       = (-math.log(model.rfactlong[i]/model.rfactlong[i-1])/5) #NEW

    ## FAIR Climate Module Equations
    model.res0[i] = (params._emshare0 * params._tau0 * model.alpha[i] * 
                            (model.eco2[i] / 3.667) * 
                            (1 - np.exp(-params._tstep / (params._tau0 * model.alpha[i]))) + 
                            model.res0[i-1] * np.exp(-params._tstep / (params._tau0 * model.alpha[i])))

    model.res1[i] = (params._emshare1 * params._tau1 * model.alpha[i] * 
                            (model.eco2[i] / 3.667) * 
                            (1 - np.exp(-params._tstep / (params._tau1 * model.alpha[i]))) + 
                            model.res1[i-1] * np.exp(-params._tstep / (params._tau1 * model.alpha[i])))

    model.res2[i] = (params._emshare2 * params._tau2 * model.alpha[i] * 
                            (model.eco2[i] / 3.667) * 
                            (1 - np.exp(-params._tstep / (params._tau2 * model.alpha[i]))) + 
                            model.res2[i-1] * np.exp(-params._tstep / (params._tau2 * model.alpha[i])))

    model.res3[i] = (params._emshare3 * params._tau3 * model.alpha[i] * 
                            (model.eco2[i] / 3.667) * 
                            (1 - np.exp(-params._tstep / (params._tau3 * model.alpha[i]))) + 
                            model.res3[i-1] * np.exp(-params._tstep / (params._tau3 * model.alpha[i])))
    model.mat[i] = params._mateq + model.res0[i] + model.res1[i] + model.res2[i] + model.res3[i]
    model.cacc[i] = (model.ccatot[i-1] - (model.mat[i-1] - params._mateq))
    self.FORC[i] = (params._fco22x * ((math.log((model.mat[i-1]+1e-9/params._mateq))/math.log(2)) 
                        + model.F_Misc[i-1] + model.F_GHGabate[i-1]))
    model.tbox1[i] = (model.tbox1[i-1] *
                            np.exp(params._tstep/params._d1) + params._teq1 *
                            model.force[i] * (1-np.exp(params._tstep/params._d1)))

    model.tbox2[i] = (model.tbox2[i-1] *
                            np.exp(params._tstep/params._d2) + params._teq2 *
                            model.force[i] * (1-np.exp(params._tstep/params._d2)))

    model.tatm[i] = np.clip(model.tbox1[i-1] + model.tbox2[i-1], 0.5, 20)
    model.irfeqlhs[i]   =  ((model.alpha[i] * params._emshare0 * params._tau0 * (1 - np.exp(-100 / (model.alpha[i] * params._tau0)))) +
                            (model.alpha[i] * params._emshare1 * params._tau1 * (1 - np.exp(-100 / (model.alpha[i] * params._tau1)))) +
                            (model.alpha[i] * params._emshare2 * params._tau2 * (1 - np.exp(-100 / (model.alpha[i] * params._tau2)))) +
                            (model.alpha[i] * params._emshare3 * params._tau3 * (1 - np.exp(-100 / (model.alpha[i] * params._tau3)))))
    model.irfeqrhs[i]   =  params._irf0 + params._irC * model.cacc[i] + params._irT * model.tatm[i]
    self.IRFt[i] = model.irfeqlhs[i] - model.irfeqrhs[i] # IRF equation: constrain lhs to equal rhs at all times

    ## NonCO2 Forcings Equations
    model.eland[i] = params._eland0 * (1 - params._deland) ** (model.t[i]-1)
    if params._t[i-1] <= 16:
        model.CO2E_GHGabateB[i-1] = params._ECO2eGHGB2020 + ((params._ECO2eGHGB2100-params._ECO2eGHGB2020)/16)*(params._t[i-1]-1)
        model.F_Misc[i-1]=params._F_Misc2020 + ((params._F_Misc2100-params._F_Misc2020)/16)*(params._t[i-1]-1)
    else:
        model.CO2E_GHGabateB[i-1] = params._ECO2eGHGB2100
        model.F_Misc[i-1]=params._F_Misc2100
    if model.t[i] <= 16:
        model.emissrat[i] = params._emissrat2020 +((params._emissrat2100-params._emissrat2020)/16)*(model.t[i]-1)
    else:
        model.emissrat[i] = params._emissrat2100
    model.sigmatot[i]     = model.sigma[i]*model.emissrat[i]
    model.cost1tot[i]     = model.pbacktime[i]*model.sigmatot[i]/params._expcost2/params._MILLE
    
    # For Computing Objective
    model.periodu[i]      = ((model.C[i]*params._MILLE/model.L[i])**(1.0-params._elasmu)-1.0) / (1.0 - params._elasmu) - 1.0
    model.totperiodu[i]   = model.periodu[i] * model.L[i] * model.rr[i]
    
    
    
    
    


    ###########################################################################
    # Objective Function: MAXIMIZE UTILITY = tstep * scale1 * sum(t,  TOTPERIODU(t)) + scale2
    ###########################################################################
    def obj_rule(model):
        return simulateDynamics([value(model.MIU[t]) for t in model.time_periods], *args)
    model.obj = Objective(rule=obj_rule)







    ###########################################################################
    # Optimization
    ###########################################################################
    opt = SolverFactory('gurobi')

    # Solve the model
    start = time.time()
    results = opt.solve(model, tee=True)
    end = time.time()
    print("Time Elapsed:", end - start)







    ###########################################################################
    # Post-Processing
    ###########################################################################

    print("Dumping graphs to file.")
    output = simulateDynamics([value(model.MIU[t]) for t in model.time_periods], *args)
    dumpState(years, output, "./results/base_case_state_post_opt.csv")

    # Put these below equations and dumpState in separate py file to call as functions
    # ### Post-Solution Parameter-Assignment ###
    # model.scc[i]        = -1000 * model.eco2[i] / (.00001 + model.C[i]) # NOTE: THESE (model.eco2[i] and model.C[i]) NEED TO BE MARGINAL VALUES, NOT THE SOLUTIONS THEMSELVES
    # model.ppm[i]        = model.mat[i] / 2.13
    # model.abaterat[i]   = model.abatecost[i] / model.Y[i]
    # model.atfrac2020[i] = (model.mat[i] - params._mat0) / (model.ccatot[i] + .00001 - params._CumEmiss0)
    # model.atfrac1765[i] = (model.mat[i] - params._mateq) / (.00001 + model.ccatot[i])
    # model.FORC_CO2[i]   = params._fco22x * ((math.log((model.mat[i] / params._mateq)) / math.log(2)))

    print("Completed.")