'''
Authors: Jacob Wessel, George Moraites
'''

from pyomo.environ import Var, ConcreteModel, RangeSet, Objective, value, Reals, NonNegativeReals, Constraint, maximize, summation, exp, log
from pyomo.environ import *
from pyomo.opt import SolverFactory

import numpy as np
import time

from DICE_params import defaultParams
#from dice_dynamics import simulateDynamics, dumpState



if __name__ == '__main__':
    
    solver_name = 'ipopt'   # enter name of solver here
    
    p = defaultParams()        # get model parameters
    
    outputType = 1

    years = np.linspace(p._yr0, p._finalyr, p._numtimes+1, dtype=np.int32)

    c_to_co2 = 11/3          # conversion factor from carbon to equivalent CO2


    # argsv = [-1.0, outputType, p._numtimes, p._tstep,
    #          p._al, p._l, p._sigma,
    #          p._cost1tot, p._eland, p._scale1, p._scale2,
    #          p._a1, p._a2base, p._a3,
    #          p._rr, p._gama,
    #          p._elasmu, p._prstp, p._expcost2,
    #          p._k0, p._dk, p._pbacktime]
    # args = tuple(argsv)


    ###########################################################################
    # Pyomo Model Definition
    ###########################################################################

    m = ConcreteModel()
    m.time_periods = RangeSet(1, p._numtimes+1)



    ###########################################################################
    # Boundaries and Initial Conditions
    ###########################################################################
    
    def MIUBounds(m,t):
        if t == 1:
            return (0, p._miu1) # initial value
        elif t == 2:
            return (0, p._miu2) # 2nd initial value
        elif (t > 2) & (t <= 8):
            return (0, p._delmiumax * (t - 1))
        elif (t > 8) & (t <= 11):
            return (0, 0.85 + 0.05 * (t - 8))
        elif (t > 11)&(t<=20):
            return (0, p._limmiu2070)
        elif (t > 20) & (t <= 37):
            return (0, p._limmiu2120)
        elif (t > 37) & (t <= 57):
            return (0, p._limmiu2200)
        elif t > 57:
            return (0, p._limmiu2300)

    def SBounds(m,t): # lag 10
        if (t <= 37) & (t <= p._numtimes - 10):
            return (-np.inf,np.inf)
        elif (t > 37) & (t <= p._numtimes - 10):
            return (p._sfx2200, p._sfx2200) # 2nd initial value
        elif t > p._numtimes - 10:
            optlrsav = ((p._dk + .004) / (p._dk + .004 * p._elasmu +
                       exp(p._prstp + p._betaclim * p._pi) - 1)) * p._gama
            return (optlrsav, optlrsav)

    def KBounds(m,t):
        return (p._k0, p._k0) if t==1 else (p._klo, np.inf)

    def LBounds(m,t):
        return (p._pop1, p._pop1) if t==1 else (0, np.inf)

    def gABounds(m,t):
        return (p._gA1, p._gA1) if t==1 else (0, np.inf)

    def aLBounds(m,t):
        return (p._AL1, p._AL1) if t==1 else (0, np.inf)

    def gsigBounds(m,t):
        return (p._gsigma1, p._gsigma1) if t==1 else (-np.inf, np.inf)

    def sigmaBounds(m,t):
        sig1 = p._e1 / (p._q1 * (1 - p._miu1))
        return (sig1, sig1) if t==1 else (0, np.inf)

    def CCATOTBounds(m,t):
        return (p._CumEmiss0, p._CumEmiss0) if t==1 else (-np.inf, np.inf)

    def MATBounds(m,t):
        return (p._mat0, p._mat0) if t==1 else (p._matlo, np.inf)

    def TATMBounds(m,t):
        return (p._tatm0, p._tatm0) if t==1 else (p._tatmlo, p._tatmup)

    def RES0Bounds(m,t):
        return (p._res00, p._res00) if t==1 else (-np.inf, np.inf)

    def RES1Bounds(m,t):
        return (p._res10, p._res10) if t==1 else (-np.inf, np.inf)

    def RES2Bounds(m,t):
        return (p._res20, p._res20) if t==1 else (-np.inf, np.inf)

    def RES3Bounds(m,t):
        return (p._res30, p._res30) if t==1 else (-np.inf, np.inf)

    def TBOX1Bounds(m,t):
        return (p._tbox10, p._tbox10) if t==1 else (-np.inf, np.inf)

    def TBOX2Bounds(m,t):
        return (p._tbox20, p._tbox20) if t==1 else (-np.inf, np.inf)

    def elandBounds(m,t):
        return (p._eland0, p._eland0) if t==1 else (-np.inf, np.inf)

    def F_GHGabateBounds(m,t):
        return (p._F_GHGabate2020, p._F_GHGabate2020) if t==1 else (-np.inf, np.inf)

    def RFACTLONGBounds(m,t):
        return (p._rfactlong1, p._rfactlong1) if t==1 else (p._rfactlonglo, np.inf)

    def alphaBounds(m,t):
        return (p._rfactlong1, p._rfactlong1) if t==1 else (p._rfactlonglo, np.inf)





    ###########################################################################
    # Time-varying parameters (defined as variables so they can be endogenized in the future, lowercase naming convention except for L)
    ###########################################################################
    
    m.L          = Var(m.time_periods, domain=NonNegativeReals, bounds=LBounds)                 #level of population and labor
    m.aL         = Var(m.time_periods, domain=NonNegativeReals, bounds=aLBounds)                #level of productivity
    m.sigma      = Var(m.time_periods, domain=NonNegativeReals, bounds=sigmaBounds)             #CO2-emissions output ratio
    m.sigmatot   = Var(m.time_periods, domain=NonNegativeReals)                                 #GHG-output ratio
    m.gA         = Var(m.time_periods, domain=NonNegativeReals, bounds=gABounds)                #productivity growth rate
    m.gL         = Var(m.time_periods, domain=NonNegativeReals)                                 #labor growth rate
    m.gsig       = Var(m.time_periods, domain=Reals, bounds=gsigBounds)                         #Change in sigma (rate of decarbonization)
    m.eland      = Var(m.time_periods, domain=NonNegativeReals, bounds=elandBounds)             #Emissions from deforestation (GtCO2 per year)
    m.cost1tot   = Var(m.time_periods, domain=NonNegativeReals)                                 #Abatement cost adjusted for backstop and sigma
    m.pbacktime  = Var(m.time_periods, domain=NonNegativeReals)                                 #Backstop price 2019$ per ton CO2
    m.cpricebase = Var(m.time_periods, domain=NonNegativeReals)                                 #Carbon price in base case
    m.gbacktime  = Var(m.time_periods, domain=NonNegativeReals)                                 #Decline rate of backstop price
    
    # precautionary dynamic parameters
    m.varpcc     = Var(m.time_periods, domain=NonNegativeReals)                                 #Variance of per capita consumption
    m.rprecaut   = Var(m.time_periods, domain=NonNegativeReals)                                 #Precautionary rate of return
    m.rr         = Var(m.time_periods, domain=NonNegativeReals)                                 #STP with precautionary factor
    m.rr1        = Var(m.time_periods, domain=NonNegativeReals)                                 #STP without precautionary factor
    
    # for post-processing:
    #m.scc        = Var(m.time_periods, domain=NonNegativeReals)                                 #Social cost of carbon
    #m.ppm        = Var(m.time_periods, domain=NonNegativeReals)                                 #Atmospheric concentrations parts per million
    #m.atfrac2020 = Var(m.time_periods, domain=NonNegativeReals)                                 #Atmospheric share since 2020
    #m.atfrac1765 = Var(m.time_periods, domain=NonNegativeReals)                                 #Atmospheric fraction of emissions since 1765
    #m.abaterat   = Var(m.time_periods, domain=NonNegativeReals)                                 #Abatement cost per net output
    #m.FORC_CO2         = Var(m.time_periods, domain=Reals)                                        #CO2 Forcings

    # Time-varying parameters in FAIR and Nonco2 modules (defined as variables so they can be endogenized in the future)
    m.CO2E_GHGabateB   = Var(m.time_periods, domain=NonNegativeReals)                             #Abateable non-CO2 GHG emissions base
    m.CO2E_GHGabateact = Var(m.time_periods, domain=NonNegativeReals)                             #Abateable non-CO2 GHG emissions base (actual)
    m.F_Misc           = Var(m.time_periods, domain=Reals)                                        #Non-abateable forcings (GHG and other)
    m.emissrat         = Var(m.time_periods, domain=NonNegativeReals)                             #Ratio of CO2e to industrial emissions



    ###########################################################################
    # Variables (capitalized naming convention except for alpha)
    ###########################################################################
    # Nonnegative variables: MIU, TATM, MAT, Y, YNET, YGROSS, C, K, I, RFACTLONG, IRFt, alpha)
    
    m.UTILITY    = Var(domain=Reals)                                                            #utility function to maximize in objective
    m.MIU        = Var(m.time_periods, domain=NonNegativeReals, bounds=MIUBounds)               #emission control rate
    m.C          = Var(m.time_periods, domain=NonNegativeReals, bounds=(p._clo,np.inf))         #consumption (trillions 2019 US dollars per year)
    m.K          = Var(m.time_periods, domain=NonNegativeReals, bounds=KBounds)                 #capital stock (trillions 2019 US dollars)
    m.CPC        = Var(m.time_periods, domain=NonNegativeReals, bounds=(p._cpclo,np.inf))       #per capita consumption
    m.I          = Var(m.time_periods, domain=NonNegativeReals)                                 #Investment (trillions 2019 USD per year)
    m.S          = Var(m.time_periods, domain=Reals, bounds=SBounds)                            #gross savings rate as fraction of gross world product
    m.Y          = Var(m.time_periods, domain=NonNegativeReals)                                 #Gross world product net of abatement and damages (trillions 2019 USD per year)
    m.YGROSS     = Var(m.time_periods, domain=NonNegativeReals)                                 #Gross world product GROSS of abatement and damages (trillions 2019 USD per year)
    m.YNET       = Var(m.time_periods, domain=NonNegativeReals)                                 #Output net of damages equation (trillions 2019 USD per year)
    m.DAMAGES    = Var(m.time_periods, domain=Reals)                                            #Damages (trillions 2019 USD per year)
    m.DAMFRAC    = Var(m.time_periods, domain=Reals)                                            #Damages as fraction of gross output
    m.ABATECOST  = Var(m.time_periods, domain=Reals)                                            #Cost of emissions reductions (trillions 2019 USD per year)
    m.MCABATE    = Var(m.time_periods, domain=Reals)                                            #Marginal cost of abatement (2019$ per ton CO2)
    m.CCATOT     = Var(m.time_periods, domain=Reals, bounds=CCATOTBounds)                       #Total emissions (GtC)
    m.PERIODU    = Var(m.time_periods, domain=Reals)                                            #One period utility function
    m.CPRICE     = Var(m.time_periods, domain=Reals)                                            #Carbon price (2019$ per ton of CO2)
    m.TOTPERIODU = Var(m.time_periods, domain=Reals)                                            #Period utility
    m.RFACTLONG  = Var(m.time_periods, domain=NonNegativeReals, bounds=RFACTLONGBounds)         #Long interest factor
    m.RSHORT     = Var(m.time_periods, domain=Reals)                                            #Long interest factor
    m.RLONG      = Var(m.time_periods, domain=Reals)                                            #Long interest factor

    # Variables in FAIR and Nonco2 modules
    m.FORC       = Var(m.time_periods, domain=Reals)                                            #Increase in radiative forcing (watts per m2 from 1765)
    m.TATM       = Var(m.time_periods, domain=NonNegativeReals, bounds=TATMBounds)              #temperature increase in atmosphere (degrees C from 1765)
    m.TBOX1      = Var(m.time_periods, domain=Reals, bounds=TBOX1Bounds)                        #Increase temperature of box 1 (degrees C from 1765)
    m.TBOX2      = Var(m.time_periods, domain=Reals, bounds=TBOX2Bounds)                        #Increase temperature of box 2 (degrees C from 1765)
    m.RES0       = Var(m.time_periods, domain=Reals, bounds=RES0Bounds)                         #carbon concentration reservoir 0 (GtC from 1765)
    m.RES1       = Var(m.time_periods, domain=Reals, bounds=RES1Bounds)                         #carbon concentration reservoir 1 (GtC from 1765)
    m.RES2       = Var(m.time_periods, domain=Reals, bounds=RES2Bounds)                         #carbon concentration reservoir 2 (GtC from 1765)
    m.RES3       = Var(m.time_periods, domain=Reals, bounds=RES3Bounds)                         #carbon concentration reservoir 3 (GtC from 1765)
    m.MAT        = Var(m.time_periods, domain=NonNegativeReals, bounds=MATBounds)               #carbon concentration increase in atmosphere (GtC from 1765)
    m.CACC       = Var(m.time_periods, domain=Reals)                                            #Accumulated carbon in ocean and other sinks (GtC)
    m.IRFt       = Var(m.time_periods, domain=NonNegativeReals)                                 #IRF100 at time t
    m.ECO2       = Var(m.time_periods, domain=Reals)                                            #Total CO2 emissions (GtCO2 per year)
    m.ECO2E      = Var(m.time_periods, domain=Reals)                                            #Total CO2e emissions including abateable nonCO2 GHG (GtCO2 per year)
    m.EIND       = Var(m.time_periods, domain=Reals)                                            #Industrial CO2 emissions (GtCO2 per yr)
    m.F_GHGabate = Var(m.time_periods, domain=Reals, bounds=F_GHGabateBounds)                   #Forcings of abatable nonCO2 GHG
    m.alpha      = Var(m.time_periods, domain=NonNegativeReals, bounds=(p._alphalo,p._alphaup)) #Carbon decay time scaling factor
    





    ###########################################################################
    # Constraints (Equations)
    ###########################################################################
    
    ## Parameters
    
    def varpccEQ(m,t): # Variance of per capita consumption
        return m.varpcc[t] == min((p._siggc1**2)*5*(t-1), (p._siggc1**2)*5*47)
    m._varpccEQ = Constraint(m.time_periods, rule=varpccEQ)
    
    def rprecautEQ(m,t): # Precautionary rate of return
        return m.rprecaut[t] == -0.5 * m.varpcc[t] * p._elasmu**2
    m._rprecautEQ = Constraint(m.time_periods, rule=rprecautEQ)
    
    def rr1EQ(m,t): # STP factor without precautionary factor
        rartp = exp(p._prstp + p._betaclim * p._pi)-1
        return m.rr1[t] == 1/((1 + rartp)**(p._tstep * (t-1)))
    m._rr1EQ = Constraint(m.time_periods, rule=rr1EQ)
    
    def rrEQ(m,t): # STP factor with precautionary factor
        return m.rr[t] == m.rr1[t] * (1 + m.rprecaut[t])**(-p._tstep * (t-1))
    m._rrEQ = Constraint(m.time_periods, rule=rrEQ)
    
    def LEQ(m,t): # Population adjustment over time (note initial condition)
        return m.L[t] == m.L[t-1]*(p._popasym / m.L[t-1])**p._popadj if t > 1 else Constraint.Skip
    m._LEQ = Constraint(m.time_periods, rule=LEQ)
    
    def gAEQ(m,t): # Growth rate of productivity (note initial condition - can either enforce through bounds or this constraint)
        return m.gA[t] == p._gA1 * exp(-p._delA * p._tstep * (t-1)) if t > 1 else Constraint.Skip
    m._gAEQ = Constraint(m.time_periods, rule=gAEQ)
    
    def aLEQ(m,t): # Level of total factor productivity (note initial condition)
        return m.aL[t] == m.aL[t-1] /((1-m.gA[t-1])) if t > 1 else Constraint.Skip
    m._aLEQ = Constraint(m.time_periods, rule=aLEQ)

    def cpricebaseEQ(m,t): # Carbon price in base case of model
        return m.cpricebase[t] == p._cprice1 * ((1 + p._gcprice)**(p._tstep * (t-1)))
    m._cpricebaseEQ = Constraint(m.time_periods, rule=cpricebaseEQ)
    
    def pbacktimeEQ(m,t): # Backstop price 2019$ per ton CO2
        j = 0.01 if t <= 7 else 0.001
        return m.pbacktime[t] == p._pback2050 * exp(-p._tstep * j * (t-7))
    m._pbacktimeEQ = Constraint(m.time_periods, rule=pbacktimeEQ)
    
    def gsigEQ(m,t): # Change in rate of sigma (rate of decarbonization)
        return m.gsig[t] == min(p._gsigma1 * (p._delgsig**(t-1)), p._asymgsig)
    m._gsigEQ = Constraint(m.time_periods, rule=gsigEQ)
    
    def sigmaEQ(m,t): # CO2 emissions output ratio (note initial condition)
        return m.sigma[t] == m.sigma[t-1] * exp(p._tstep * m.gsig[t-1]) if t > 1 else Constraint.Skip
    m._sigmaEQ = Constraint(m.time_periods, rule=sigmaEQ)
    
    
    
    ## Main DICE Equations
    
    # Emissions and Damages
    
    def ECO2EQ(m,t):
        return m.ECO2[t] == (m.sigma[t] * m.YGROSS[t] + m.eland[t]) * (1 - m.MIU[t])
    m._ECO2EQ = Constraint(m.time_periods, rule=ECO2EQ)
    
    def EINDEQ(m,t):
        return m.EIND[t] == m.sigma[t] * m.YGROSS[t] * (1 - m.MIU[t])
    m._EINDEQ = Constraint(m.time_periods, rule=EINDEQ)
    
    def ECO2EEQ(m,t):
        return m.ECO2E[t] == (m.sigma[t] * m.YGROSS[t] + m.eland[t] + m.CO2E_GHGabateB[t]) * (1-m.MIU[t])
    m._ECO2EEQ = Constraint(m.time_periods, rule=ECO2EEQ)
    
    def F_GHGabateEQ(m,t): # note initial condition
        return m.F_GHGabate[t] == p._Fcoef2 * m.F_GHGabate[t-1] + p._Fcoef1 * m.CO2E_GHGabateB[t-1] * (1-m.MIU[t-1]) if t > 1 else Constraint.Skip
    m._F_GHGabateEQ = Constraint(m.time_periods, rule=F_GHGabateEQ)
    
    def CCATOTEQ(m,t): # note initial condition
        return m.CCATOT[t] == m.CCATOT[t-1] + m.ECO2[t-1] * (p._tstep / c_to_co2) if t > 1 else Constraint.Skip
    m._CCATOTEQ = Constraint(m.time_periods, rule=CCATOTEQ)
    
    def DAMFRACEQ(m,t):
        return m.DAMFRAC[t] == (p._a1 * m.TATM[t]) + (p._a2base * (m.TATM[t] ** p._a3))
    m._DAMFRACEQ = Constraint(m.time_periods, rule=DAMFRACEQ)
    
    def DAMAGESEQ(m,t):
        return m.DAMAGES[t] == m.YGROSS[t] * m.DAMFRAC[t]
    m._DAMAGESEQ = Constraint(m.time_periods, rule=DAMAGESEQ)
    
    def ABATECOSTEQ(m,t):
        return m.ABATECOST[t] == m.YGROSS[t] * m.cost1tot[t] * (m.MIU[t]**p._expcost2)
    m._ABATECOSTEQ = Constraint(m.time_periods, rule=ABATECOSTEQ)
    
    def MCABATEEQ(m,t):
        return m.MCABATE[t] == m.pbacktime[t] * (m.MIU[t]**(p._expcost2 - 1))
    m._MCABATEEQ = Constraint(m.time_periods, rule=MCABATEEQ)
    
    def CPRICEEQ(m,t):
        return m.CPRICE[t] == m.pbacktime[t] * ((m.MIU[t])**(p._expcost2 - 1))
    m._CPRICEEQ = Constraint(m.time_periods, rule=CPRICEEQ)


    
    # Economic Variables
    
    def YGROSSEQ(m,t): # Gross world product GROSS of abatement and damages (trillions 20i9 USD per year)
        return m.YGROSS[t] == (m.aL[t] * ((m.L[t] / 1000)**(1 - p._gama))) * (m.K[t]**p._gama)
    m._YGROSSEQ = Constraint(m.time_periods, rule=YGROSSEQ)
    
    def YNETEQ(m,t):
        return m.YNET[t] == m.YGROSS[t] * (1 - m.DAMFRAC[t])
    m._YNETEQ = Constraint(m.time_periods, rule=YNETEQ)
    
    def YEQ(m,t):
        return m.Y[t] == m.YNET[t] - m.ABATECOST[t]
    m._YEQ = Constraint(m.time_periods, rule=YEQ)
    
    def CEQ(m,t):
        return m.C[t] == m.Y[t] - m.I[t]
    m._CEQ = Constraint(m.time_periods, rule=CEQ)
    
    def CPCEQ(m,t):
        return m.CPC[t] == 1000 * m.C[t] / m.L[t]
    m._CPCEQ = Constraint(m.time_periods, rule=CPCEQ)
    
    def IEQ(m,t):
        return m.I[t] == m.S[t] * m.Y[t]
    m._IEQ = Constraint(m.time_periods, rule=IEQ)
    
    def KEQ(m,t): # note initial condition
        return m.K[t] <= ((1 - p._dk)**p._tstep) * m.K[t-1] + p._tstep * m.I[t-1] if t > 1 else Constraint.Skip
    m._KEQ = Constraint(m.time_periods, rule=KEQ)
    
    def RFACTLONGEQ(m,t): # note initial condition
        return m.RFACTLONG[t] == p._SRF * ((m.CPC[t] / m.CPC[1])**(-p._elasmu))*m.rr[t] if t > 1 else Constraint.Skip #NEW
    m._RFACTLONGEQ = Constraint(m.time_periods, rule=RFACTLONGEQ) ##
    
    def RLONGEQ(m,t): # note initial condition
        return m.RLONG[t] == -log(m.RFACTLONG[t] / p._SRF) / (p._tstep * (t-1)) if t > 1 else Constraint.Skip #NEW
    m._RLONGEQ = Constraint(m.time_periods, rule=RLONGEQ) ##
    
    def RSHORTEQ(m,t): # note initial condition
        return m.RSHORT[t] == -log(m.RFACTLONG[t] / m.RFACTLONG[t-1]) / p._tstep if t > 1 else Constraint.Skip #NEW
    m._RSHORTEQ = Constraint(m.time_periods, rule=RSHORTEQ)



    ## FAIR Climate Module Equations
    
    def RES0LOM(m,t): # note initial condition
        return m.RES0[t] == (p._emshare0 * p._tau0 * m.alpha[t] * (m.ECO2[t] / c_to_co2) * (1 - exp(-p._tstep / (p._tau0 * m.alpha[t]))) + 
                                m.RES0[t-1] * exp(-p._tstep / (p._tau0 * m.alpha[t]))) if t > 1 else Constraint.Skip #NEW
    m._RES0LOM = Constraint(m.time_periods, rule=RES0LOM)
    
    def RES1LOM(m,t): # note initial condition
        return m.RES1[t] == (p._emshare1 * p._tau1 * m.alpha[t] * (m.ECO2[t] / c_to_co2) * (1 - exp(-p._tstep / (p._tau1 * m.alpha[t]))) + 
                                m.RES1[t-1] * exp(-p._tstep / (p._tau1 * m.alpha[t]))) if t > 1 else Constraint.Skip #NEW
    m._RES1LOM = Constraint(m.time_periods, rule=RES1LOM)
    
    def RES2LOM(m,t): # note initial condition
        return m.RES2[t] == (p._emshare2 * p._tau2 * m.alpha[t] * (m.ECO2[t] / c_to_co2) * (1 - exp(-p._tstep / (p._tau2 * m.alpha[t]))) + 
                                m.RES2[t-1] * exp(-p._tstep / (p._tau2 * m.alpha[t]))) if t > 1 else Constraint.Skip #NEW
    m._RES2LOM = Constraint(m.time_periods, rule=RES2LOM)
    
    def RES3LOM(m,t): # note initial condition
        return m.RES3[t] == (p._emshare3 * p._tau3 * m.alpha[t] * (m.ECO2[t] / c_to_co2) * (1 - exp(-p._tstep / (p._tau3 * m.alpha[t]))) + 
                                m.RES3[t-1] * exp(-p._tstep / (p._tau3 * m.alpha[t]))) if t > 1 else Constraint.Skip #NEW
    m._RES3LOM = Constraint(m.time_periods, rule=RES3LOM)
    
    def MATEQ(m,t): # note initial condition
        return m.MAT[t] == p._mateq + m.RES0[t] + m.RES1[t] + m.RES2[t] + m.RES3[t] if t > 1 else Constraint.Skip #NEW
    m._MATEQ = Constraint(m.time_periods, rule=MATEQ)
    
    def CACCEQ(m,t):
        return m.CACC[t] == (m.CCATOT[t] - (m.MAT[t] - p._mateq))
    m._CACCEQ = Constraint(m.time_periods, rule=CACCEQ)
    
    def FORCEQ(m,t):
        return m.FORC[t] == p._fco22x * ((log(m.MAT[t] / p._mateq)) / log(2)) + m.F_Misc[t] + m.F_GHGabate[t]
    m._FORCEQ = Constraint(m.time_periods, rule=FORCEQ)
    
    def TBOX1EQ(m,t): # note initial condition
        return m.TBOX1[t] == (m.TBOX1[t-1] * exp(-p._tstep / p._d1)) + (p._teq1 * m.FORC[t] * (1 - exp(-p._tstep / p._d1))) if t > 1 else Constraint.Skip #NEW
    m._TBOX1EQ = Constraint(m.time_periods, rule=TBOX1EQ)
    
    def TBOX2EQ(m,t): # note initial condition
        return m.TBOX2[t] == (m.TBOX2[t-1] * exp(-p._tstep / p._d2)) + (p._teq2 * m.FORC[t] * (1 - exp(-p._tstep / p._d2))) if t > 1 else Constraint.Skip #NEW
    m._TBOX2EQ = Constraint(m.time_periods, rule=TBOX2EQ)
    
    def TATMEQ(m,t): # note initial condition
        return m.TATM[t] == m.TBOX1[t] + m.TBOX2[t] if t > 1 else Constraint.Skip #NEW
    m._TATMEQ = Constraint(m.time_periods, rule=TATMEQ)
    
    def irfeqlhs(m,t):
        return m.IRFt[t] == ((m.alpha[t] * p._emshare0 * p._tau0 * (1 - exp(-100 / (m.alpha[t] * p._tau0)))) +
                             (m.alpha[t] * p._emshare1 * p._tau1 * (1 - exp(-100 / (m.alpha[t] * p._tau1)))) +
                             (m.alpha[t] * p._emshare2 * p._tau2 * (1 - exp(-100 / (m.alpha[t] * p._tau2)))) +
                             (m.alpha[t] * p._emshare3 * p._tau3 * (1 - exp(-100 / (m.alpha[t] * p._tau3))))) #NEW
    m._irfeqlhs = Constraint(m.time_periods, rule=irfeqlhs)
    
    def irfeqrhs(m,t):
        return m.IRFt[t] == p._irf0 + (p._irC * m.CACC[t]) + (p._irT * m.TATM[t]) #NEW
    m._irfeqrhs = Constraint(m.time_periods, rule=irfeqrhs)



    ## NonCO2 Forcings Equations
    
    def elandEQ(m,t):
        return m.eland[t] == p._eland0 * (1 - p._deland)**(t-1)
    m._elandEQ = Constraint(m.time_periods, rule=elandEQ)
    
    def CO2E_GHGabateBEQ(m,t):
        return m.CO2E_GHGabateB[t] == p._ECO2eGHGB2020 + ((p._ECO2eGHGB2100 - p._ECO2eGHGB2020) / 16) * (t-1) if t <= 16 else m.CO2E_GHGabateB[t] == p._ECO2eGHGB2100
    m._CO2E_GHGabateBEQ = Constraint(m.time_periods, rule=CO2E_GHGabateBEQ)
    
    def F_MiscEQ(m,t):
        return m.F_Misc[t] == p._F_Misc2020 + ((p._F_Misc2100 - p._F_Misc2020) / 16) * (t-1) if t <= 16 else m.F_Misc[t] == p._F_Misc2100
    m._F_MiscEQ = Constraint(m.time_periods, rule=F_MiscEQ)
    
    def emissratEQ(m,t):
        return m.emissrat[t] == p._emissrat2020 + ((p._emissrat2100 - p._emissrat2020) / 16) * (t-1) if t <= 16 else m.emissrat[t] == p._emissrat2100
    m._emissratEQ = Constraint(m.time_periods, rule=emissratEQ)
    
    def sigmatotEQ(m,t):
        return m.sigmatot[t] == m.sigma[t] * m.emissrat[t]
    m._sigmatotEQ = Constraint(m.time_periods, rule=sigmatotEQ)
    
    def cost1totEQ(m,t):
        return m.cost1tot[t] == m.pbacktime[t] * m.sigmatot[t] / p._expcost2 / 1000
    m._cost1totEQ = Constraint(m.time_periods, rule=cost1totEQ)
    




    ###########################################################################
    # Objective Function:
    ###########################################################################
    
    def PERIODUEQ(m,t):
        return m.PERIODU[t] == ((((m.C[t] * 1000 / m.L[t])**(1 - p._elasmu)) - 1) / (1 - p._elasmu)) - 1
    m._PERIODUEQ = Constraint(m.time_periods, rule=PERIODUEQ)
    
    def TOTPERIODUEQ(m,t):
        return m.TOTPERIODU[t] == m.PERIODU[t] * m.L[t] * m.rr[t]
    m._TOTPERIODUEQ = Constraint(m.time_periods, rule=TOTPERIODUEQ)
    
    def UTILITYEQ(m):
        return m.UTILITY == p._tstep * p._scale1 * summation(m.TOTPERIODU) + p._scale2
    m._UTILITYEQ = Constraint(rule=UTILITYEQ)
    
    ###########################################################################
    
    def obj_rule(m):
        return m.UTILITY
    m.obj = Objective(rule=obj_rule, sense=maximize)
    
    



    ###########################################################################
    # Optimization
    ###########################################################################
    
    opt = SolverFactory(solver_name)

    start = time.time()
    
    results = opt.solve(m, tee=True)
    
    end = time.time()
    
    print("Time Elapsed:", end - start)





    ###########################################################################
    # Post-Processing
    ###########################################################################

    # print("Dumping graphs to file.")
    # output = simulateDynamics([value(m.MIU[t]) for t in m.time_periods], *args) #fix
    # dumpState(years, output, "./results/base_case_state_post_opt.csv") #fix

    # fix
    # Put these below equations and dumpState in separate py file to call as functions
    # ### Post-Solution Parameter-Assignment ###
    # m.scc[t]        = -1000 * m.ECO2[t] / (.00001 + m.C[t]) # NOTE: THESE (m.eco2[t] and m.C[t]) NEED TO BE MARGINAL VALUES, NOT THE SOLUTIONS THEMSELVES
    # m.ppm[t]        = m.MAT[t] / 2.13
    # m.abaterat[t]   = m.ABATECOST[t] / m.Y[t]
    # m.atfrac2020[t] = (m.MAT[t] - p._mat0) / (m.CCATOT[t] + .00001 - p._CumEmiss0)
    # m.atfrac1765[t] = (m.MAT[t] - p._mateq) / (.00001 + m.CCATOT[t])
    # m.FORC_CO2[t]   = p._fco22x * ((log((m.MAT[t] / p._mateq)) / log(2)))

    print("Completed.")