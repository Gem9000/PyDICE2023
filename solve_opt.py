'''
Authors: Jacob Wessel, George Moraites
'''

from pyomo.environ import Var, ConcreteModel, RangeSet, Objective, value, Reals, NonNegativeReals
from pyomo.environ import *
from pyomo.opt import SolverFactory

import numpy as np
import time

from DICE_params import modelParams
from dice_dynamics import simulateDynamics, dumpState

if __name__ == '__main__':
    num_times = 82
    tstep = 5
    t = np.arange(1, num_times+1)

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
    model.t = RangeSet(1, num_times)

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
    
    model.L          = Var(model.t, domain=NonNegativeReals, bounds=LBounds)                  #level of population and labor
    model.aL         = Var(model.t, domain=NonNegativeReals, bounds=aLBounds)                 #level of productivity
    model.sigma      = Var(model.t, domain=NonNegativeReals, bounds=sigmaBounds)              #CO2-emissions output ratio
    model.sigmatot   = Var(model.t, domain=NonNegativeReals)                                  #GHG-output ratio
    model.gA         = Var(model.t, domain=NonNegativeReals, bounds=gABounds)                 #productivity growth rate
    model.gL         = Var(model.t, domain=NonNegativeReals)                                  #labor growth rate
    model.gsig       = Var(model.t, domain=Reals, bounds=gsigBounds)                          #Change in sigma (rate of decarbonization)
    model.eland      = Var(model.t, domain=NonNegativeReals, bounds=elandBounds)              #Emissions from deforestation (GtCO2 per year)
    model.cost1tot   = Var(model.t, domain=NonNegativeReals)                                  #Abatement cost adjusted for backstop and sigma
    model.pbacktime  = Var(model.t, domain=NonNegativeReals)                                  #Backstop price 2019$ per ton CO2
    model.cpricebase = Var(model.t, domain=NonNegativeReals)                                  #Carbon price in base case
    model.gbacktime  = Var(model.t, domain=NonNegativeReals)                                  #Decline rate of backstop price
    # precautionary dynamic parameters
    model.varpcc     = Var(model.t, domain=NonNegativeReals)                                  #Variance of per capita consumption
    model.rprecaut   = Var(model.t, domain=NonNegativeReals)                                  #Precautionary rate of return
    model.rr         = Var(model.t, domain=NonNegativeReals)                                  #STP with precautionary factor
    model.rr1        = Var(model.t, domain=NonNegativeReals)                                  #STP without precautionary factor
    
    # for post-processing:
    model.scc        = Var(model.t, domain=NonNegativeReals)                                  #Social cost of carbon
    model.ppm        = Var(model.t, domain=NonNegativeReals)                                  #Atmospheric concentrations parts per million
    model.atfrac2020 = Var(model.t, domain=NonNegativeReals)                                  #Atmospheric share since 2020
    model.atfrac1765 = Var(model.t, domain=NonNegativeReals)                                  #Atmospheric fraction of emissions since 1765
    model.abaterat   = Var(model.t, domain=NonNegativeReals)                                  #Abatement cost per net output

    # Time-varying parameters in FAIR and Nonco2 modules (defined as variables so they can be endogenized in the future)
    model.CO2E_GHGabateB = Var(model.t, domain=NonNegativeReals)                              #Abateable non-CO2 GHG emissions base
    model.CO2E_GHGabateB = Var(model.t, domain=NonNegativeReals)                              #Abateable non-CO2 GHG emissions base (actual)
    model.F_Misc         = Var(model.t, domain=Reals)                                         #Non-abateable forcings (GHG and other)
    model.emissrat       = Var(model.t, domain=NonNegativeReals)                              #Ratio of CO2e to industrial emissions
    model.FORC_CO2       = Var(model.t, domain=Reals)                                         #CO2 Forcings

    ###########################################################################
    # Variables (capitalized naming convention except for alpha)
    ###########################################################################
    # Nonnegative variables: MIU, TATM, MAT, Y, YNET, YGROSS, C, K, I, RFACTLONG, IRFt, alpha)
    
    model.UTILITY    = Var(domain=Reals)                                                      #utility function to maximize in objective
    model.MIU        = Var(model.t, domain=NonNegativeReals, bounds=MIUBounds)                #emission control rate
    model.C          = Var(model.t, domain=NonNegativeReals, bounds=(params._clo,np.inf))     #consumption (trillions 2019 US dollars per year)
    model.K          = Var(model.t, domain=NonNegativeReals, bounds=KBounds)                  #capital stock (trillions 2019 US dollars)
    model.CPC        = Var(model.t, domain=NonNegativeReals, bounds=(params._cpclo,np.inf))   #per capita consumption
    model.I          = Var(model.t, domain=NonNegativeReals)                                  #Investment (trillions 2019 USD per year)
    model.S          = Var(model.t, domain=Reals, bounds=SBounds)                             #gross savings rate as fraction of gross world product
    model.Y          = Var(model.t, domain=NonNegativeReals)                                  #Gross world product net of abatement and damages (trillions 2019 USD per year)
    model.YGROSS     = Var(model.t, domain=NonNegativeReals)                                  #Gross world product GROSS of abatement and damages (trillions 2019 USD per year)
    model.YNET       = Var(model.t, domain=Reals)                                             #Output net of damages equation (trillions 2019 USD per year)
    model.DAMAGES    = Var(model.t, domain=Reals)                                             #Damages (trillions 2019 USD per year)
    model.DAMFRAC    = Var(model.t, domain=Reals)                                             #Damages as fraction of gross output
    model.ABATECOST  = Var(model.t, domain=Reals)                                             #Cost of emissions reductions (trillions 2019 USD per year)
    model.MCABATE    = Var(model.t, domain=Reals)                                             #Marginal cost of abatement (2019$ per ton CO2)
    model.CCATOT     = Var(model.t, domain=Reals, bounds=CCATOTBounds)                        #Total emissions (GtC)
    model.PERIODU    = Var(model.t, domain=Reals)                                             #One period utility function
    model.CPRICE     = Var(model.t, domain=Reals)                                             #Carbon price (2019$ per ton of CO2)
    model.TOTPERIODU = Var(model.t, domain=Reals)                                             #Period utility
    model.RFACTLONG  = Var(model.t, domain=NonNegativeReals, bounds=RFACTLONGBounds)          #Long interest factor
    model.RSHORT     = Var(model.t, domain=Reals)                                             #Long interest factor
    model.RLONG      = Var(model.t, domain=Reals)                                             #Long interest factor

    # Variables in FAIR and Nonco2 modules
    model.FORC       = Var(model.t, domain=Reals)                                                      #Increase in radiative forcing (watts per m2 from 1765)
    model.TATM       = Var(model.t, domain=NonNegativeReals, bounds=TATMBounds)                        #temperature increase in atmosphere (degrees C from 1765)
    model.TBOX1      = Var(model.t, domain=Reals, bounds=TBOX1Bounds)                                  #Increase temperature of box 1 (degrees C from 1765)
    model.TBOX2      = Var(model.t, domain=Reals, bounds=TBOX2Bounds)                                  #Increase temperature of box 2 (degrees C from 1765)
    model.RES0       = Var(model.t, domain=Reals, bounds=RES0Bounds)                                   #carbon concentration reservoir 0 (GtC from 1765)
    model.RES1       = Var(model.t, domain=Reals, bounds=RES1Bounds)                                   #carbon concentration reservoir 1 (GtC from 1765)
    model.RES2       = Var(model.t, domain=Reals, bounds=RES2Bounds)                                   #carbon concentration reservoir 2 (GtC from 1765)
    model.RES3       = Var(model.t, domain=Reals, bounds=RES3Bounds)                                   #carbon concentration reservoir 3 (GtC from 1765)
    model.MAT        = Var(model.t, domain=NonNegativeReals, bounds=MATBounds)                         #carbon concentration increase in atmosphere (GtC from 1765)
    model.CACC       = Var(model.t, domain=Reals)                                                      #Accumulated carbon in ocean and other sinks (GtC)
    model.IRFt       = Var(model.t, domain=NonNegativeReals)                                           #IRF100 at time t
    model.alpha      = Var(model.t, domain=NonNegativeReals, bounds=(params._alphalo,params._alphaup)) #Carbon decay time scaling factor
    model.ECO2       = Var(model.t, domain=Reals)                                                      #Total CO2 emissions (GtCO2 per year)
    model.ECO2E      = Var(model.t, domain=Reals)                                                      #Total CO2e emissions including abateable nonCO2 GHG (GtCO2 per year)
    model.EIND       = Var(model.t, domain=Reals)                                                      #Industrial CO2 emissions (GtCO2 per yr)
    model.F_GHGabate = Var(model.t, domain=Reals, bounds=F_GHGabateBounds)                             #Forcings of abatable nonCO2 GHG
    


    ###########################################################################
    # Constraints
    ###########################################################################
    
    # def capital_stock_constraint(model,t):
    #self._K[i] <= (1.0 - self.params._dk)**self.params._tstep * self._K[i-1] + self.params._tstep * self._I[i]

    # IRF equation: constrain lhs to equal rhs at all times


    ###########################################################################
    # Objective Function: MAXIMIZE UTILITY = tstep * scale1 * sum(t,  TOTPERIODU(t)) + scale2
    ###########################################################################
    def obj_rule(model):
        return simulateDynamics([value(model.MIU[t]) for t in model.t], *args)
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
    output = simulateDynamics([value(model.MIU[t]) for t in model.t], *args)
    dumpState(years, output, "./results/base_case_state_post_opt.csv")

    # Put these below equations and dumpState in separate py file to call as functions
    # ### Post-Solution Parameter-Assignment ###
    # self._scc[i]        = -1000 * self._eco2[i] / (.00001 + self._C[i]) # NOTE: THESE (self._eco2[i] and self._C[i]) NEED TO BE MARGINAL VALUES, NOT THE SOLUTIONS THEMSELVES
    # self._ppm[i]        = self._mat[i] / 2.13
    # self._abaterat[i]   = self._abatecost[i] / self._Y[i]
    # self._atfrac2020[i] = (self._mat[i] - self.params._mat0) / (self._ccatot[i] + .00001 - self.params._CumEmiss0)
    # self._atfrac1765[i] = (self._mat[i] - self.params._mateq) / (.00001 + self._ccatot[i])
    # self._FORC_CO2[i]   = self.params._fco22x * ((math.log((self._mat[i] / self.params._mateq)) / math.log(2)))

    print("Completed.")