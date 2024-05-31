# computation of the exogenous variables
params['l'] = getPopulation(params['numPeriods'],params['pop0'],params['popasym'],params['popadj']) 
params['ga'] = getGrowthRateOfProductivity(params['ga0'],params['dela'],params['numPeriods'],params['tstep'])
params['al'] = getLevelOfProductivity(params['a0'],params['ga'],params['numPeriods'])
params['gsig'] = getCumulativeEfficiencyImprovement(params['gsigma1'],params['dsig'],params['tstep'],params['numPeriods'])
params['sigma'] = getGrowthRate(params['sig0'],params['gsig'],params['tstep'],params['numPeriods'])
params['pbacktime'] = getBackstopPrice(params['pback'],params['gback'],params['numPeriods'])
params['cost1'] = getAdjustedCostForBackstop(params['pbacktime'],params['sigma'],params['expcost2'],params['numPeriods'])
params['etree'] = getEmissionsFromDeforestation(params['eland0'],params['deland'],params['numPeriods'])
params['cumetree'] = getCumulativeEmissionsFromLand(params['etree'], params['numPeriods'])
params['rr'] = getAverageUtilitySocialDiscountRate(params['prstp'],params['tstep'],params['numPeriods'])
params['forcoth'] = getExogenousForcingOfOtherGreenhouseGases(params['fex0'],params['fex1'],params['numPeriods'])
params['cpricebase'] = getCarbonPrice(params['cprice0'],params['gcprice'],params['tstep'],params['numPeriods'])

def diceModel2016(**kwargs):
    model = pe.ConcreteModel("Dice2016R") # Dice2016 is a PyOmo concrete model
    ###SETS AND INDICES###
    model.time_periods = pe.Set( initialize=kwargs['rr'].keys()) # t = 1 .. numPeriods
    ###VARIABLES### 
    ## Boundaries
    # Control rate limit
    def miuBounds(model,t):
        if (t == 1) and (kwargs['ifopt'] == 1):    # if optimal control on emissions
            return (kwargs['miu0'],kwargs['miu0']) # initial value
        elif t < 30:
            return (0.,1.)
        else:
            return (0.,kwargs['limmiu']) # upper limit on control rate after 2150
    # upper and lower bounds for stability    
    def tatmBounds(model,t):
        if t == 1: 
            return (kwargs['tatm0'],kwargs['tatm0'])  # initial value
        else:
            return (0.,20.)  # range for increase temperature in atmosphere
    def toceanBounds(model,t):
        if t == 1: 
            return (kwargs['tocean0'],kwargs['tocean0']) # initial value
        else:
            return (-1.,20.) # range for increase temperature in oceans

    def matBounds(model,t):
        if t == 1: 
            return (kwargs['mat0'],kwargs['mat0']) # initial value
        else:
            return (10.,np.inf) # range for carbon concentration increase in atmosphere

    def muBounds(model,t):
        if t == 1: 
            return (kwargs['mu0'],kwargs['mu0']) # initial value
        else:
            return (100.,np.inf) # range for carbon concentration increase in shallow oceans

    def mlBounds(model,t):
        if t == 1: 
            return (kwargs['ml0'],kwargs['ml0'])  # initial value
        else:
            return (1000.,np.inf) # range for carbon concentration in lower oceans

    def cBounds(model,t):
        return (2.,np.inf) # lower bound for consumption

    def kBounds(model,t):
        if t == 1: 
            return (kwargs['k0'],kwargs['k0']) # initial value
        else:
            return (1.,np.inf) # lower bound for capital stock

    def cpcBounds(model,t):
        return (.01,np.inf) # lower bound for per capita consumption

    # control variable
    def sBounds(model,t):
        if t <= kwargs['numPeriods'] - 10:
            return (-np.inf,np.inf)
        else:
            return (kwargs['optlrsav'],kwargs['optlrsav']) # constraint saving rate at the end 

    # Ressource limit
    def ccaBounds(model,t):
        if t == 1:
            return (kwargs['cca0'],kwargs['cca0']) # initial value
        else:
            return (-np.inf,kwargs['fosslim']) # maximum cumulative extraction fossil fuels

    ## Declarations    
        
    #Emission control rate GHGs    
    model.MIU = pe.Var(model.time_periods,domain=pe.NonNegativeReals,bounds=miuBounds) 
    
    # Increase in radiative forcing (watts per m2 from 1900)
    model.FORC = pe.Var(model.time_periods,domain=pe.Reals) 
    
    # Increase temperature of atmosphere (degrees C from 1900)
    model.TATM = pe.Var(model.time_periods,domain=pe.NonNegativeReals,bounds=tatmBounds) 
    
    # Increase temperatureof lower oceans (degrees C from 1900)
    model.TOCEAN = pe.Var(model.time_periods,domain=pe.Reals,bounds=toceanBounds) 

    # Carbon concentration increase in atmosphere (GtC from 1750)
    model.MAT = pe.Var(model.time_periods,domain=pe.NonNegativeReals,bounds=matBounds) 

    # Carbon concentration increase in shallow oceans (GtC from 1750)
    model.MU = pe.Var(model.time_periods,domain=pe.NonNegativeReals,bounds=muBounds) 

    # Carbon concentration increase in lower oceans (GtC from 1750)
    model.ML = pe.Var(model.time_periods,domain=pe.NonNegativeReals,bounds=mlBounds) 

    # Total CO2 emissions (GtCO2 per year)
    model.E = pe.Var(model.time_periods,domain=pe.Reals) 
    
    #Industrial emissions (GtCO2 per year)
    model.EIND = pe.Var(model.time_periods,domain=pe.Reals) 

    # Consumption (trillions 2005 US dollars per year)
    model.C = pe.Var(model.time_periods,domain=pe.NonNegativeReals,bounds=cBounds) 

    # Capital stock (trillions 2005 US dollars)
    model.K = pe.Var(model.time_periods,domain=pe.NonNegativeReals,bounds=kBounds) 

    # Per capita consumption (thousands 2005 USD per year)
    model.CPC = pe.Var(model.time_periods,domain=pe.NonNegativeReals,bounds=cpcBounds) 

    # Investment (trillions 2005 USD per year)
    model.I = pe.Var(model.time_periods,domain=pe.NonNegativeReals) 

    # Gross savings rate as fraction of gross world product
    model.S = pe.Var(model.time_periods,domain=pe.Reals,bounds=sBounds) 

    # Real interest rate (per annum)
    model.RI = pe.Var(model.time_periods,domain=pe.Reals) 

    # Gross world product net of abatement and damages (trillions 2005 USD per year)
    model.Y = pe.Var(model.time_periods,domain=pe.NonNegativeReals) 

    # Gross world product GROSS of abatement and damages (trillions 2005 USD per year)
    model.YGROSS = pe.Var(model.time_periods,domain=pe.NonNegativeReals) 

    # Output net of damages equation (trillions 2005 USD per year)
    model.YNET = pe.Var(model.time_periods,domain=pe.Reals) 

    # Damages (trillions 2005 USD per year)
    model.DAMAGES = pe.Var(model.time_periods,domain=pe.Reals) 

    # Damages as fraction of gross output
    model.DAMFRAC = pe.Var(model.time_periods,domain=pe.Reals) 

    # Cost of emissions reductions  (trillions 2005 USD per year)
    model.ABATECOST = pe.Var(model.time_periods,domain=pe.Reals) 

    # Marginal cost of abatement (2005$ per ton CO2)
    model.MCABATE = pe.Var(model.time_periods,domain=pe.Reals) 

    # Cumulative industrial carbon emissions (GTC)
    model.CCA = pe.Var(model.time_periods,domain=pe.Reals,bounds=ccaBounds) 
    
    # Total carbon emissions (GtC)
    model.CCATOT = pe.Var(model.time_periods,domain=pe.Reals)

    # One period utility function
    model.PERIODU = pe.Var(model.time_periods,domain=pe.Reals) 

    # Carbon price (2005$ per ton of CO2) !!! No initial upper bound in version 2016R
    model.CPRICE = pe.Var(model.time_periods,domain=pe.Reals) 

    # Period utility
    model.CEMUTOTPER = pe.Var(model.time_periods,domain=pe.Reals) 

    # Welfare function
    model.UTILITY = pe.Var(domain=pe.Reals) 

    ### Equations
    
    ## Emissions and damages

    # eeq(t)
    def emissionsEquation(model,t):
        return (model.E[t] == model.EIND[t] + kwargs['etree'][t])
    
    model.emissionsEquation = pe.Constraint(model.time_periods,rule=emissionsEquation)

    # eindeq(t)
    def industrialEmissions(model,t): 
        # EIND(t) =E= sigma(t) * YGROSS(t) * (1-(MIU(t)));
        return (model.EIND[t] == kwargs['sigma'][t] * model.YGROSS[t] * (1 - model.MIU[t]))

    model.industrialEmissions = pe.Constraint(model.time_periods,rule=industrialEmissions)

    # ccacca(t+1)
    def cumCarbonEmissions(model,t):
        if t == 1:
            return pe.Constraint.Skip # to use boundary value instead of  constraint
        else:
            return (model.CCA[t] == model.CCA[t-1] + model.EIND[t-1] * kwargs['tstep'] / 3.666)

    model.cumCarbonEmissions = pe.Constraint(model.time_periods,rule=cumCarbonEmissions)

    # ccatoteq(t)
    def totCarbonEmissions(model,t):
        return (model.CCATOT[t] == model.CCA[t] + kwargs['cumetree'][t])
    
    model.totCarbonEmissions = pe.Constraint(model.time_periods,rule=totCarbonEmissions)
    
    # force(t)
    def radiativeForcing(model,t): 
        return (model.FORC[t] == kwargs['fco22x'] * (pe.log10(model.MAT[t]/588.0)/pe.log10(2)) + kwargs['forcoth'][t])

    model.radiativeForcing = pe.Constraint(model.time_periods,rule=radiativeForcing)

    # damfraceq(t)
    def damageFraction(model,t):
        return (model.DAMFRAC[t] == (kwargs['a1'] * model.TATM[t]) + (kwargs['a2'] * model.TATM[t]**kwargs['a3']))

    model.damageFraction = pe.Constraint(model.time_periods,rule=damageFraction)

    # dameq(t)
    def damagesConst(model,t):
        return (model.DAMAGES[t] == (model.YGROSS[t] * model.DAMFRAC[t]))

    model.damagesConst = pe.Constraint(model.time_periods,rule=damagesConst)

    # abateeq(t)
    def abatementCost(model,t):
        return (model.ABATECOST[t] == model.YGROSS[t] * kwargs['cost1'][t] * (model.MIU[t]**kwargs['expcost2']))

    model.abatementCost = pe.Constraint(model.time_periods,rule=abatementCost)

    # mcabateeq(t)
    def mcAbatement(model,t): 
  
        myExp = kwargs['expcost2'] - 1
        return (model.MCABATE[t] == kwargs['pbacktime'][t] * (model.MIU[t])**myExp)
          
    model.mcAbatement = pe.Constraint(model.time_periods,rule=mcAbatement)

    # carbpriceeq(t)            
    def carbonPriceEq(model,t): 
        myExp = kwargs['expcost2'] - 1
        return (model.CPRICE[t] == kwargs['pbacktime'][t] * (model.MIU[t])**myExp)

    model.carbonPriceEq = pe.Constraint(model.time_periods,rule=carbonPriceEq)

    ## Climate and carbon cycle
                
    # mmat(t+1)
    def atmosphericConcentration(model,t):
        if t == 1:
            return pe.Constraint.Skip
        else: 
            #MAT(t+1)       =E= MAT(t)*b11 + MU(t)*b21 + (E(t)*(tstep/3.666));
            return (model.MAT[t] == model.MAT[t-1]*kwargs['b11'] + model.MU[t-1] * kwargs['b21'] + model.E[t-1] * kwargs['tstep'] / 3.666)

    model.atmosphericConcentration = pe.Constraint(model.time_periods,rule=atmosphericConcentration)

    # mml(t+1)            
    def lowerOceanConcentration(model,t):
        if t == 1:
            return pe.Constraint.Skip
        else:
            return (model.ML[t] == model.ML[t-1]*kwargs['b33'] + model.MU[t-1] * kwargs['b23'] )

    model.lowerOceanConcentration = pe.Constraint(model.time_periods,rule=lowerOceanConcentration)

    # mmu(t+1)
    def upperOceanConcentration(model,t):
        if t == 1:
            return pe.Constraint.Skip
        else:
            return (model.MU[t] == model.ML[t-1]*kwargs['b32'] + model.MU[t-1] * kwargs['b22'] + model.MAT[t-1] * kwargs['b12'])

    model.upperOceanConcentration = pe.Constraint(model.time_periods,rule=upperOceanConcentration)

    # tatmeq(t+1)
    def atmosphericTemperature(model,t):
        if t == 1:
            return pe.Constraint.Skip
        else: 
            return (model.TATM[t] == model.TATM[t-1] + kwargs['c1'] * ((model.FORC[t] - (kwargs['fco22x']/kwargs['t2xco2'])*model.TATM[t-1]) \
                                                                       - (kwargs['c3'] * (model.TATM[t-1] - model.TOCEAN[t-1]))))

    model.atmosphericTemperature = pe.Constraint(model.time_periods,rule=atmosphericTemperature)

    # toceaneq(t+1)            
    def oceanTemperature(model,t):
        if t == 1:
            return pe.Constraint.Skip
        else:
            return (model.TOCEAN[t] == model.TOCEAN[t-1] + kwargs['c4'] * (model.TATM[t-1] - model.TOCEAN[t-1]))

    model.oceanTemperature = pe.Constraint(model.time_periods,rule=oceanTemperature)

    ## Economic variables            
    
    # ygrosseq(t)
    def grossOutput(model,t):
        # YGROSS(t) =E= (al(t)*(L(t)/1000)**(1-GAMA))*(K(t)**GAMA);
        coeff = kwargs['al'][t]*(kwargs['l'][t]/1000)**(1-kwargs['gama'])
        return (model.YGROSS[t] == coeff*(model.K[t]**kwargs['gama']))

    model.grossOutput = pe.Constraint(model.time_periods,rule=grossOutput)

    # yneteq(t)
    def netOutput(model,t):
        return (model.YNET[t] == model.YGROSS[t] * (1-model.DAMFRAC[t]))

    model.netOutput = pe.Constraint(model.time_periods,rule=netOutput)

    # yy(t)
    def outputNetEqn(model,t):
        return (model.Y[t] == model.YNET[t] - model.ABATECOST[t])

    model.outputNetEqn = pe.Constraint(model.time_periods,rule=outputNetEqn)

    # cc(t)
    def consumptionEqn(model,t):
        return (model.C[t] == model.Y[t] - model.I[t])

    model.consumptionEqn = pe.Constraint(model.time_periods,rule=consumptionEqn)


    # cpce(t)
    def perCapitaConsumption(model,t):
        return (model.CPC[t] == 1000 * model.C[t] / kwargs['l'][t])
                
    model.perCapitaConsumption = pe.Constraint(model.time_periods,rule=perCapitaConsumption)
    # seq(t)
    def savingsRate(model,t):
        return (model.I[t] ==  model.S[t] * model.Y[t])        
    model.savingsRate = pe.Constraint(model.time_periods,rule=savingsRate)
    # kk(t+1)
    def capitalBalance(model,t):
        if t == 1:
            return pe.Constraint.Skip # initial value defined by the boundary
        else: 
            #K(t+1)   =L= (1-dk)**tstep * K(t) + tstep * I(t);
            return (model.K[t] == (1 - kwargs['dk'])**kwargs['tstep'] * model.K[t-1] + kwargs['tstep'] * model.I[t-1])
    model.capitalBalance = pe.Constraint(model.time_periods,rule=capitalBalance)
    # rieq(t)             
    def interestRateEqn(model,t):
        if t == 1:
            return pe.Constraint.Skip # initial value defined by the boundary
        else:
            return (model.RI[t] == (1 + kwargs['prstp']) * (model.CPC[t]/model.CPC[t-1])** (kwargs['elasmu']/kwargs['tstep']) - 1)
    model.interestRateEqn = pe.Constraint(model.time_periods,rule=interestRateEqn)
    # cemutotpereq(t)
    def periodUtilityEqn(model,t):
        return (model.CEMUTOTPER[t] ==  model.PERIODU[t] * kwargs['l'][t] * kwargs['rr'][t])
    model.periodUtilityEqn = pe.Constraint(model.time_periods,rule=periodUtilityEqn)
    # periodueq(t)
    def instUtilityEqn(model,t):
        return (model.PERIODU[t] ==  ((model.C[t] * 1000 / kwargs['l'][t])**(1-kwargs['elasmu'])-1) / (1-kwargs['elasmu']) - 1)
    model.instUtilityEqn = pe.Constraint(model.time_periods,rule=instUtilityEqn)
    # util 
    def utilityCalc(model):
        return (model.UTILITY == kwargs['tstep'] * kwargs['scale1'] * pe.summation(model.CEMUTOTPER) + kwargs['scale2'])
    model.utilityCalc = pe.Constraint(rule=utilityCalc)
    ### objective function
    def obj_rule(model):
        return  model.UTILITY       
    model.OBJ = pe.Objective(rule=obj_rule, sense=pe.maximize)
    return model