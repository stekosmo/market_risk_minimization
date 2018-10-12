"""
Risk test

Tests the risk in gibbs sampled markets. Outputs IRS, FX, and Credit results
one at a time, channge whiich result is shown at the bottom

@author: Stephen
"""
import models
import numpy as np
import getResults as gr

###########################################################################
#Network Models
###########################################################################
'''
Use gibbs sampling to define the chain of bilateral markets
Results taken from natalie's code and data
'''
n = 25 #the length of each gross vector aka number of financial institutions
'''Change this for faster test, or greater accuracy (full test is 1,000)'''
num_loop = 1 #number of times we append new bilateral chains
num_chain = 10*num_loop #the number of bilateral networks returned by Gibbs Sampling

#OTC Interest Rate Swaps in millions of $
irs = np.array([  3.14775554e+07,   2.85344403e+07,   2.89009327e+07,   1.48787464e+07,
   6.31817238e+06,   2.59408419e+06,   1.56768044e+04,   1.87426228e+04,
   3.39909922e+05,   3.47324751e+05,   1.87646372e+05,   1.37080930e+04,
   2.19214487e+05,   1.67973015e+05,   1.43942796e+05,   8.96531234e+04,
   7.21581753e+04,   7.12042522e+04,   6.21986720e+04,   5.18648491e+04,
   6.05128016e+04,   4.25462107e+04,   3.82678111e+04,   3.06402180e+04,
   3.27928908e+04])
illiq_irs = 0.0039
num_ccp_irs = 3 #the number of CCP's in the multiple CCP network
#distribution of exposures to each CCP
dist_irs = np.array([[0.22, 0.76, 0.02],]*(num_chain*4))

#OTC SwapsForeign Exchange derivatives in millions of $
fx = np.array([  9.04425536e+06,   1.00758371e+07,   1.82369522e+06,   3.92494523e+06,
   3.66772329e+05,   9.92189867e+05,   1.38240912e+06,   1.31064484e+06,
   4.88309620e+05,   1.45028536e+04,   5.73156249e+03,   2.65769150e+05,
   4.50847529e+04,   1.03753300e+04,   5.40557710e+03,   5.80279116e+02,
   6.08562924e+03,   1.83752909e+03,   7.99282000e+03,   1.07994447e+04,
   6.11240420e+02,   1.14003780e+03,   1.27559370e+03,   8.79378200e+03,
   2.75864366e+03])
illiq_fx = 0.0039
num_ccp_fx = 1 #the number of CCP's in the multiple CCP network
dist_fx = np.array([[],]*(num_chain*4))

#OTC Credit Derivatives in millions of $
credit = np.array([1.86205257e+06,   1.81695423e+06,   1.54550442e+05,   9.21160614e+05,
   2.76809305e+04,   1.17396511e+05,   0.00000000e+00,   9.37131140e+03,
   0.00000000e+00,   5.94988867e+03,   4.23637227e+03,   0.00000000e+00,
   4.85943444e+03,   5.36655000e+02,   0.00000000e+00,   2.32111646e+03,
   3.16136584e+02,   2.83285734e+03,   2.47050800e+03,   2.75145088e+03,
   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
   1.55416544e+03])
illiq_credit = 0.0098
num_ccp_credit = 2 #the number of CCP's in the multiple CCP network
dist_credit = np.array([[0.08, 0.92],]*(num_chain*4))

#Reserve levels (Consol Assets) in millions of $
#Data taken from https://www.federalreserve.gov/releases/lbr/20161231/default.htm

assets = np.array([2.082803e+06, 1.349581e+06, 1.59116e+05, 1.677490e+06, 1.727235e+06,
              1.97206e+05, 2.39203e+05, 1.29288e+05, 2.57576e+05, 3.56000e+05,
              2.00558e+05, 1.23548e+05, 4.41010e+05, 2.69031e+05, 1.15553e+05,
              2.86080e+05, 1.34362e+05, 1.25042e+05, 1.16940e+05, 1.39776e+05,
              2.14433e+05, 3.2160e+04, 8.3640e+04, 1.13153e+05, 9.9555e+04])
#Distribution of total assets
assets = assets/np.sum(assets)
#As of dec 2017, total assets-liabilities for all us banks
#Data: https://www.federalreserve.gov/releases/h8/current/default.htm
#This is distributed across all above banks
c = assets*((1.67885e+04)-(1.49378e+04))

#create tiered p matrix for modeling OTC trading frequency
#There are two tiers, the top 4 instiutions are the large institution tier
p = np.ones([25,25])*0.2
p[:, 0:4] = 0.8
p[0:4,:] = 0.8
p[0:4, 0:4] = 1
np.fill_diagonal(p, 0)
#create tiered lambda matrix for modeling OTC derivative 
lam = np.ones([25,25])*(1./19400.)
lam[0:4,0:4] = (1./6770000.)
np.fill_diagonal(lam, 0)

'''
- Results can take some time (20-30min for full 10,000 samples),so it is
best to run one test at a time. Change this to match desired test
'''
#append many chains to account for different initial conditions
bilateral_irs = []
bilateral_fx = []
bilateral_credit = []
for i in range(num_loop):
    #Create the models we are going to compare, from irs class
    bilateral_irs.extend(models.basicBilateralNetwork(irs, illiq_irs, p, lam, nsamples=10))
    #Create the models we are going to compare, from fx class
    bilateral_fx.extend(models.basicBilateralNetwork(fx, illiq_fx, p, lam, nsamples=10))
    #Create the models we are going to compare, from credit derivatives class
    bilateral_credit.extend(models.basicBilateralNetwork(credit, illiq_credit, p, lam, nsamples=10, burnin = 100000, distribution = np.random.standard_t(3)))

###########################################################################
#Run tests
########################################k###################################
'''
- Results can take some time (20-30min for full 10,000 samples),so it is
best to run one test at a time
- It also makes keeping track of output easier, as graphs are not labeled for
the type of derivative
'''
print('IRS Results:')
dfi, dfmi = gr.gibbsResults(bilateral_irs, num_chain, num_ccp_irs, dist_irs, c)
print('FX Results:')
dff, dfmf = gr.gibbsResults(bilateral_fx, num_chain, num_ccp_fx, dist_fx, c)
print('Credit Results:')
dfc, dfmc = gr.gibbsResults(bilateral_credit, num_chain, num_ccp_credit, dist_credit, c)

gr.graph_results(dfi,dfmi, num_chain, "IRS Derivatives")
gr.graph_results(dff,dfmf, num_chain, "FX Derivatives")
gr.graph_results(dfc,dfmc, num_chain, "Credit Derivatives")