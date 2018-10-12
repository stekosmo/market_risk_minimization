'''models.py
Natalie Wellen
April 11, 2017
This file contains the functions that create the 3 different kinds of models for OTC derivative networks.'''

import random
import scipy.stats
import numpy as np
#import rpy2 as rp
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

#From the data there is a 33% chance the net notional will be positive
#For postive values least squares estimates a value of lamP = 28.
#For negative values lamN = 15.3


'''basicBilateralNetwork, models a bilateral trading network using the basic model created by 
		Gandy and Veraart in cran/systemicrisk
	This function uses helper functions from notionalDist.py and expoDistribution.py
	input gross_notionals, the OTC class gross_notional vector, where each entry in the vector is a different bank
	input p, the probability of an edge existing between two banks, can be a scalar or matrix
	input lam, the distribution of trading values, can be a scalar or matrix
	input net_p, the probability a net notional will be positive; default=0.33
	input net_lamP, lambda of positive side exponential dist that samples net magnitude; default=28
	input net_lamN, lambda of negative side exponential dist that samples net magnitude; default=15.3
	input nsamples, the number of samples to be returned in a chain
	input thin, the thinning to occur during Gibbs Sampling
	input burnin, the length of burnin for Gibbs Sampling
	input beta, the illiquidity measure of the OTC asset class being modeled
	output rbilateral, the rlist of rmatrices representing bilateral trading between banks
'''
def basicBilateralNetwork(gross_notionals, beta, p, lam, net_p = .33, net_lamP = 28., net_lamN = 15.3, nsamples = 5, thin = 1000000, burnin = 10000, distribution = np.random.normal()):
	n = np.size(gross_notionals) #number of banks
	
	#split the notional values into assets and liabilities
	#first the gross notionals are converted into net notional values
	net_notionals = netNotional(gross_notionals, n, net_p, net_lamP, net_lamN)
	#then the asset and liabilites of each bank are computed
	a, l = aAndL(gross_notionals, net_notionals)
	
	#next use Gandy and Veraart's methodology
	systemicrisk = importr("systemicrisk")
	#Here we are using the basic/basic tiered model from their paper
	rsampleERE = robjects.r['sample_ERE']
	rbilateral = rsampleERE(robjects.FloatVector(l), robjects.FloatVector(a), p, lam, nsamples, thin, burnin)

	#convert net values to exposure values
	bilateral = [0]* nsamples
	for i in range(0, nsamples):
		bilateral[i] = symmetricExD(np.array(rbilateral[i]), gross_notionals, beta)
	
	return bilateral


'''hierarchicalBilateralNetwork, models a bilateral trading network using the conjugate distribution model created by Gandy and Veraart in cran/systemicrisk
	input gross_notionals, the OTC class gross_notional vector, where each entry in the vector is a different bank
	input illiquid, the illiquidity measure of the OTC asset class being modeled
	input beta1, first parameter of a beta dist., default = 1 (decides if two banks trade)
	input beta2, first parameter of a beta dist., default = 1 (decides if two banks trade)
	input gamma, first parameter of a gamma dist., default = 1 (decides how much is traded)
	input gamma_scale, scale paramer for gamma dist., default = 1 (decides how much is traded)
	input net_p, the probability a net notional will be positive; default=0.33
	input net_lamP, lambda of positive side exponential dist that samples net magnitude; default=28
	input net_lamN, lambda of negative side exponential dist that samples net magnitude; default=15.3
	input nsamples, the number of samples to be returned in a chain
	input thin, the thinning to occur during Gibbs Sampling
	input burnin, the length of burnin for Gibbs Sampling
	output rbilateral, the rlist of rmatrices representing bilateral trading between banks
'''
def hierarchicalBilateralNetwork(gross_notionals, illiquid, beta1 = 1, beta2 = 1, gamma = 1, gamma_scale = 1, net_p = .33, net_lamP = 28., net_lamN = 15.3, nsamples = 5, thin = 20, burnin = 10):
	n = np.size(gross_notionals) #number of banks
	
	#split the notional values into assets and liabilities
	#first the gross notionals are converted into net notional values
	net_notionals = netNotional(gross_notionals, n, net_p, net_lamP, net_lamN)
	#then the asset and liabilites of each bank are computed
	a, l = aAndL(gross_notionals, net_notionals)

	#next use Gandy and Veraart's methodology
	systemicrisk = importr("systemicrisk")
	#Here we are using the non-tiered conjugate distribution model 
	rsampleHM = robjects.r['sample_HierarchicalModel']
	rmodel = robjects.r['Model.Indep.p.lambda']
	rbeta = robjects.r['Model.p.BetaPrior']
	rgamma = robjects.r['Model.lambda.GammaPrior']
	rchoosethin = robjects.r['choosethin']

	rl = robjects.FloatVector(l)
	ra = robjects.FloatVector(a)
	m = rmodel(rbeta(n, beta1, beta2), rgamma(n, gamma, gamma_scale))
	rbilateral = rsampleHM(rl, ra, m, nsamples, thin=rchoosethin(rl, ra, model = m, silent = True))

	#convert net values to exposure values
	bilateral = [0]* nsamples
	for i in range(0, nsamples):
		bilateral[i] = symmetricExD(np.array(rbilateral[i]), gross_notionals, beta)
	
	return bilateral


'''ccpNetwork, creates a model of exposure values traded through a CCP for an OTC derivative market
	input bilateral_chain, a list of matrices that represents exposures from bilateral trades
	output ccp_chain, list of CCP trading networks each represented by a matrix
'''
def ccpNetwork(bilateral_chain):
	#number of samples we in the chain
	nsamples = len(bilateral_chain)
	#the number of banks in the network
	hold = bilateral_chain[0]
	n = np.size(hold[0])
	
	ccp_chain = [0]*nsamples
	#create the corresponding CCP network for each bilateral network
	for i in range(0, nsamples):
		#the asset and liabilites of each bank are computed
		l = np.sum(bilateral_chain[i], axis=1) #finds row sums
		a = np.sum(bilateral_chain[i], axis=0) #finds column sums
		#finally we store these values into the trading network, where the ccp
			#is the zeroth row and column of the matrix
		ccp = np.zeros((n+1, n+1))
		for j in range(1, n+1):
			ccp[0, j] = a[j-1]
			ccp[j, 0] = l[j-1]
		ccp_chain[i] = ccp
	return ccp_chain


'''multiCcpNetwork, handles the case where multiple CCPs trade int the same derivative class
	input num, the number of ccps trading in the class, num>1
	input chain, the chain of ccp networks modeled already for the gross_notional data
	input distribution of CCPs in the given market
	output multi_chain, the trading networks modeled with number of ccps = num
'''
def multiCcpNetwork(num, chain, distribution):
	nsamples = len(chain)
	multi_chain = [0]*nsamples
	hold = chain[0]
	n = np.size(hold[0])-1

	#define distributions for how derivatives are traded 
	ratio = distribution
	#For each sample a new model with multiple CCPs is formed
	for i in range(0, nsamples):
		ccps_left = np.zeros((n+num, n+num)) #new matrix
		exposures = chain[i]
		exposures = np.delete(exposures[0], [0]) #total bank exposure in the network
		for j in range(0, n):
			trades = np.round(np.multiply(exposures[j], -1*ratio[j]), 2)
			ccps_left[num+j, 0:num] = trades
		ccps_top = -1*np.transpose(ccps_left)
		multi_chain[i] = ccps_top+ccps_left
    
	return multi_chain


'''doubleE, samples a value for the net to gross ratio of CDS
	by using and expoential distribution that is uniformly positive or negative valued
	input n, the number of samples to take
	input p, the probability a value is positive
	input lamP, the lambda of the exponential distribution if the value is positive
	input lamN, the lambda of the exponential distribution if the value is positive
	output ratio, a vector of net/gross notional values for n banks
	'''
def doubleE(n, p, lamP, lamN):
	#create an exponential distribution in [0, 1]
	XP = scipy.stats.truncexpon(b = lamP, loc = 0, scale = 1./lamP)
	XN = scipy.stats.truncexpon(b = lamN, loc = 0, scale = 1./lamN)
	ratio = [0]*n
	for i in range(0, n):
		if(random.random()< p):
			ratio[i] = XP.rvs(1)[0]
		else:
			ratio[i] = -1*XN.rvs(1)[0]
	return ratio 

'''netNotional, converts a gross notional vector into a net notional vector
	input gross_notionals, the gross notional vector
	input n, the number of banks we are working with
	input lamb, the mean of the distribution used to estimate net notionals
	output net_notionals, the new vector that contains net notional values
'''
def netNotional(gross_notionals, n, p, lamP, lamN):
	net_notionals = np.multiply(doubleE(n, p, lamP, lamN), gross_notionals)
	return net_notionals

'''aAndL, finds the asset and liability values for each bank from the gross and net notional vals
	note that the last value is changed to sum(a) = sum(l)
	input gross_notionals, the vector of gross notional derivative values
	input net_notionals, the estimated vector of net notionals
	output a, assets are the amount each bank is owed from derivative contracts
	output l, liabilities are the amount each bank owes from derivative contracts
'''
def aAndL(gross_notionals, net_notionals):
	a = np.divide(np.add(gross_notionals, net_notionals),2)
	l = np.subtract(gross_notionals, a)
	#round to the value of cents
	L = np.sum(l)
	change = (np.sum(a)-L) #how much is needed for sum(a)=sum(l)
	l = l*(1 + change/L)
	a = np.round(a, 2)
	l = np.round(l, 2)
	return a, l	
	'''
    n = np.size(gross_notionals)
	a = a-change
	l = l+change
	zeroa = 0 #keeps track of the number of 0's in a and l
	zerol = 0
	inda = [] #keeps track of the indicies of zero values in a and l
	indl = []
	i = 0
	while i < n:	#check that we will not get negative values
		if(a[i]<0):
			zeroa = zeroa+1.
			change = -a[i]
			a = a - change/(n-zeroa)
			inda.append(i)
			a[inda] = 0
			i = -1
		if(l[i]<0):
			zerol = zerol+1.
			change = -l[i]
			l = l - change/(n-zerol)
			indl.append(i)
			l[indl] = 0
			i = -1
		i = i+1'''
	


'''symmetricExD, a function that finds exposures based on values in a bilateral trading network
	in other words it estimates exposures directly from the bilateral network
	input bilateral, a matrix of gross notional values traded between any two banks
	input Z, the vector of total gross notional values for each bank in the network (unused)
	input m, the illiquidity measure of the derivative class
	input distribution, the probability distribution function to be sampled from (normal on default)
	output exposure, the estimated exposure values for the network
	'''
def symmetricExD(bilateral, Z, m, distribution = np.random.normal(0,1)):
	#create matrix of standard deviation to be sampled from
	n = Z.size
	frac = m*bilateral
	#check that bilateral and Z have correct dimensions
	m1 = np.size(bilateral[0])
	m2 = np.size(np.transpose(bilateral)[0])
	if(not (n == m1 == m2)):
		print ("The gross notional vector and matrix do not have matching dimensions!")
		return 							
	#take samples from distribution
	sample = np.zeros((n,n))
	for i in range(0,n):
		for j in range(0,n):
			if(i == j):
				pass
			else:
				sample[i,j] = np.round(frac[i,j]*distribution,2)
	#create final antisymmetric exposure matrix
	exposure = np.zeros((n,n))
	for i in range(0,n):
		for j in range(i+1,n):
			if(sample[i,j]!=0):
				if(sample[j,i]!=0):
					exposure[i,j] = (1/np.sqrt(2))*(sample[i,j]-sample[j,i])
					exposure[j,i] = -1*exposure[i,j]
				else:
					exposure[i,j] = sample[i,j]
					exposure[j,i] = -1*exposure[i,j]
			if(sample[j,i]!=0):
				exposure[j,i] = sample[j,i]
				exposure[i,j] = -1*exposure[j,i]
	return exposure
