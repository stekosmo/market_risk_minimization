# -*- coding: utf-8 -*-
"""
Single matrix test

- Input: CSV adjacency matrix
- Applies each type of compression, then runs each market through riskModel
- Output: Graph of each market, the loss in each case, and the average loss

@author: Stephen
"""

import getResults as gr
import numpy as np
import warnings

'''Import csv data (adjacency matrix of market)
Replace 'testMatrix.csv' with actual csv file
Can also replace this wityh a line defining a numpy matrix'''

M = np.genfromtxt("presData.csv", delimiter=',')

#To keep graph output clean
warnings.filterwarnings("ignore")

#Now call the results function
gr.simpleResults(M)

'''
print('Bilateral')
gr.simpleResults(Mc)
print('Single CCP')
gr.simpleResults(Mm)
print('Multi CCP')

print(bilateralMin,'\n', CCPmin,'\n', MultiMin)
'''

