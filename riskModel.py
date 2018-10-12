"""
Model for risk propagation in financial market

Input: M - numpy adjacency matrix of a market
       c - reserve levels
       num - the number of CCPs in a market (default is 0)
Output: The average value lost after modeling default propagation starting
        at each node in the market

@author: Stephen
"""
import numpy as np

###########################################################################
#Risk Model
###########################################################################
'''
This is only a function to be called. Running this code will do nothing on its own
'''
def riskModel(M, c, num = 0):
    E = np.matrix.copy(M)
    #Get initial assets in system
    E[E < 0] = 0
    Es = np.sum(np.sum(E, axis=0))
    cs = np.sum(c)
    #Initialize sum effect
    effect = 0
    
    #Simulate default at every node
    for node in range(E.shape[0]):
        #If any CCPs are present, skip
        if node < num:
            continue
        else:
            #Redefine E, C for editing
            C = np.copy(c)
            E = np.matrix.copy(M)
            #Initialize gamma
            defaults = []
            
            #Simulate initial default
            defaults.append(node)
            #Whipe out all liabilities tied to initial default
            #because they cannot pay. Assets remain, as we assume
            #they are needed to cover default
            E[:,node] *= 0
            #Recalculate assets and liabilities
            a = np.sum(E, axis=1)
            a[a<0] = 0
            l = np.sum(E, axis=0)
            l[l<0] = 0
            #Initialize while loop variables
            n = 1
            m = 2
            #Loop until no new nodes default
            while n < m:
                #Check initial length
                n = len(defaults)
                #Now update for shock
                for party in defaults:
                    #check what can be covered
                    p = C[party] - l[party] + a[party]
                    #empty reserves
                    np.put(C, party, 0)
                    #update E
                    i = 0
                    for each in E[:,party]:
                        if l[i] == 0:
                            continue
                        else:
                            each *= ((l[i]+p)/l[i])
                        i+=1
                    #get surviving neighbors
                    neighbors = []
                    i = 0
                    for each in M[party,:]:
                        if each != 0 and i not in defaults:
                            neighbors.append(i)
                        i+=1
                    #check if any neighbors default
                    new = []
                    #Recalculate assets and liabilities
                    a = np.sum(E, axis=1)
                    a[a<0] = 0
                    l = np.sum(E, axis=0)
                    l[l<0] = 0
                    for party in neighbors:
                        p = C[party] - l[party] + a[party]
                        #add defaulted neighbors to gamma
                        if p < 0:
                            new.append(party)
                    #check end length (if it increases, we keep going)
                if len(new) != 0:
                    for each in new:
                        defaults.append(each)
                m = len(defaults)
            E[E < 0] = 0
            Ens = np.sum(np.sum(E, axis=0))
            effect += (cs-np.sum(C)) + (Es - Ens)
    totalEffect = effect/(E.shape[0]-num)
    
    return totalEffect