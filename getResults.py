"""
Helper functions for results

gibbsResults gives results for a bilateral chain of matrices obtained by Gibbs sampling
Input: gibbs sampled chain of bilateral markets, the length of the chain,
        and the number of ccps to be used in the multi ccp network
Output: average loss for each market, as well graphs of the loss at each step

simpleResults applies compression and riskModel to a single test matrix
Input: numpy adjacency matrix of a market
Output: - single case risk results for each market
        - Graphs of each market, with weight being represented by edge color
        (darker color signifies a higher weight)

@author: Stephen
"""
import tradeCompression as tc
import riskModel as rm
import models
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import math

###########################################################################
#Gibbs
###########################################################################
'''
These are only functions to be called. Running this code will do nothing on its own
'''

def gibbsResults(bilateral_chain, num_chain, num_ccp, distribution, c):
    
    #Initialize averages
    baseAvgBilateral = 0
    conAvgBilateral = 0
    hybridAvgBilateral = 0
    nonconAvgBilateral = 0
    
    baseAvg = 0
    conAvg = 0
    hybridAvg = 0
    nonconAvg = 0
    
###########################################################################
#Bilateral Data
###########################################################################
    #Will be used to add ccp to compressed networks
    cBilateral = []
    hBilateral = []
    nBilateral = []
    
    #Used to disply data
    DataBb = []
    DataCb = []
    DataHb = []
    DataNb = []
    
    for matrix in bilateral_chain:
        #redefine new matrix
        M = matrix
        Gc,Mc = tc.conservativeComp(M)
        Gh,Mh = tc.hybridComp(M)
        Gn,Mn = tc.non_conComp(M)
        
        cBilateral.append(Mc)
        hBilateral.append(Mh)
        nBilateral.append(Mn)
        
        bi = rm.riskModel(M,c)
        con = rm.riskModel(Mc,c)
        hy = rm.riskModel(Mh,c)
        nc = rm.riskModel(Mn,c)
        
        DataBb.append(bi)
        DataCb.append(con)
        DataHb.append(hy)
        DataNb.append(nc)
        
        baseAvgBilateral += bi
        conAvgBilateral += con
        hybridAvgBilateral += hy
        nonconAvgBilateral += nc
        
    baseAvgBilateral = baseAvgBilateral/num_chain
    conAvgBilateral = conAvgBilateral/num_chain
    hybridAvgBilateral = hybridAvgBilateral/num_chain
    nonconAvgBilateral = nonconAvgBilateral/num_chain
    
###########################################################################
#CCP Data
###########################################################################
    ccp_chain = models.ccpNetwork(bilateral_chain)
    c_ccp_chain = models.ccpNetwork(cBilateral)
    h_ccp_chain = models.ccpNetwork(hBilateral)
    n_ccp_chain = models.ccpNetwork(nBilateral)
    
    #analysis if market has only 1 ccp
    if num_ccp == 1:
        #Used to disply data
        DataB = []
        DataC = []
        DataH = []
        DataN = []
        
        #Base
        for matrix in ccp_chain:
            #redefine new matrix
            M = matrix
            #Redefine c to take into account ccp
            #Note: assuming "cover 2" + 13% additional
            #13% is average taken from http://firds.esma.europa.eu/webst/ESMA70-151-1154%20EU-wide%20CCP%20Stress%20Test%202017%20Report.pdf
            N = np.matrix.copy(M)
            N[N < 0] = 0
            m = sorted(np.transpose(N[:,0]).tolist())
            k = max(m[-1], (m[-2] + m[-3]))*1.13
            cc = np.insert(c,0,k)
            bi = rm.riskModel(M,cc,1)
            DataB.append(bi)
            baseAvg += bi
            
        #Conservative
        for matrix in c_ccp_chain:
            #redefine new matrix
            M = matrix
            #Redefine c to take into account ccp
            N = np.matrix.copy(M)
            N[N < 0] = 0
            m = sorted(np.transpose(N[:,0]).tolist())
            k = max(m[-1], (m[-2] + m[-3]))*1.13
            cc = np.insert(c,0,k)
            con = rm.riskModel(M,cc,1)
            DataC.append(con)
            conAvg += con
        
        #Hybrid
        for matrix in h_ccp_chain:
            #redefine new matrix
            M = matrix
            #Redefine c to take into account ccp
            N = np.matrix.copy(M)
            N[N < 0] = 0
            m = sorted(np.transpose(N[:,0]).tolist())
            k = max(m[-1], (m[-2] + m[-3]))*1.13
            cc = np.insert(c,0,k)
            hy = rm.riskModel(M,cc,1)
            DataH.append(hy)
            hybridAvg += hy
        
        #Non-con
        for matrix in n_ccp_chain:
            #redefine new matrix
            M = matrix
            #Redefine c to take into account ccp
            N = np.matrix.copy(M)
            N[N < 0] = 0
            m = sorted(np.transpose(N[:,0]).tolist())
            k = max(m[-1], (m[-2] + m[-3]))*1.13
            cc = np.insert(c,0,k)
            nc = rm.riskModel(M,cc,1)
            DataN.append(nc)
            nonconAvg += nc
        
        baseAvg = baseAvg/num_chain
        conAvg = conAvg/num_chain
        hybridAvg = hybridAvg/num_chain
        nonconAvg = nonconAvg/num_chain
    
###########################################################################
#Multi CCP Data
###########################################################################
    #analysis for >1 ccp
    else:
        #Combine all ccps into one list
        all_ccp = []
        all_ccp.extend(ccp_chain)
        all_ccp.extend(c_ccp_chain)
        all_ccp.extend(h_ccp_chain)
        all_ccp.extend(n_ccp_chain)
        #This way the three ccp model is applied to all chains the same way
        full_chain = models.multiCcpNetwork(num_ccp, all_ccp, distribution)
        multi_chain = full_chain[0:num_chain]
        c_multi_ccp = full_chain[num_chain:2*num_chain]
        h_multi_ccp = full_chain[2*num_chain:3*num_chain]
        n_multi_ccp = full_chain[3*num_chain:4*num_chain]
        
        #Used to disply data
        DataB = []
        DataC = []
        DataH = []
        DataN = []
        
        #Base
        for matrix in multi_chain:
            #redefine new matrix
            M = matrix
            #Redefine c to take into account ccp
            N = np.matrix.copy(M)
            N[N < 0] = 0
            K = []
            for ccp in range(0,(num_ccp+1)):
                m = sorted(np.transpose(N[:,ccp]).tolist())
                k = max(m[-1], (m[-2] + m[-3]))*1.13
                K.append(k)
            cm = np.insert(c, 0, K)
            bi = rm.riskModel(M,cm,num_ccp)
            DataB.append(bi)
            baseAvg += bi
            
        #Conservative
        for matrix in c_multi_ccp:
            #redefine new matrix
            M = matrix
            #Redefine c to take into account ccp
            N = np.matrix.copy(M)
            N[N < 0] = 0
            K = []
            for ccp in range(0,(num_ccp+1)):
                m = sorted(np.transpose(N[:,ccp]).tolist())
                k = max(m[-1], (m[-2] + m[-3]))*1.13
                K.append(k)
            cm = np.insert(c, 0, K)
            con = rm.riskModel(M,cm,num_ccp)
            DataC.append(con)
            conAvg += con
        
        #Hybrid
        for matrix in h_multi_ccp:
            #redefine new matrix
            M = matrix
            #Redefine c to take into account ccp
            N = np.matrix.copy(M)
            N[N < 0] = 0
            K = []
            for ccp in range(0,(num_ccp+1)):
                m = sorted(np.transpose(N[:,ccp]).tolist())
                k = max(m[-1], (m[-2] + m[-3]))*1.13
                K.append(k)
            cm = np.insert(c, 0, K)
            hy = rm.riskModel(M,cm,num_ccp)
            DataH.append(hy)
            hybridAvg += hy
        
        #Non-con
        for matrix in n_multi_ccp:
            #redefine new matrix
            M = matrix
            #Redefine c to take into account ccp
            N = np.matrix.copy(M)
            N[N < 0] = 0
            K = []
            for ccp in range(0,(num_ccp+1)):
                m = sorted(np.transpose(N[:,ccp]).tolist())
                k = max(m[-1], (m[-2] + m[-3]))*1.13
                K.append(k)
            cm = np.insert(c, 0, K)
            nc = rm.riskModel(M,cm,num_ccp)
            DataN.append(nc)
            nonconAvg += nc
        
        baseAvg = baseAvg/num_chain
        conAvg = conAvg/num_chain
        hybridAvg = hybridAvg/num_chain
        nonconAvg = nonconAvg/num_chain
    
###########################################################################
#Results
###########################################################################
    totalAvgNoComp = (baseAvgBilateral + baseAvg)/2
    totalAvgConComp = (conAvgBilateral + conAvg)/2
    totalAvgHybrid = (hybridAvgBilateral + hybridAvg)/2
    totalAvgNonCon = (nonconAvgBilateral + nonconAvg)/2
    
    print('\n'
          'Bilateral:', '\n',
          'Base Market Loss:', baseAvgBilateral, '\n',
          'Conservative Market Loss:', conAvgBilateral, '\n',
          'Hybrid Market Loss:', hybridAvgBilateral, '\n',
          'Non-conservative Market Loss:', nonconAvgBilateral, '\n''\n'

          'Cleared:', '\n',
          'Base Market Loss:', baseAvg, '\n',
          'Conservative Market Loss:', conAvg, '\n',
          'Hybrid Market Loss:', hybridAvg, '\n',
          'Non-conservative Market Loss:', nonconAvg, '\n''\n'
          
          'Averages:','\n',
          'Average loss in Base Market:', totalAvgNoComp, '\n',
          'Average loss in Conservative Market:', totalAvgConComp, '\n',
          'Average loss in Hybrid Market:', totalAvgHybrid, '\n',
          'Average loss in Non-Conservative Market:', totalAvgNonCon, '\n'
          )

###########################################################################
#Graphs
###########################################################################
    #Set up data frames
    df=pd.DataFrame({'x': range(1,num_chain+1),
                     '1.0 Base Market': DataBb,
                     '1.1 Conservative Compression': DataCb,
                     '1.2 Hybrid Compression': DataHb,
                     '1.3 Non-Conservative Compression': DataNb})
    dfm=pd.DataFrame({'x': range(1,num_chain+1),                     
                     '1. Base Market': DataB,
                     '2. Conservative Compression': DataC,
                     '3. Hybrid Compression': DataH,
                     '4. Non-Conservative Compression': DataN,})
    
    return df, dfm

def graph_results(df, dfm, num_chain, title):
    ymin = min(dfm.drop('x', axis=1).values.min(), df.drop('x', axis=1).values.min())
    ymax = max(dfm.drop('x', axis=1).values.max(), df.drop('x', axis=1).values.max())
    
#    #No CCP
    fig, axs = plt.subplots(1,4, sharey = 'row')
    num=0
    for column in df.drop('x', axis=1):
        plt.axes(axs[num])
        
        for v in df.drop('x', axis=1):
            plt.plot(df['x'], df[v], marker='', color='grey', linewidth=0.6, alpha=0.3)
        num+=1
        
        plt.plot(df['x'], df[column], marker='', linewidth=2.4, alpha=0.9, label=column)
        plt.xlim(1,num_chain)
        plt.xlabel('Market Iteration')
        plt.title(column, loc='left', fontsize=12, fontweight=0)
        plt.ylabel('Value Lost')
    plt.ylim(ymin,ymax)
    plt.ticklabel_format(useOffset=False, axis='y')
    plt.suptitle(title + ": Losses from Default (Bilateral Market)", fontsize=13, fontweight=0, color='black', style='italic')
    plt.show()
    
#Histogram of results
    fig, ax = plt.subplots()
    bins = int(10*((math.log(num_chain))+1))
    x = [df['1.0 Base Market'], df['1.1 Conservative Compression'],
            df['1.2 Hybrid Compression'], df['1.3 Non-Conservative Compression']]
    legend = ['1.0 Base Market', '1.1 Conservative Compression',
              '1.2 Hybrid Compression', '1.3 Non-Conservative Compression']
    
    plt.hist(x, bins, range=[ymin,ymax], )
    plt.legend(legend)
    plt.ticklabel_format(useOffset=False, axis='y')
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Value Lost")
    plt.suptitle(title + ": Distribution of Losses (Bilateral Market)", fontsize=13, fontweight=0, color='black', style='italic')
    plt.show()
    

#    #Multi CCP
    fig, axs = plt.subplots(1,4, sharey = 'row')
    num=0
    for column in dfm.drop('x', axis=1):
        plt.axes(axs[num])
        
        for v in dfm.drop('x', axis=1):
            plt.plot(dfm['x'], dfm[v], marker='', color='grey', linewidth=0.6, alpha=0.3)
        num+=1
        
        plt.plot(dfm['x'], dfm[column], marker='', linewidth=2.4, alpha=0.9, label=column)
        plt.xlim(1,num_chain)
        plt.xlabel('Market Iteration')
        plt.title(column, loc='left', fontsize=12, fontweight=0)
        plt.ylabel('Value Lost')
    plt.ylim(ymin,ymax)
    plt.ticklabel_format(useOffset=False, axis='y')
    plt.suptitle(title + ": Losses from Default (Cleared Market)", fontsize=13, fontweight=0, color='black', style='italic')
    plt.show()

#Histogram of results
    fig, ax = plt.subplots()
    bins = int(10*((math.log(num_chain))+1))
    x = [dfm['1. Base Market'], dfm['2. Conservative Compression'],
              dfm['3. Hybrid Compression'], dfm['4. Non-Conservative Compression']]
    legend = ['1. Base Market', '2. Conservative Compression',
              '3. Hybrid Compression', '4. Non-Conservative Compression']
    
    plt.hist(x, bins, range=[ymin,ymax])
    plt.legend(legend)
    plt.ticklabel_format(useOffset=False, axis='y')
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Value Lost")
    plt.suptitle(title + ": Distribution of Losses (Cleared Market)", fontsize=13, fontweight=0, color='black', style='italic')
    plt.show()
    

###########################################################################
#Single Matrix
###########################################################################

def simpleResults(numpyMatrix):
    
    M = numpyMatrix
    N = np.matrix.copy(M)
    N[N < 0] = 0
    #Here, reserves are defined as 50% liabilities for simplicity
    l = np.sum(N, axis=0)
    c = 0.5*np.absolute(l)

    G=nx.from_numpy_matrix(N, create_using=nx.DiGraph())
    Gc,Mc = tc.conservativeComp(M)
    Gh,Mh = tc.hybridComp(M)
    Gn,Mn = tc.non_conComp(M)
    
    baseBilateral = rm.riskModel(M,c)
    conBilateral = rm.riskModel(Mc,c)
    hybridBilateral = rm.riskModel(Mh,c)
    nonconBilateral = rm.riskModel(Mn,c)
    totalBilateral = (baseBilateral + conBilateral + hybridBilateral + nonconBilateral)/4
    
    #Output with CCP
#    n = np.size(M,axis=0)
#    a = np.sum(M, axis=1)
#    l = np.sum(M, axis=0)
#    ccp = np.zeros((n+1, n+1))
#    for j in range(1, n+1):
#    	ccp[0, j] = a[j-1]
#    	ccp[j, 0] = l[j-1]
#    Ncl = ccp
#    Ncl[Ncl < 0] = 0
#    Gcl = nx.from_numpy_matrix(Ncl, create_using=nx.DiGraph())
    
    ###########################################################################
    #Graphs
    ###########################################################################
    
    labels = {0:'C',1:'D',2:'D',3:'D',4:'C'}
    
    fig = plt.figure(1)
    nx.draw_circular(G, node_size=1000, labels = labels,)
    nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G), edge_labels = nx.get_edge_attributes(G,'weight'))
    nx.draw_networkx_edges(G,pos=nx.circular_layout(G),width=4)
    plt.suptitle('Base Market')
    plt.savefig('noComp.png', facecolor = fig.get_facecolor(), transparent = True)
    plt.show()
    
    fig = plt.figure(2)
    nx.draw_circular(Gc, node_size=1000, labels = labels,)
    nx.draw_networkx_edge_labels(Gc, pos=nx.circular_layout(Gc), edge_labels = nx.get_edge_attributes(Gc,'weight'))
    nx.draw_networkx_edges(Gc,pos=nx.circular_layout(Gc),width=4)
    plt.suptitle('Conservative Compression')
    plt.savefig('conComp.png', facecolor = fig.get_facecolor(), transparent = True)
    plt.show()
    
    fig = plt.figure(3)
    nx.draw_circular(Gh, node_size=1000, labels = labels,)
    nx.draw_networkx_edge_labels(Gh, pos=nx.circular_layout(Gh), edge_labels = nx.get_edge_attributes(Gh,'weight'))
    nx.draw_networkx_edges(Gh,pos=nx.circular_layout(Gh),width=4)
    plt.suptitle('Hybrid Compression')
    plt.savefig('hybridComp.png', facecolor = fig.get_facecolor(), transparent = True)
    plt.show()
    
    fig = plt.figure(4)
    nx.draw_circular(Gn, node_size=1000, labels = labels,)
    nx.draw_networkx_edge_labels(Gn, pos=nx.circular_layout(Gn), edge_labels = nx.get_edge_attributes(Gn,'weight'))
    nx.draw_networkx_edges(Gn,pos=nx.circular_layout(Gn),width=4)
    plt.suptitle('Non-Conservative Compression')
    plt.savefig('nonConComp.png', facecolor = fig.get_facecolor(), transparent = True)
    plt.show()
    
#    fig = plt.figure(5)
#    edges = Gcl.edges()
#    weights = [Gcl[u][v]['weight']/10 for u,v in edges]
#    pos = nx.circular_layout(Gcl)
#    pos[0] = np.array([0,0])
#    labels = {}
#    for node in Gcl.nodes():
#        if node == 0:
#            labels[node] = 'CCP'
#        else:
#            labels[node] = node-1
#    nx.draw(Gcl, pos, node_size=1000, edge_color=weights, edge_cmap=plt.cm.Blues)
#    nx.draw_networkx_labels(Gcl, pos, labels, font_size=12)
#    plt.suptitle('Cleared Market')
#    fig.set_facecolor("#909090")
#    plt.savefig('cleared.png', facecolor = fig.get_facecolor(), transparent = True)
#    plt.show()
    
    ###########################################################################
    #Results
    ###########################################################################
    
    print('Base Market Loss:', baseBilateral, '\n',
          'Conservative Market Loss:', conBilateral, '\n',
          'Hybrid Market Loss:', hybridBilateral, '\n',
          'Non-conservative Market Loss:', nonconBilateral, '\n',
          'Average Loss:', totalBilateral, '\n','\n')