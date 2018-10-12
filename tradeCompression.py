"""
Trade compression models for conservative, nonconservative, and hybrid compression
- All functions have similar input and output

Input: numpy adjacency matrix of a market
Output: networkx graph of compressed market, and the compressed numpy matrix

@author: Stephen Kosmo
"""
import numpy as np
import networkx as nx
import scipy.optimize as optimize
import networksimplex as ns

'''
These are only functions to be called. Running this code will do nothing on its own
'''

###########################################################################
#Conservative Compression (network simplex method)
###########################################################################
def conservativeComp(M):
    h = np.matrix.copy(M)
    h[h < 0] = 0
    H = nx.from_numpy_matrix(h, create_using=nx.DiGraph())
    a = np.sum(M, axis=0)
    
    #Define demands (i.e. net positions)
    for node in H:
        try:
            H.node[node]['demand'] = a[node]
        #Some graphs treat a as a matrix with 1 row
        #So you need to choose row 0 first. It's wierd, but it works
        except:
            H.node[node]['demand'] = a[0,node]
    #Define capacity (i.e. compression tolerance), and weight (weight is always 1)
    for i,j,w in H.edges(data=True):
        #Capacity always maters
        H[i][j]['capacity'] = w['weight']
        w['weight'] = 1
    
    #Conservative compression is not always possible
    try:
        #Run network simplex
        flowCost, flowDict = ns.network_simplex(H)
        
        #Build graph
        K = nx.DiGraph(flowDict)
        #Add back edge weights
        for i,j in K.edges():
            K[i][j]['weight'] = flowDict[i][j]
        #Now remove edges with weight = 0
        if K[i][j]['weight'] == 0:
            K.remove_edge(i,j)
    
        #convert K to matrix
        Mc = nx.to_numpy_matrix(K)
        Mc += -np.transpose(Mc)
        return K,np.array(Mc)
    except:
        return H,np.array(M)


###########################################################################
#Hybrid Compression (network simplex method)
###########################################################################
def hybridComp(M):
    h = np.matrix.copy(M)
    h[h < 0] = 0
    H = nx.from_numpy_matrix(h, create_using=nx.DiGraph())
    a = np.sum(M, axis=0)
    
    #Find customers
    customers = []
    i = 0
    for row in M:
        pos = 0
        neg = 0
        for each in row:
            sgnThis = np.sign(each)
            if sgnThis == 1:
                pos+=1
            elif sgnThis == -1:
                neg+=1
        if pos != 0 and neg != 0:
            continue
        else:
            customers.append(i)
        i+=1
    print(customers)
    #Define demands (i.e. net positions)
    for node in H:
        try:
            H.node[node]['demand'] = a[node]
        #Some graphs treat a as a matrix with 1 row
        #So you need to choose row 0 first. It's wierd, but it works
        except:
            H.node[node]['demand'] = a[0,node]
            
    #Define capacity (i.e. compression tolerance), and weight (weight is always 1)
    for i,j,w in H.edges(data=True):
        if i in customers or j in customers:
            #Capacity only maters if i or j is a customer
            H[i][j]['capacity'] = w['weight']
        w['weight'] = 1
    
    #Add in potential inter-dealer trades
    for i in H.nodes():
        for j in H.nodes():
            if i in customers and j in customers and i not in H.neighbors(j):
                H.add_edge(i,j, weight = 1)
    
    #Run network simplex
    flowCost, flowDict = ns.network_simplex(H)
    
    #Build graph
    K = nx.DiGraph(flowDict)
    #Add back edge weights
    for i,j in K.edges():
        K[i][j]['weight'] = flowDict[i][j]
    #Now remove edges with weight = 0
    if K[i][j]['weight'] == 0:
        K.remove_edge(i,j)

    #convert K to matrix
    Mh = nx.to_numpy_matrix(K)
    Mh += -np.transpose(Mh)
    return K,np.array(Mh)
    

###########################################################################
#Nonconservative compression (scipy solution to L1)
###########################################################################
def non_conComp(M):
    n = M.shape[0]
    #Coefficients of obj fn
    c = np.ones(n**2)
    i=0
    for row in range(len(c)):
        if(i > n-1):
            break
        else:
            np.put(c, [(n*i)+i], [0])
            i+=1
    
    #Vector of assets
    b = np.sum(M, axis=1)
    
    #Setup asset eq constraint coefficients
    A = np.zeros((n,n**2))
    put1 = [1]*n
    put2 = [-1]*n
    put1.extend(put2)
    i=0
    for row in A:
        if(i > n-1):
            break
        else:
            rowsum = []
            colsum = []
            x=0
            while x < n:
                rowsum.append((n*i)+x)
                colsum.append(i+(n*x))
                x+=1
            rowsum.extend(colsum)
            np.put(row, rowsum, put1)
            #Go back and 0 the diagonal
            np.put(row, [(n*i)+i], [0])
        i+=1
    
    #Run optimization
    opt = optimize.linprog(c, A_eq=A, b_eq=b, options=dict(tol=1e-8))
    
    #Convert vector of values back into matrix
    Z = opt.x
    Z = np.reshape(Z, (n,n))
    X = np.matrix.copy((Z))
    X[X < 0] = 0
    Gn=nx.from_numpy_matrix(X, create_using=nx.DiGraph())
    return Gn,np.array(Z)
