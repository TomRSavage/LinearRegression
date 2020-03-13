import numpy as np 
from scipy.optimize import minimize 
from scipy.optimize import NonlinearConstraint
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import pandas as pd 

def loss(B):
    B0 = B[0]
    B = B[1:]
    error = 0 
    for i in range(len(data_train)):
        error += (response_train[i,:]-B0-sum(data_train[i,j]*B[j] for j in range(len(B))))**2
    return error 

def loss_test(B):
    B0 = B[0]
    Bj = B[1:]
    error = 0 
    for i in range(len(test_data)):
        error += (test_response[i,:]-B0-sum(test_data[i,j]*Bj[j] for j in range(len(Bj))))**2
    return error 

def constraint_fun(B):
    Bj = B[1:]
    #return sum(np.abs(B[i]) for i in range(len(B))) # lasso regression
    return Bj.T@Bj # ridge regression


def fold(i,k,data):
    '''
    takes the ith fold of a set of k total folds and a data and returns this fold
    i takes 0-k
    '''
    percen = 1/k
    fold_amount = percen*len(data)
    start_in = int(fold_amount * i)
    end_in = int(start_in + fold_amount)
    data_test = data[start_in:end_in,:]
    data_train1 = data[:start_in,:]
    data_train2 = data[end_in:,:]
    data_train = np.concatenate((data_train1,data_train2))
    return data_train,data_test

data = pd.read_excel(r'/Users/tomsavage/Dropbox/Documents/Projects/ElementsOfStatisticalLearning/prostate_data.xlsx')
data = data.to_numpy()

for i in range(len(data[0,:])):
    data[:,i] = (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])

for i in range(1):
    data_aug1 = data * np.random.normal(1,0.05,data.shape)
    data_aug2 = data * np.random.normal(1,0.05,data.shape)
    data = np.concatenate((data,data_aug1,data_aug2),axis = 0)

np.random.shuffle(data)
labels = pd.read_excel (r'/Users/tomsavage/Dropbox/Documents/Projects/ElementsOfStatisticalLearning/prostate_data.xlsx',\
        nrows = 0)


t_limit = 0.7
folds = 4
t_fidel = 20
bias = np.zeros((folds,t_fidel))
var = np.zeros((folds,t_fidel))
t_vals = np.linspace(0,t_limit,t_fidel)
B_init = [0 for i in range(len(data[0,:]))]

for j in range(folds):
    data_train_full,data_test_full = fold(j,folds,data)
    test_data = data_test_full[:,:-1]
    test_response = data_test_full[:,-1:]
    data_train = data_train_full[:,:-1]
    response_train = data_train_full[:,-1:]

    for i in tqdm(range(t_fidel)):
        t = t_vals[i]
        constraint = NonlinearConstraint(constraint_fun,0,t)
        sol = minimize(loss,B_init,constraints=(constraint),method='SLSQP')
        B_opt = sol.x 
        loss_val = sol.fun 
        bias[j,i] = loss_val
        var[j,i] = loss_test(B_opt)

bias_mean = np.mean(bias,axis=0)/len(data_train)
var_mean = np.mean(var,axis=0)/len(test_data)

plt.figure()
plt.plot(t_vals,bias_mean,label='Train')
plt.plot(t_vals,var_mean,label='Test')
plt.xlabel(r'$t$ where, $\sum _j| \beta _j| \leq t$')
plt.ylabel('MSE')
plt.legend()
plt.grid()
plt.show()

data = data[:,:-1]
response = data[:,-1:]
B_store = np.zeros((t_fidel,1+len(data[0,:])))
for i in tqdm(range(t_fidel)):
    t = t_vals[i]
    constraint = NonlinearConstraint(constraint_fun,0,t)
    sol = minimize(loss,B_init,constraints=(constraint),method='SLSQP')
    B_opt = sol.x 
    cons_val = constraint_fun(B_opt)
    loss_val = sol.fun 
    B_store[i,:] = B_opt[:]

plt.figure()
for i in range(1,len(B_init)):
    plt.plot(t_vals,B_store[:,i])
    plt.text(t_vals[-1],B_store[-1,i],labels.columns[i-1])
plt.grid()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$t$ where, $\sum _j| \beta _j| \leq t$')
plt.show()









