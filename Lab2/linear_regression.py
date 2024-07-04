import numpy as np
import pandas as pd
def reg(data_set, t, x):
    num=0
    den=0
    x_column=list(data_set[x])
    t_column=list(data_set[t])
    for i in range(len(data_set)):
        num = num+(x_column[i]*t_column[i])
        den+= (x_column[i]*x_column[i])
    w=num/den
    return w

def reg_offset(dataframe, x, t):
    t_column=list(dataframe[t])
    x_column = list(dataframe[x])
    w1=0
    t_s=0
    x_s=0
    for i in range(len(dataframe)):
        t_s+=t_column[i]
        x_s+=x_column[i]
    t_m=t_s/len(dataframe)
    x_m=x_s/len(dataframe)
    num=0
    den=0
    for i in range(len(dataframe)):
        num+= (x_column[i]-x_m)*(t_column[i]-t_m)
        den+= (x_column[i]-x_m)*(x_column[i]-x_m)
    if (den==0) and (num==0):
        w1=(x_m*t_m)/x_m
    else : w1=num/den
    w0= t_m-(w1*x_m)
    w=[w0,w1]
    return w

def mv_reg(dataframe, t, v):
    matrix_x= np.array(dataframe[v]) #X matrix
    MPpi= np.linalg.pinv(matrix_x) #Moore-Penrose pseudo inverso of X
    t_column= np.array(dataframe[t]) #T column
    w=np.dot(MPpi,t_column) #row w_d
    return w