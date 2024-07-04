
from operator import countOf
from random import sample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import linear_regression as lr

#this function will return 2 subset on containg per% of the dataframe and one containf (100-per)% of the dataframe
def sub_set(df, per):
    rows=len(df)
    indexes= list(range(rows))
    sample_len= round(rows*per)
    sample_ind= sample(indexes, sample_len)
    for val in sample_ind:
        indexes.remove(val)
    elements=[]
    for i in range(sample_len):
        elements.append(df.loc[sample_ind[i]])
    rem_el=[]
    for i in range(len(indexes)):
        rem_el.append(df.loc[indexes[i]])
    sub_set_e= pd.DataFrame(elements)
    sub_set_r= pd.DataFrame(rem_el)
    subsets=[sub_set_e, sub_set_r]
    return subsets

def mse(yo, yc, n):
    se=0
    for i in range(n):
        se+= (yo[i]-yc[i])*(yo[i]-yc[i])
    m=se/n
    return m

def test1(test_set, w):
    yo=list(test_set[1])
    yc=[]
    x=list(test_set[0])
    for i in range(len(test_set)):
        yc.append(w*x[i])
    m= mse(yo,yc,len(test_set))
    return m

def test3(test_set,w, tar, var):
    yo=list(test_set[tar]) #expected output
    yc=[] #computed output
    x=list(test_set[var]) #variable values
    for i in range(len(test_set)):
        y=(w[1]*x[i])+w[0]
        yc.append(y)
    m= mse(yo, yc, len(test_set))
    return m

def test4(test_set, w, t ,x):
    yo=list(test_set[t])
    yc=[]
    x_column=np.array(test_set[x])
    for i in range(len(test_set)):
        y=np.dot(w,x_column[i])
        yc.append(y)
    m= mse(yo,yc, len(test_set))
    return m

def rerunw(data_set, sub_set):
    w1= lr.reg(data_set,1,0)
    w3=lr.reg_offset(sub_set, ' weight', ' mpg')
    w4=lr.mv_reg(sub_set,' mpg',[' disp', ' hp', ' weight'] )
    w=[w1, w3, w4]
    return w

def test(data_set, sub_set,w):
    mse1= test1(data_set, w[0])
    mse3= test3(sub_set, w[1], ' mpg', ' weight')
    mse4= test4(sub_set, w[2],' mpg',[' disp', ' hp', ' weight'])
    ms=[mse1, mse3, mse4]
    return ms

def normalize(l):
    max_val=l[0]
    min_val=l[0]
    l_norm=[]
    for i in range(1, len(l)):
        if(l[i]>max_val):
            max_val=l[i]
        if(l[i]<min_val):
            min_val=l[i]
    for i in range(0, len(l)):
        l_n=((l[i]-min_val)/(max_val-min_val))
        l_norm.append(l_n)
    return l_norm

#load data_set
print("TASK 1\nloading data_sets...")
file = pd.read_csv('turkish-se-SP500vsMSCI.csv', header= None)
motors = pd.read_csv("mtcarsdata-4features.csv")
print("data_set loaded")

#task2.1
print("TASK 2")
print("task 2.1)computing w for the whole turkish data set")
file.plot.scatter(x=0, y=1, color='gray', label="data_set", s=0.5)
w= lr.reg(file,1,0)
print("\t w=", w)
x=np.arange(-1, 1, 0.1)
plt.plot(x, w*x, color='blue', label ="w_ds:"+str(w))
plt.axis([-0.075, 0.075, -0.075, 0.075])
plt.title("turkish data_set")
plt.show()

#task 2.2
print("task2.2) Comparing the solution obtained on different random set(10%) of the whole turkish data set")
file.plot.scatter(x=0, y=1, color='gray', label="data_set", s=0.5)
w_new=[] #to keep the different omega
rows=len(file) #to know number of lines of the whole data_set
indexes= list(range(rows))
sample_len= int(rows*0.1)
colors=['magenta','green','red', 'orange']
for i in range(0,4): #to obatain the 4 data_set
    sample_ind= sample(indexes, sample_len)
    for val in sample_ind:
        indexes.remove(val)
    elements=[]
    for j in range(sample_len):
        elements.append(file.loc[sample_ind[j]])
    subset= pd.DataFrame(elements)
    w=lr.reg(subset,1,0)
    w_new.append(w)
    plt.plot(x, w_new[i]*x, color=colors[i], label="w_ss"+str(i)+":"+str(w_new[i]))
    print("w",str(i),"=", str(w))
plt.title("turkish regresion: different w")
plt.xlabel('x')
plt.ylabel('t')
plt.axis([-0.075, 0.075, -0.075, 0.075])
plt.show()

#task2.3
print("task 2.3) mono dimensional linear regression model on motors data_set on mpg and weight")
w=lr.reg_offset(motors,' weight', ' mpg')
print("y=",w[1],"x +",w[0])
x=np.arange(-10, 100, 10)
motors.plot.scatter(x=' weight', y=' mpg', color ='gray', s=1)
plt.plot(x, w[1]*x+w[0], color='cyan')
plt.xlabel("weight")
plt.ylabel("mpg")
plt.axis([1.5,5.5, 10, 30])
plt.title("motors regression mpg-weight")
plt.show()

#task2.4
print("task 2.4) Computing w for multi dimensional problem on motor data set")
w_md= lr.mv_reg(motors, ' mpg', [' disp', ' hp', ' weight']) #row w_d
print("w:",w_md)

#task3
print("TASK 3")
print("divide in subset of 5 and 95 percent")
sub_sets_t= sub_set(file, 0.05)
sub_sets_m= sub_set(motors, 0.05)

# task 3.1 rerun 2.1,3,4 on 5% of the data
print("task3.1) rerun task 2.1 2.3 and 2.4 on the data set of 5%")
w=rerunw(sub_sets_t[0], sub_sets_m[0])
print("1) w:", w[0], "\n2) [w0, w1]=", w[1], "\n3) w=", w[2])

#task 3.2 compute MSE on test_set 5%
print("task 3.2) compute mse on the train_set 5%")
ms =test(sub_sets_t[0], sub_sets_m[0],w)
print("MSE1=", ms[0],"\nMSE2=", ms[1],"\nMSE3=", ms[2] )

#task 3.3 compute MSE on test_set 95%
print("task 3.2) compute mse on the test_set 95%")
ms2=test(sub_sets_t[1],sub_sets_m[1],w)
print("MSE1=", ms2[0],"\nMSE2=", ms2[1],"\nMSE3=", ms2[2] )

#repeat task 3 different time on different subset
print("rerun task 3 for multiple for different random splits of test/train set")
mse1_1=[]
mse3_1=[]
mse4_1=[]
mse1_2=[]
mse3_2=[]
mse4_2=[]
for i in range(100):
    sub_sets_t=sub_set(file, 0.05)
    sub_sets_m=sub_set(motors, 0.05)
    #rerun 1,3,4 on 5% of the data
    w=rerunw(sub_sets_t[0], sub_sets_m[0])
    #compute MSE on test_set 5%
    ms =test(sub_sets_t[0], sub_sets_m[0],w)
    mse1_1.append(ms[0])
    mse3_1.append(ms[1])
    mse4_1.append(ms[2])
    #compute MSE on test_set 95%
    ms2=test(sub_sets_t[1],sub_sets_m[1],w)
    mse1_2.append(ms2[0])
    mse3_2.append(ms2[1])
    mse4_2.append(ms2[2])
mse3_2_n= normalize(mse3_2)
mse4_2_n=normalize(mse4_2)
#plot histogram of the MSE
plt.figure(figsize=(7,4))
plt.subplot(211)
plt.title('mse turkish data set')
plt.hist(mse1_1, color='orange', ec='black')
plt.xlabel('mse ')
plt.ylabel('count')
plt.subplot(212)
plt.hist(mse1_2,color = 'gold', ec='black')
plt.xlabel('mse ')
plt.ylabel('count')

fig, axes= plt.subplots(figsize=(7,4))
plt.title('mse motors data set monodimensional linear regression')
b=list(np.arange(0,0.7,0.1))
more=[]
[more.append(v) for v in mse3_2_n if v>0.6]
count=len(more)
[mse3_2_n.append(0.6)for i in range(count)]
axes.hist(mse3_2_n, bins=b, color= 'lime', ec='black')
plt.xticks(b)
b[-1]=round(max(mse3_2_n))
axes.set_xticklabels(b)
plt.xlabel('mse')
plt.ylabel('count')

fig, axes = plt.subplots(figsize=(7,4))
plt.title('mse motors data set multidimensional linear regression')
b=list(np.arange(0, 0.7, 0.1))
more=[]
[more.append(v) for v in mse4_2_n if v>0.6]
count=len(more)
[mse4_2_n.append(0.6) for i in range(count)]
axes.hist(mse4_2_n, bins=b,color= 'green', ec='black')
plt.xticks(b)
b[-1]=round(max(mse4_2_n))
axes.set_xticklabels(b)
plt.xlabel('mse')
plt.ylabel('count')
plt.show()

print("Regression Lab terminated")