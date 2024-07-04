from tensorflow.keras.datasets import mnist
import kNN as knn
import matplotlib.pyplot as plt
from tabulate import tabulate as tab
import random
import numpy as np 
import class_qualify as qualify

def rearrange(list_int, n):
    """
    Function that produce a new list that is the list passed rearranged by keeping the value n
    and putting -1 when in the list there is something different than n
    return the rearranged list
    """

    new_list=[]
    for i in range(0,len(list_int)):
        if(list_int[i]!=n):
            new_list.append(-1)
        else:
            new_list.append(list_int[i])
    return new_list

#TASK 1: Download the mnist dataset
(train_X, train_Y), (test_X, test_Y)= mnist.load_data()

#TASK 3:test the kNN classifier with 10 task each recognize a digit considering the other as not the digit with different k
ks=[1,2,3,4,5,10,15,20,30,40,50] #different k over we want to classify the test_set to confronte the error rate

#get n subset of indexes of the train set each made on the 10% of the whole trainset to confront it and do the average later
indexes=list()
n=10
for i in range(n):
    ind=list()
    for indx, val in random.sample(list(enumerate(train_X)), int(len(train_X)*0.01)):
        ind.append(indx)
    indexes.append(ind)

#taking only 10%of the mnist dataset because otherwise i don't get results in time
sub_testXl=list()
sub_testY=list()
for indx, new in random.sample(list(enumerate(test_X)),int(len(test_X)*0.01)):
    sub_testXl.append(new)
    sub_testY.append(test_Y[indx])
sub_testX=np.array(sub_testXl)

#to have the 10 task, each recognize a value use a for 
for i in range(10):
    #each iteration a classification for recognize value=i or not
    plt.figure() #open a new figure one for every value we want to recognize
    tablist=list() #list to obtain the table describe the quality of the classification
    
    #rearrengine testY (test label) to contain only i and -1 
    rarr_test=rearrange(sub_testY,i)
    accuracy=list() #list containing accuracy of the whole train for value with every k
    accuracy_v=list() #list contain accuracy of recognize a value with every k
    #classify for every k in the list of k
    for k in range(len(ks)):
        sum1=0 #sum over sensitivity
        sum2=0 #sum of specificity
        sum3=0 #sum of precision
        sum4=0 #sum of recall
        sum5=0 #sum of F1
        q_val=list() #keep teh qualification to do the standard deviation
        sumacc=0 
        sumacc_v=0 
        #for every subset of the train_set
        for j in range(n):
            #j subset of the train set
            sub_trainXl=list()
            sub_trainY=list()
            for ind in indexes[j]:
                sub_trainXl.append(train_X[ind])
                sub_trainY.append(train_Y[ind])
            sub_trainX=np.array(sub_trainXl)
            #rearrrange label of the j subset in order to have only i and -1
            rarr_train=rearrange(sub_trainY, i)
            #call the classification function wich return the classification, the accuracy of it and teh accuracy on recognize the value 
            classification, acc, acc_v = knn.classifier(sub_trainX,rarr_train,sub_testX, ks[k],rarr_test, i)
            #get the quality of the classification (sensitivity, specificity, precision, recall, F1)
            q = qualify.quality(classification, rarr_test)
            #to get the avegare update the sum
            sum1+=q[0] #sum of sens
            sum2+=q[1] #sum of spec
            sum3+=q[2] #sum of prec
            sum4+=q[3] #sum of recall
            sum5+=q[4] #sum of F1
            #to upadte the stadard deviation 
            q_val.append(q)
            sumacc+=acc
            sumacc_v+=acc_v
        qsum1=0
        qsum2=0
        qsum3=0
        qsum4=0
        qsum5=0
        for j in range(n):
            qsum1+=(q_val[j][0]-sum1/n)**2
            qsum2+=(q_val[j][1]-sum2/n)**2
            qsum3+=(q_val[j][2]-sum3/n)**2
            qsum4+=(q_val[j][3]-sum4/n)**2
            qsum5+=(q_val[j][4]-sum5/n)**2
        #create a variable that will be the row of the table will have 
            #k, avg(sens) (stddv(sens)), avg(spec) (stddv(spec)), avg(prec) (stddv(prec)), avg(re) (stddv(re)), avg(F1) (stddv(F1))
        row=[ks[k], str(round(sum1/n,4))+" ("+str(round(np.sqrt(qsum1/n),4))+")", 
        str(round(sum2/n,4))+" ("+str(round(np.sqrt(qsum2/n),4))+")",
        str(round(sum3/n,4))+" ("+str(round(np.sqrt(qsum3/n),4))+")", 
        str(round(sum4/n,4))+" ("+str(round(np.sqrt(qsum4/n),4))+")",
        str(round(sum5/n,4))+" ("+str(round(np.sqrt(qsum5/n),4))+")"]
        #add the row to the list of element for the table
        tablist.append(row)
        accuracy.append(sumacc/n) #average accuracy of the classificatiom for ks[k]
        accuracy_v.append(sumacc_v/n) #average accuracy on recognize the value i for ks[k]
    #print table showing the quality of the classification
    print(tab(tablist, headers=["k", "sensitivity", "specificity", "precision", "recall", "F1"]))
 
    #plot histogram on accuracy
    plt.bar([i-0.5 for i in range(1,4*len(ks)+1,4)], accuracy, color="blue", width=1, label="accuracy of the whole classification", ec='black')
    plt.bar([i+0.5 for i in range(1,4*len(ks)+1,4)],accuracy_v, color="skyblue", width=1, label="accuracy of recognize "+str(i), ec="black" )
    plt.title("average accuracy of the classification regnozing " + str(i)+ " or not over "+str(n)+" subsets of the train test")
    plt.xticks([i for i in range(1, 4*len(ks)+1,4)], ks)
    plt.xlabel("k")
    plt.ylabel("accuracy %")
    plt.legend()
plt.show()