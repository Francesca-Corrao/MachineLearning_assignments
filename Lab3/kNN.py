import numpy as np

#task2: create the kNN classifier

def distance(x, xi):
    """
    Function to compute the distance between the matrix x and xi
    the matrix are 28x28 rappresenting the images of the dataset
    
    return distance(float)
    """

    sum=0
    for i in range(len(x)):
        for j in range(len(x[i])):
            sum+=((x[i,j]/784)-(xi[i,j]/784))**2 
    return np.sqrt(sum)

def most_frequent(list):
    """
    Function that find the most frequent label in the list of distances

    return an int corresponding to the label 
    """
    
    values=[]
    for i in range(len(list)):
        values.append(list[i][1])
    val=-1
    n=0
    for i in values:
        if(values.count(i)>n):
            n=values.count(i)
            val=i
    return val

def accuracy(clas, lab, val):
    """
    Function to compute the accuracy of the classification in recognize the value val 
    will compute error rate as number of time there was the value either in the classification or the test_label
    over the number of time it recognize wrong the value

    return accuracy as float    
    """

    n=0
    s=0
    for i in range(0,len(clas)):
        if ((clas[i]==val) | (lab[i]==val)):
            n+=1
            if (clas[i]!=lab[i]):
                s+=1
    if(n!=0):
        acc=(1-(s/n))*100
    else: acc=100
    return acc


def classifier(train_set, label_train, test_set, k, test_label=None, v=None):
    """"
    Function to classify a test_set given a train_set and the relative train_label 
    will verify that test and train have the same number of column and that k is in the right range
    then compute the distances between a image of the test set and the images of the train set
    takes the k minimum distances and choose the most frequent label of the label train that appears as the right one
    If the test label is given will compute the accuracy of the whole classification
    If the paramether v is passed will compute the occuracy on recognize the value v

    return if computed
        -(float) accuracy of the whole classification
        -(float) accuracy of the classification in recognize a value 
    """

    print("value "+ 
    str(v)+" -"+str(k)+"-NN classification")
    (n_elements, r_train, c_train) =train_set.shape
    n_test,r_test, c_test= test_set.shape
    #Check that the number of columns of the second matrix equals the number of columns of the first matrix
    
    if(c_train != c_test) & (r_train != r_test):
        print("test and train have different shapes of images")
        exit(-1)

    #Check that k>0 and k<=cardinality of the training set
    if((k<0) or (k>n_elements)):
        print("k is not on the right range: ", k, r_train)
        exit(-1)

    #Classify the test set according to the kNN rule
    print("classification start")
    classification=[]
    for xi in test_set:
        distances=[]
        for j in range(n_elements):
            d=distance(train_set[j],xi)
            label=(label_train[j])
            distances.append((d, label))
        #ordinare distance per d crescente
        sorted_d=sorted(distances)
        k_min_distances=sorted_d[0:k]
        value=k_min_distances[0][1]
        if(k>1):
            #take the most frequent label in k_min_distances
            value=most_frequent(k_min_distances)
        classification.append(value)
    print("classification: ", classification)

    # If the test set has the optional additional column compute and return the error rate obtained 
    acc=-1
    accuracy_v=-1
    if(not(test_label is None)):
        print("test label is present: ", test_label)
        print("computing accuracy....")
        n_errors= 0
        for i in range(n_test):
            if(classification[i]!= test_label[i]):
                n_errors +=1
        error_rate=n_errors/len(classification)
        acc=(1-error_rate)*100
        accuracy_v=-1
        if(v != -1):    
            accuracy_v=accuracy(classification, test_label, v)     
    print(acc, accuracy_v)
    return classification,acc, accuracy_v
