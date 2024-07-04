import numpy as np

def quality(classification, expected):
    """
    Function to compute the confusion matrix of a classification given the classification an the target.
    On that will compute sensitivity, specificity, precision, recall, F1 

    return a list [sensitivity, specificity, precision, recall, F1]

    """
    classes=np.sort(np.unique(np.array(expected)))
    cm=np.zeros((2,2))
    for i in range(len(classification)):
        if(classification[i]== expected[i] & expected[i]==classes[0]):
            #true negative
            cm[0,0]+=1
        elif(classification[i]== expected[i] & expected[i]==classes[1]):
            #true positive
            cm[1,1]+=1
        elif(classification[i]!= expected[i] & expected[i]==classes[0]):
            #false positive
            cm[0,1]+=1
        elif(classification[i]!= expected[i] & expected[i]==classes[1]):
            #false negative
            cm[1,0]+=1 
    cm[0,0]=(cm[0,0]/len(classification))*100
    cm[0,1]=(cm[0,1]/len(classification))*100
    cm[1,0]=(cm[1,0]/len(classification))*100
    cm[1,1]=(cm[1,1]/len(classification))*100
    sensitivity = (cm[1,1]/(cm[1,1]+cm[1,0]))
    specificity = (cm[0,0]/(cm[0,0]+cm[0,1]))
    precision = (cm[1,1]/(cm[1,1]+cm[0,1]))
    recall=sensitivity
    F1=(2*((precision*recall)/(precision + recall)))
    return [sensitivity,specificity,precision,recall,F1]

