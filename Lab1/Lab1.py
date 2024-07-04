import NBCL as nbc
import random as random

#open and read a file
file = open('data_set.txt','r')
info= file.read()

#to create matrix , make a list of rows
#for each row split in elements
rows = info.split("\n") #list of rows
matrix_elem=[] #elements of the file
vs=rows[0].split() #row that contains the number of value for each variable 
n_v=[]
for v in vs:
    n_v.append(int(v))
for i in range(1,len(rows)):
    elem = rows[i].split() #get list of elements of each row
    m=[]
    for e in elem:
        m.append(int(e)) #recreate the row now will contain int
    matrix_elem.append(m) #add the row to the list of element of the file
n_c= len(matrix_elem[0])

#create matrix whit random rows of the row in the file
ind=[]
for i in range(0,len(matrix_elem)):
    ind.append(i)
n_rts= int(len(matrix_elem)*0.75) #numbers of row for the train_set

train_set_elem=[] #list contain train_set elements
test_set_elem=[] #list_contain test_set elements

for i in range(n_rts): 
    j=random.randrange(len(ind)) #get a random index to access the row to put on the train set
    k=ind[j] #get the indes
    ind.remove(k) #remove the index
    train_set_elem.append(matrix_elem[k]) #add the row to the list of train_set

#the remain index will be use to get the row for the test_set
for i in range(len(ind)):
    j=ind[i]
    test_set_elem.append(matrix_elem[j])

#Train and test the Naive Bayes Classifier calling it
nbc.classifier(n_v, train_set_elem, test_set_elem)