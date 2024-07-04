import numpy as np

#global variables use by different function to update or get their values
P_out={} #a priori probabilities of each output
outputs=[] #output values
n_output=0 #number of outout that can be seen
P_variables={} #probabilities of each value for every variables
values=[] #variable values
n_variables=[] #the number of value that each variable can have
likelyhood={} #likelihood of each value of every variables
P_post={} #posteriori probability of the output for every value of each variable
d=0 #numeber of column of the train_set
c=0 #number of column of the test_set
i_tot=0 #number of row of the train_set

#function to update likelyhood and postetiori probabilities
def add_l_p(c,v,o,count):
    global P_out
    global P_variables
    global outputs
    global values
    global likelyhood
    global P_post
    global i_tot
    global n_variables

    index_l="P"+str(c)+"("+str(v)+"|"+str(o)+")" #index of likelyhood
    index_p="P"+str(c)+"("+str(o)+"|"+str(v)+")" #index of P_post
    L1=(count+1)/(i_tot+n_variables[c]) #likelyhood that the c column will have the value v given the fact the output is o
    P1=(L1*P_out["P"+str(o)])/P_variables["P"+str(c)+str(v)] #posteriori probability that the output will be o given the fact the column c has the value v
    L=round(L1,4) #keep only the first 4 decimal to have a more clean output
    P=round(P1,4)
    P_post.update({index_p:P}) #add the posteriori probability
    likelyhood.update({index_l:L}) #add likelyhood

#function to update the a priori probabilities of the output
def add_Pout(i,i_o):
    global P_out
    global n_output
    global i_tot

    index= "P"+str(i) #index to add to P_out will be "P+i" whit i index of the output in the list outputs
    P1=(i_o+1)/(i_tot+n_output) #compute a priori probabilities af number the out appear(i_o)+a/(i_tot+a*n_output) given a=1 
    P=round(P1,4)
    P_out.update({index:P}) #add the probability to the dictionary

#function to update the probability that a variables as a value
def add_P(i,j,count):
    global P_variables
    global n_variables
    global i_tot

    index="P"+str(i)+str(j) #index to add the probability the column i will have the value = vales[j]
    P1=(count+1)/(i_tot+n_variables[i]) #compute probability whit Laplace smoothening whit a=1
    P=round(P1,4)
    P_variables.update({index:P}) #add P to the dictionary

#function training the Classifier 
def train(train_set):
    print("train_set:\n",train_set) #print the train_set
    global d
    global values
    global outputs
    global n_variables
    global i_tot

    #get the values of the column seen on the trains_set excluding the last one
    for i in range(0,d):  
        vs=np.unique(train_set[:,i])
        value=[]
        for v in vs:
            value.append(v)
        values.append(value) #each row will contain the values present in the train_set
    
    #get the output seen on the train_set that are the one in the last column
    out=np.unique(train_set[:,d])
    for o in out:
        outputs.append(o)

    #define all probabilities whit the Laplace smoothing
    i_tot= np.count_nonzero(train_set[:,d]>=1) #count number or row since we know all elements are >=1
    
    #define P_out
    #get P_out for the ouputs that are seen on the train_set
    for i in range(0,len(outputs)):
        out=outputs[i]
        i_o=np.count_nonzero(train_set[:,d]==out) #count how many outputs=out
        add_Pout(i,i_o)
    #if thare aren't all possible outputs then add an a priori probability given count =0 
    if len(outputs)<n_output:
            add_Pout(len(outputs),0) #there will be only one because is the same for every output left out
    
    #define P_variables
    #for the values that are on the train_Set
    for i in range(0,d): #for on the column 
        for j in range(0, len(values[i])): #for on the values of a variable
            val=values[i][j] #get the value of the column i in the position j
            count=np.count_nonzero(train_set[:,i]==val) #count how many variable=val there are
            add_P(i,j,count) #call function to update the P
        #if there are value that aren't on the train_set add a probabilities for them
        if len(values[i])<n_variables[i]:
            add_P(i,len(values[i]),0)
    
    #define likelihood
    #define P_post
    for i in range(0,d): #for on colums (variable)
        for j in range(len(values[i])): #for on values of the variable
            for k in range(0,len(outputs)): #for on outputs
                val=values[i][j] #get the value
                out= outputs[k] #get the output
                count=np.count_nonzero((train_set[:,i]==val)&(train_set[:,d]==out)) #count how many instance of output=out and variable = val
                add_l_p(i,j,k,count) #call function to update likelyhood and P_post
            #if there are outputs non presents update their likelyhoof and P_out whit laplace smoothing
            if len(outputs)<n_output:
                add_l_p(i,j,len(outputs),0)
        #if there are value non presents update 
        if len(values[i])<n_variables[i]:
            for k in range(0,n_output):
                add_l_p(i,len(values[i]),k,0)
            #likelyhood and P_out for values and output non presents
            if len(outputs)<n_output:
                add_l_p(i,len(values[i]),len(outputs),0)
              
#function to test the classifier               
def test(test_set):
    global c
    global values
    global P_post
    results=[] #keep the output decided by the classifier
    expect_r=[] #keep the output expected in the row

    print("test_set:",test_set)

    for row in test_set:
        P=[] #probabilities of the different output
        for i in range(0,n_output):
            P.append(-1) #inizilize P for each output -1
        #naive Bayes:probabilities as the variables are indipendent  
            for j in range(0,len(row)-1): #for every column of the test_set
                if(row[j] in values[j]): # if the value of the column j of the row is in the value of the column j of the train set
                    k= values[j].index(row[j]) #get the index
                else: k=len(values[j]) #otherwise the index is the len(values[i])
                index_p="P"+str(j)+"("+str(i)+"|"+str(k)+")"
                if(P[i]==-1): #if P=-1 first time we use it for the row of test_set
                    P[i]=round((P_post[index_p]),4) #P=P_post
                else:
                    P[i]=round(P[i]*P_post[index_p],4) #else naive assumption is the product                    
        #decide the output   
        output=0 #inizialize to 0
        P_output=0 #inizialize max probabilities to 0 because the real one will be > 0
        #find max probabilities
        for i in range(0, len(P)):
            if P[i]>=P_output: 
                if(i<len(outputs)):
                    output=outputs[i]
                else: 
                    output=i
                P_output=P[i]
        #to know error rate update results and expect_r
        results.append(output)
        if(c==d+1):
            expect_r.append(row[c-1])
    
    #compute ERROR RATE 
    if len(expect_r)>0:
        count=0
        for i in range (0, len(results)):
            if(results[i]!= expect_r[i]):
                count+=1
        error_rate=count/len(expect_r)
        print("output:", results)
        print("error_rate:", error_rate)
    
def classifier(number_of_values,train_set_elem, test_set_elem):
    train_set= np.array(train_set_elem)#train_set nxd+1
    test_set= np.array(test_set_elem) #test_set mxc
    global d
    global c
    global n_output
    global n_variables

    d = int(len(number_of_values) -1) #define d
    c = int(len(test_set[0])) #define c 
    
    if c!=d and c!=d+1:
        print('errore nelle matrici', c , d)
        exit
    else:
        print("good number of columns")
    
    #check values of train_set aren't <1
    for i in range(len(train_set_elem)):
        for value in train_set_elem[i]:
            if value<1:
                print('errore numero maggiore di 1')
                exit
    
    #same check on the test_set
    for i in range(len(test_set_elem)):
        for value in test_set_elem[i]:
            if value<1:
                print('errore numero maggiore di 1')
                exit
    
    n_output = number_of_values[d] #define number of variables
    
    for i in range(0,d):
        n_variables.append(number_of_values[i]) 
    
    #train the classifier
    print("training the classifier....")
    train(train_set)
    print("train end")
    
    #test the classifier
    print("test the classifier...")
    test(test_set)
    print("test end")