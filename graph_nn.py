
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 19:47:15 2025

@author: manid
"""

#imports
import numpy as np
import matplotlib.pyplot as plt


#first: make a fully connected neural network. then do sparse
def forward_pass(model, inp):
   
   #replace node values with inputs --> is O(x) where x = input size
   count=0
   for inp_value in inp:
       model[count][2] = inp_value
       count+=1


   
   for start_node in range(0, 5):
       #bfs traversal
       fringe=[start_node] #start with dummy node
       while len(fringe)!=0:
           current=fringe.pop()
           
           
           #iterate children (nodes going outward)
           for adj_node in model[current][1]:
                model[adj_node][2] += (model[current][2] * model[current][1][adj_node])
                model[adj_node][3] += 1
               
                
                if model[adj_node][3] == len(model[adj_node][0]):
                    fringe.insert(0, adj_node)
               
               
    
    #guaranteed to have traversed all paths, so guaranteed to have calculated output node. all values stored in nodes. so no return value
    

def backpropogation(model, target):
    learning_rate=1e-4
    
    #calculate error (MSE)
    error=0
    count=0
    for end_node in range(8,10):
       
       model[end_node][4] = 2*(model[end_node][2] - target[count])*1     #dE/d(final node)
       count += 1
       
       #debugging
       #print("final_values: ",model[end_node][2])    #just prints the predicted value
    #print("------------------------")
    
    for end_node in range(8,10):
        fringe=[end_node]  #bfs
        while len(fringe) != 0:
            current=fringe.pop()
            
            
            #traverse children (reverse)
            for adj_node in model[current][0]:
                #calculate gradient and update the weights connecting adj_node to current
                grad=model[current][-1] * model[adj_node][2]   #dE/d(final node)...  *d(node)/d(weight)
                
                
                
                #update adj_node gradient
                model[adj_node][-1] = model[current][-1] * model[current][0][adj_node]  #dE/d(final node) * d(final node)/d(adj_node) ...
                
                
                #for debugging
                # print("dE/d(final node): ", model[current][-1] , "----- adj value: ",model[adj_node][2])
                # if adj_node == 7 and current == 8:  #just focusing on one particular weight value
                #   print("grad: ",model[current][-1] * model[adj_node][2])   #prints respective grad witho one
                #print("-----------------------------------")
                # print("previous weight: ", model[current][0][adj_node])
                
                
                #update weight connection adj_node to current
                model[current][0][adj_node] -= learning_rate * grad   # -= since we minimize. we do += for maximize
                model[adj_node][1][current] = model[current][0][adj_node]
                
                #for debugging
                # print("learning rate: ",learning_rate,"------------ grad: ",grad)
                # print("new_weight: ", model[current][0][adj_node])
        
                
                #insert adjacent node into fringe
                #fringe.insert(0,adj_node)
                fringe.append(adj_node)   #does not matter if we perform BFS or DFS. 
    
    #for debugging
    # print(model)
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")
    
    
    for node in range(0,10):  #to reset for forward propogation
        if node <= 4:
            model[node][2]=None
        else:
            model[node][2] = 0  
        
        model[node][3] = 0
        model[node][4] = None








#create graph
model={
       
       
       #network(layer 1: 5 nodes to 3 nodes)
       0:[{},{5:0.1, 6:0.1, 7:0.1},None, 0, None],  #input_nodes, output_nodes, value, total inputs actually recieved, node's gradient
       1:[{},{5:0.1, 6:0.1, 7:0.1}, None, 0, None],
       2:[{},{5:0.1, 6:0.1, 7:0.1}, None, 0, None],
       3:[{},{5:0.1, 6:0.1, 7:0.1}, None, 0, None],
       4:[{},{5:0.1, 6:0.1, 7:0.1}, None, 0, None],
       
       #network(layer 2: 3 nodes to one node)
       5:[{0:0.1, 1:0.1, 2:0.1, 3:0.1, 4:0.1},{8:0.1}, 0, 0, None],
       6:[{0:0.1, 1:0.1, 2:0.1, 3:0.1, 4:0.1},{8:0.1}, 0, 0, None],
       7:[{0:0.1, 1:0.1, 2:0.1, 3:0.1, 4:0.1},{8:0.1}, 0, 0, None],
       
       #one node
       8:[{5:0.1, 6:0.1, 7:0.1},{}, 0, 0, None],
       9:[{5:0.1, 6:0.1, 7:0.1},{}, 0, 0, None]
       
       }

#also for debugging
losses=[]

for i in range(300):
   #set up inputs and prediction functions
   inp_tensor=np.random.randint(1,10,size=(5,))
   first_node_target=2*inp_tensor[0]+0.45*inp_tensor[1]+0.73*inp_tensor[2]+1.45*inp_tensor[3]+0.92*inp_tensor[4]
   second_node_target=3.45*inp_tensor[0]+0.23*inp_tensor[1]+1.42*inp_tensor[2]+1.89*inp_tensor[3]+0.333*inp_tensor[4]
   #print("target: ",[first_node_target, second_node_target])
   
   #train model
   forward_pass(model, inp_tensor)
   losses.append((first_node_target - model[8][2])**2 + (second_node_target - model[9][2])**2)
   backpropogation(model, [first_node_target, second_node_target])

plt.plot(losses)
plt.show()









