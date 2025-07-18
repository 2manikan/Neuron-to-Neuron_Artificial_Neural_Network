
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 19:47:15 2025

@author: manid
"""

#first: make a fully connected neural network. then do sparse


from LinkedListQueue import LinkedListQueue

def forward_pass(model, inp):
   
   #replace node values with inputs --> is O(x) where x = input size
   count=0
   for inp_value in inp:
       model[count][2] = inp_value
       count+=1


   
   for start_node in range(0, 5):
       #bfs traversal
       fringe=LinkedListQueue()  
       fringe.enqueue(start_node)  #start with dummy node
       
       while len(fringe)!=0:
           current=fringe.dequeue()
           
           
           #iterate children (nodes going outward)
           for adj_node in model[current][1]:
                model[adj_node][2] += (model[current][2] * model[current][1][adj_node])
                model[adj_node][3] += 1
               
                
                if model[adj_node][3] == len(model[adj_node][0]):
                    fringe.enqueue(adj_node)
               
               
    
    #guaranteed to have traversed all paths, so guaranteed to have calculated output node. all values stored in nodes. so no return value
    

def backpropogation(model, target):
    learning_rate=0.0001
    
    #calculate error (MSE)
    error=0
    count=0
    for end_node in range(8,10):
       
       model[end_node][4] = 2*(model[end_node][2] - target[count])*1     #dE/d(final node)
       count += 1
       
       #debugging
       print("final_values: ",model[end_node][2])    #just prints the predicted value
    print("------------------------")
    
    for end_node in range(8,10):
        fringe=[end_node]  #dfs
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
                fringe.append(adj_node)   #Must be DFS. BFS WILL NOT WORK for node-node skip connection graphs. 
    
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





inp_tensor=[1,2,3,4,5]


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


for i in range(50):
   forward_pass(model, inp_tensor)
   backpropogation(model, [13,200])









