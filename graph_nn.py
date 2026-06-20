
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 19:47:15 2025

@author: manid
"""

#first: make a fully connected neural network. then do sparse


from LinkedListQueue import LinkedListQueue
import math
import matplotlib.pyplot as plt 


#---------------------------------------

class Node:
    def __init__(self):
        self.node_value = 0
        self.actual_number_of_inputs = 0
        self.node_gradient = None  #is dL/d(node)
    
    
    def get_value(self):
        return self.node_value
    
    def set_value(self, value):
        self.node_value = value
    
    def get_actual_number_of_inputs(self):
        return self.actual_number_of_inputs
    
    def increase_actual_number_of_inputs(self):
        self.actual_number_of_inputs += 1
    
    def get_node_gradient(self):
        return self.node_gradient 
    
    def set_node_gradient(self, grad):
        self.node_gradient = grad
        
    def reset(self):
        self.node_value = 0
        self.actual_number_of_inputs = 0
        self.node_gradient = None


class Model:
    def __init__(self):
        self.model = {} #node id: [node's input nodes, node's output nodes, actual node]
        self.number_of_nodes = 0 #used to create new id's
        self.init_value_for_weights = 0.1
        self.input_nodes = []
        self.output_nodes = []
    
    #handles nodes not yet added to the network
    def add_node_to_network(self, is_input = False, is_output = False):
        new_node = Node()
        self.model[self.number_of_nodes] = [set(), {}, new_node]  #input nodes, output nodes, node info
        
        if is_input == True:
            self.input_nodes.append(self.number_of_nodes)
        
        if is_output == True:
            self.output_nodes.append(self.number_of_nodes)
        
        self.number_of_nodes += 1
        
        return new_node
    
    #nodes must ALREADY EXIST
    def connect_nodes(self, node_id_1, node_id_2): #node1-->node2
        if (node_id_1 >= self.number_of_nodes) or (node_id_2 >= self.number_of_nodes):
            return False
        else:
            self.model[node_id_1][1][node_id_2] = [self.init_value_for_weights, None, None]
            self.model[node_id_2][0].add(node_id_1)
            return True
    
    
    def forward_pass(self, input_tensor):
            #replace first node values with inputs --> is O(x) where x = input size
            count=0
            for node_id in self.input_nodes:
                self.model[node_id][2].set_value(input_tensor[count])
                count+=1


           
            for start_node_id in self.input_nodes:
                #bfs traversal
                fringe=LinkedListQueue()  
                fringe.enqueue(start_node_id)  #start with dummy node
               
                while len(fringe)!=0:
                    current=fringe.dequeue()
                   
                   
                    #iterate children (nodes going outward)
                    for adj_node_id in self.model[current][1]:
                        # output node value = sum of input node value * edge weight
                        new_value = self.model[adj_node_id][2].get_value() + (self.model[current][2].get_value() * self.model[current][1][adj_node_id][0])
                        self.model[adj_node_id][2].set_value(new_value)
                        
                        #note that the adjacent node recieved an input node -- fine because the same node won't be put in the fringe again
                        self.model[adj_node_id][2].increase_actual_number_of_inputs()
                       
                        #only proceed with said node if all the inputs have come to that node (if actual == expected)
                        if self.model[adj_node_id][2].get_actual_number_of_inputs() == len(self.model[adj_node_id][0]):
                            fringe.enqueue(adj_node_id)
                        
                        
        #     #guaranteed to have traversed all paths, so guaranteed to have calculated output node. all values stored in nodes. so no return value
            

    def backpropogation(self, y, adam = False, lr = 0.0001):
            learning_rate=lr
            
            #to calculate error (MSE)
            error=0
            count=0
            alpha=0.9
            eps=1e-8
            
            
            
            for end_node_id in self.output_nodes:
               #gradient for end_node_id
               grad = 2 * (self.model[end_node_id][2].get_value() - y[count])  #MSE -- d(L)/d(final node)
               self.model[end_node_id][2].set_node_gradient(grad)
               count += 1
               
            for end_node_id in self.output_nodes:
                fringe=[end_node_id]  #dfs
                
                while len(fringe) != 0:
                    current=fringe.pop()
                    
                    
                    #traverse children (reverse) -- get input nodes
                    for adj_node_id in self.model[current][0]:
                        #calculate gradient and update the weights connecting adj_node to current
                        grad=self.model[current][2].get_node_gradient() * self.model[adj_node_id][2].get_value()   #dE/d(final node)...  *d(node)/d(weight)
                        
                        #update adj_node gradient
                        self.model[adj_node_id][2].set_node_gradient( self.model[current][2].get_node_gradient() * self.model[adj_node_id][1][current][0] ) #dE/d(final node) * d(final node)/d(adj_node) ...
                        
                        #update weight connection adj_node to current
                                        
                        if adam == True:
                            #first store weight averages
                            if self.model[adj_node_id][1][current][1] == None:
                                self.model[adj_node_id][1][current][1] = grad
                            else:
                                self.model[adj_node_id][1][current][1] = alpha * self.model[adj_node_id][1][current][1] + (1-alpha) * grad
                            
                            if self.model[adj_node_id][1][current][2] == None:
                                self.model[adj_node_id][1][current][2] = grad ** 2
                            else:
                                self.model[adj_node_id][1][current][2] = alpha * self.model[adj_node_id][1][current][2] + (1-alpha) * grad ** 2
                            
                            #then take a step
                            self.model[adj_node_id][1][current][0] -= learning_rate *  (self.model[adj_node_id][1][current][1]/math.sqrt(self.model[adj_node_id][1][current][2] + eps))
                            
                        else:
                            #if adam is false:
                            self.model[adj_node_id][1][current][0] -= learning_rate * grad
                        
                        #insert adjacent node into fringe
                        #fringe.insert(0,adj_node)
                        fringe.append(adj_node_id)   #Must be DFS. BFS WILL NOT WORK for node-node skip connection graphs. 
             
            for node in range(0,self.number_of_nodes):  #to reset for forward propogation
                self.model[node][2].reset()

    def train(self, input_tensor, y, epoch = 2000, adam = False, lr = 1e-4):
        for i in range(epoch):
            self.forward_pass(input_tensor)
            self.backpropogation(y, adam = adam, lr=lr)
            
    def predict(self,input_tensor):
        self.forward_pass(input_tensor)
        for node in self.output_nodes:
            print(self.model[node][2].get_value())
#-----------------------------------------






inp_tensor=[1,2,3,4,5]


#create graph/model
model = Model()
#add 10 nodes to the network
for i in range(10):
    if i < 5:
        model.add_node_to_network(is_input=True)
    elif i >= 8:
        model.add_node_to_network(is_output=True)
    else:
        model.add_node_to_network()
        
#creating first layer (fully connected)
for i in range(5,8): #output nodes
    for j in range(0,5): #input nodes
        model.connect_nodes(j, i)
#creating second layer fc
for i in range(8,10): #output nodes
    for j in range(5,8): #input nodes
        model.connect_nodes(j, i)



#training the model
model.train(inp_tensor, [153,234], adam = True, lr = 1e-2)
    
#final prediction
model.predict(inp_tensor)

