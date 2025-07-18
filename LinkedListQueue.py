# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 20:53:21 2025

@author: manid
"""

#Linked List


class LinkedListQueue:           #is a doubly linked list
    def __init__(self):
        self.start=None      #so that we dont have to traverse list all the way from the end node
        self.end=None         #so that we don't have to traverse whole list to get to end
        self.length=0
    

    def enqueue(self, value):   #enqueue at end of list
        if self.start==None:
            self.start=Node(value)
            self.end=self.start
        else:
            self.end.next=Node(value)
            self.end=self.end.next
        self.length+=1


    def dequeue(self):
        if self.start==None:
            return
        
        if self.start.next != None:
            value=self.start.value
            self.start=self.start.next
        else:
            value=self.start.value
            self.start==None
        self.length-=1
        return value
    
    def __len__(self):
        return self.length



class Node:
    def __init__(self, value):
        self.next=None
        self.value = value















