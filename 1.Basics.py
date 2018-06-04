# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:41:23 2018

@author: abhayakumar
"""

#Get the current working directory
import os
os.getcwd()
##Set the new directory to work in
##Give the path to the desired directory within the quotes
os.chdir("H:\C\Project 01\Training_Session 01\Python_Training_Session 01\Concise_Training 01")
##########

print("Hello! Welcome to session 01")


######## Basic Arithmetics
3+4
3*4
3**4
3/4
3-4
########

######## Values and types
#A value is one of the basic things a program works with, like a letter or a number
#These values belong to different types: 2 is an integer,42.0 is a ﬂoating-point number,and 'Hello, World!' is a string
#?
type(3)
type(42.0) 
type('Hello, World!') 
##?
type('2')
type('42.0') 

##################################################
##################################################

############ Assignment Statements
message = 'And now for something completely different' 
n = 17 
pi = 3.141592653589793
############ Expressions
n*2
n**2
#statement
print(n)


############ String operations
## concatenation
first = 'throat' 
second = 'warbler' 
#?
first + second
## repetition
#?
'Spam'*3
##what if?
'2'-'1'
'eggs'/'easy'
'third'*'a charm' 
 

##################################################
##################################################
### Functions slide
####### Basic inbuilt Functions
type(42) 
print(42)
a= int('31')
print(a)
b= int(31.00123)
print(b)
d=float(32)
print(d) 
e= str(32) 
print(e)
##### what if?
c = int('Hello')
####### Functions from "math" module
import math 
radians = 0.7
height = math.sin(radians) 
math.sqrt(2) / 2.0 
#######
####################################################
####################################################
#Boolean Expressions
5 == 5 
5 == 6 

#Logical Operators
42 and True 

#Conditional Executions
x=61
y=45
if x > 0: 
    print('x is positive')
    
#Alternate Execution
if x % 2 == 0: 
    print('x is even')
else: 
    print('x is odd') 
    
#Chained Conditionals
if x < y: 
    print('x is less than y')
elif x > y: 
    print('x is greater than y') 
else: 
    print('x and y are equal') 
    
#Nested Conditionals
if x == y:
    print('x and y are equal') 
else: 
    if x < y: 
        print('x is less than y') 
    else: 
        print('x is greater than y') 

######################################################
####################################################################### Functions
####### Defining New Fuctions
###Example
###### SYNTAX  ########
def print_lyrics():
    print("I'm a lumberjack, and I'm okay.")
    print("I sleep all night and I work all day.") 

#Deﬁning a function creates a function object, which has type function
type(print_lyrics)
print(print_lyrics) 
#
#Calling the defined function
print_lyrics()
#
####### Flow of Execution (Composite_functions)
#define
def repeat_lyrics():
    print_lyrics()
    print_lyrics()    
#call
repeat_lyrics()
#
########################### Passing arguments(parameters) to a function
########################### Variables and Parameters are local to a Function
def print_twice(bruce):
    print(bruce)
    print(bruce)
    
def cat_twice(part1, part2): 
    cat = part1 + part2 
    print_twice(cat) 
    
line1 = 'Bing tiddle ' 
line2 = 'tiddle bang.' 

cat_twice(line1, line2) 

####??
print(bruce)
print(cat) 
##############################
############################## Functions that return some value
######### An example of Friutful Function
def absolute_value(x):
    if x < 0:
        return -x
    if x > 0:
        return x
####Check?
print(absolute_value(0))

def absolute_value_correct(x):
    if x < 0:
        return -x
    else:
        return x
####Check again
print(absolute_value_correct(0))
######### 
######### An example for a Boolean function
def is_divisible(x, y):
    if x % y == 0:
        return True
    else:
        return False
## Call and check 
print(is_divisible(6, 4))
##################################################################
################################################################################### Iterations
#################################### While statement
#when we don't know of how many iterations, but we know when to terminate
def countdown(n):
    while n > 0:
        print(n)
        n = n - 1
    print('Blastoff!')
    
### Call and check
countdown(3)

#### While with a break when we even don't know when to terminate, but in what case to terminate
def get_input_done():
    while True:
        line = input('> ')
        if line == 'done':
            break
        print(line)
    return line
      
###call
the_text=get_input_done()  
##and check
print(the_text) 

#######################################
####################################### for statement
##when we know of exactly how many iterations
def take_third():
    for i in range(3):
        line = input('> ')
        print(line)
    return line

##call
the_text=take_third()
##and check
print(the_text) 

############################################################################################
############################################################################################