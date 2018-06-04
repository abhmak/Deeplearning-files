# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:59:03 2018

@author: abhayakumar
"""

############################################################################################### Strings
fruit = 'orange'

########Indexing in strings
letter = fruit[1]
letter
########Length of the string
length = len(fruit)
length
###??? the last letter in the string
last = fruit[length]
#?
last = fruit[length-1]

print(last)

#####################################Traversal through a string
#Using While Loop
index = 0
while index < len(fruit):
    letter = fruit[index]
    print(letter)
    index = index + 1
    
#Using For Loop
for letter in fruit:
    print(letter)
    
        
####################################Slicing the strings
s = 'Monty Python'
s[0:5]
s[6:12]

fruit = 'banana'
fruit[:3]

#####################################Strings are Immutable
greeting = 'Hello, world!'

######?????
greeting[0] = 'J'

#Can create a new string
new_greeting = 'J' + greeting[1:]
print(new_greeting)

############################################
####Methods on strings
############################################
#Method to convert to upper case
word = 'banana'
new_word = word.upper()
print(new_word)

#Method to find the index of a letter
word = 'banana'
index = word.find('a')
###?
print(index)

print(word.find('na'))

###########The use of in operator
#in is a Boolean operator
check= 'a' in 'banana'
print(check)

check = 'seed' in 'banana'
print(check)
#########################################################
############################################################################################ LISTS
######################################Creating new lists
cheeses = ['Cheddar', 'Edam', 'Gouda']
numbers = [42, 123]
empty = []
print(cheeses, numbers, empty)



#######################################Indexing in lists is very much similar to strings
###########The in operator also works the same in lists
cheeses = ['Cheddar', 'Edam', 'Gouda']
print('Edam' in cheeses)

###########Traversing in the lists is also similar to strings
for cheese in cheeses:
    print(cheese)
    
########### A list can contain other lists as well
new_l = ['spam', 1, ['Brie', 'Roquefort', 'Pol le Veq'], [1, 2, 3]]

#########################Lists are Mutable
numbers = [42, 123]
numbers[1] = 5
print(numbers)

###Some operations on lists
a = [1, 2, 3]
b = [4, 5, 6]
c = a + b 
#?
print(c)
d=a*3
#?
print(d)
########## Slicing is similar to strings
t = ['a', 'b', 'c', 'd', 'e', 'f']
t[1:3]
t[3:]
t[:3]

############################################### Methods on Lists
#append by an element
t1 = ['a', 'b', 'c']
t2 = ['d', 'e']
t1.append(t2)
print(t1)
#extend by another list
t1 = ['a', 'b', 'c']
t2 = ['d', 'e']
t1.extend(t2)
print(t1) 
#sort
t = ['d', 'c', 'e', 'b', 'a']
t.sort()
print(t)

###############################################################################
##########################################################################################String-List-String
#Converting a string to list type
s = 'spam'
t = list(s)
print(t)
## splitting a string at 'spaces'to a list
s = 'pining for the fords'
t = s.split()
print(t)
## splitting a string at other 'delimiter' to a list
s = 'spam-spam-spam'
delimiter = '-'
t = s.split(delimiter)
print(t)

## Joining a list of strings to a single string
t = ['pining', 'for', 'the', 'fords']
delimiter = ' '
s = delimiter.join(t)
print(s)

###############################################################################
############################################################################################## Dictionaries
################Creating Dictionaries
eng2sp = dict()
eng2sp['one'] = 'uno'
print(eng2sp)


eng2sp = {'one': 'uno', 'two': 'dos', 'three': 'tres'}
print(eng2sp)

####check
eng2sp['four']
eng2sp['three']
#length
len(eng2sp)
#in operator
'one' in eng2sp
'uno' in eng2sp
#the values specifically
vals = eng2sp.values()
'uno' in vals
#################################################### Creating a character histogram from a long string 
def histogram(strng):
    d = dict()
    for ch in strng:
        if ch not in d:
            d[ch] = 1
        else:
            d[ch] += 1
    return d

h = histogram('brontosaurus')
print(h)
######################################################
###################################################### Traversing through the keys in a dictionary
def print_hist(h):
    for key in h:
        print(key, h[key])
        
print_hist(h)

#####################################################################################
################################################################################################# TUPLES
#############Creating a tuple
t = tuple()
t

t = tuple('lupins')
t

#### Indexing is just like lists
t[0]
t[1:3]
##############################################Example use-case
### Swaping values
a=10
b=11
#
temp = a
a = b
b = temp
#

### Swaping values using tuple assignment
a=10
b=11
#
a, b = b, a
#

#######################Zipping of two sequences using tuples
s = 'abc'
t = [0, 1, 2]
z=zip(s, t)
print(z)

print(list(z))
print(z)
###Traversing through the zipped tuple
for pair in zip(s, t):
    print(pair)

#If the sequences are not the same length, the result has the length of the shorter one
z1 =zip('Anne', 'Elk')
print(list(z1))

####################################################################################
#################################################################################### CLASSES and its OBJECTS
##################### Defining a class
class Point:
    """Represents a point in 2-D space."""
#####################
print(Point)
################################# Creating an object of the class
blank = Point()
#######
print(blank)
################################# Assigning values to an instant using dot notation
########################## For the object instance "blank", x and y can be called as its' attributes
blank.x=3.0
blank.y=4.0
##################################
################### Passing an instance as an argument to a function
def print_point(p):
 print('('+str(p.x)+','+str(p.y)+')')

#call
print_point(blank)

################################### Creating another class of type Rectangles
class Rectangle:
    """Represents a rectangle.
    attributes: width, height, corner.
    """
#########Instantiating a Rectangle type object and assigning values to its attributes
box = Rectangle()
box.width = 100.0
box.height = 200.0
box.corner = Point()
box.corner.x = 0.0
box.corner.y = 0.0

##The expression box.corner.x means, “Go to the object box refers to and select the attribute named corner; then go to that object and select the attribute named x.”
    
##################################### Instances as return values
###Take the Rectangle and return me the coordinates of its center
def find_center(rect):
    p = Point()
    p.x = rect.corner.x + rect.width/2
    p.y = rect.corner.y + rect.height/2
    return p
#Call
center = find_center(box)
print_point(center)


### Objects are mutable
box.width = box.width + 50
box.height = box.height + 100
#######################

########################################### Transforming Functions into methods
#Creat a class of type Time
class Time:
    """Represents the time of day."""
    """ (hour,minute,second) as attributes"""

#Define a function with object of type Time as an argument, to print the time of the day
def print_time(time):
    print('%.2d:%.2d:%.2d' % (time.hour, time.minute, time.second)) 

#Instantiating an object of the class Time and assigning values to its attributes
start = Time()
start.hour = 9
start.minute = 45
start.second = 00
## Calling the function and passing the instance "start" as an argument
print_time(start)
##

#####################Transforming to the use of methods
##To make print_time a method, all we have to do is move the function definition inside the class definition
class Time01:
    def print_time(time):
        print('%.2d:%.2d:%.2d' % (time.hour, time.minute, time.second))
 
start = Time01()
start.hour = 9
start.minute = 45
start.second = 00        
##call the function/method
#010  Using the method syntax (more concise and meaningful)
start.print_time() #Apply the method "print_time()" on the object "start" of type Time
#"Hey start! Please print yourself"
