# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:24:42 2018

@author: abhayakumar
"""
#---------------------------------------------
# Creating Arrays - from lists 
#---------------------------------------------
import numpy as np
#from numpy import *
# Single dimensional array 
array_data1 = [1.4, 7, 8, 12.3]
array1 = np.array(array_data1)
print("Array 1 : ",array1)
#
# Multi dimensional array 
#
array_data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
array2 = np.array(array_data2)
print("Array 2: ", array2)

# Check the Dimension of the array 
print("Dimension of array2 is ",array2.ndim)

# Check the Size of Array 
print("Shape of array2 is ",array2.shape)

# Create and array of zeros 
Zero_Array = np.zeros((3, 6))
print("Zero Array",Zero_Array)

# Check Data Type of Array 
array_data1 = [1,2,3,4]
Array1 = np.array(array_data1)
print("data type of Array1 is : ",Array1.dtype)

# Convert numeric data as  strings to float 
# Create a string array 
num_strings = np.array(['1.25', '-9.6', '42'])
# Check data type of array 
print("data type of num_strings  is : ",num_strings.dtype)
# Typecast to float 
float_array = num_strings.astype(np.float64)
# Check data type of float array 
print("data type of float_array  is : ",float_array.dtype)
print(float_array)
#-------------------------------------------------------------------
# Arrays and Scalars 
#-------------------------------------------------------------------
# Notes:1)  Operations between Arrays of same size happens element wise 
# Notes:2)  Operations with Arrays and Scalars will propagate to each element 

#--------------------
# ELement wise array multiplication 
#--------------------
arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = arr1*arr1 
print("Original Array ",arr1)
print("Element wise sq of arr1 is : ",arr2)
arr3 = arr1*2
print("Double array element wise", arr3)

#--------------------------------------------
# Indexing and slicing of arrays 
#--------------------------------------------
################################################### Define Array of 10 elements 
Arr1 = np.arange(10)
print("Arr1 before manipulation is :",Arr1)
# Choose Index pos 3 to 5 and create a sub array 
# Note the 5th Index position actually does not get selected 
# Note: Only elemenst upto the 5th position get selected 
Sub_Arr1 = Arr1[3:5]
# Display elements of sub array 
print("Sub_Array before change is:",Sub_Arr1)
# Assign a fixed value to sub Array Index Pos 0 
Sub_Arr1[0] = 56
# Display elements of Sub_Arr1
print("Sub Array after change :",Sub_Arr1)
# Display elements of original array 
# Note: change in sub Array caused change in main array 
print("Original Array is",Arr1)
# Assign a element to the entire Sub array 
Sub_Arr1 = 66
# Display Sub Arry 
print("Sub Array after new assign is : ",Sub_Arr1)
# Display main Array 
print("Main Array after new assign is : ", Arr1)

##################################################### Slicing multi dimensional Arrays 
# define a 3 by 3 array 
Array1 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("Original Array is ", Array1)
Array_Slice1 = Array1[2,:3]
print("Sliced Array is : ",Array_Slice1)


# Boolean Indexing 
#-------------------------------
# Create an Array of names 
Arr_Names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# Get Array size
Arrsize = Arr_Names.shape
Len1 = Arrsize[0]
# Create a  Vector of random numbers of the same length
# Init a  Vector of size = Len1 
Rand_Set = np.array(np.arange(Len1))
# Populate Vectpr with Randome nos from 1 to 10 
for i in range(Len1):
    Rand_Set[i] = np.random.uniform(1,10)
# Display Rand Vector
print("Rand Vector",Rand_Set)
# Get Those Index positions for which name = Bob 
# Use the index set to choose a sub set of Random Numbers 
Match_name_rand = Rand_Set[Arr_Names == 'Bob']
print("Matching Sub set of Rand Num is:",Match_name_rand)

#----------------------------------------------------------
# Array Transpose 
#----------------------------------------------------------
# define A matrix 
A_Mat = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
# Print A Matrix 
print("A Matrix",A_Mat)
# Transpose
A_Trans = A_Mat.T
# Print Transpose  of A Matrix 
print("Transposed A Mat",A_Trans)
# Computing At A  
Dot_Prod = np.dot(A_Trans,A_Mat) 
# Print product 
print("Dot_Product : At A",Dot_Prod)


#----------------------------------------------------------
#   Universal  Array Functions 
# Work elementwise on an array 
# Some examples 
#----------------------------------------------------------
# Unary Function Example
#-----------------------
# Square Root 
#-------------

Arr = np.arange(10)
Sqrt_Arr = np.sqrt(Arr)
print("Sqrt Root values:",Sqrt_Arr )


#--------------------------
# Binary Function example 
#---------------------------

# Create Two Arrays of 10 random numbers each 
x = np.random.randn(10)
y = np.random.randn(10)
# Print  both the arrays 
print("X Array is :", x)
print("Y Array is :", y)
# Get the Element wise max, comparing both arrays 
Max_XY = np.maximum(x,y)
print("The Max element wise is :", Max_XY)

######??
# Conditional logic as Array Operators
# ---
# Define Boolean Array 
Bool_Arr = np.array([True,False,True,False,True])
X_Arr = np.array(['a','b','c','d','e'])
Y_Arr = np.array([10,20,30,40,50])

Choice = [(X if Z else Y) for X,Y,Z in zip(X_Arr,Y_Arr,Bool_Arr)]
print("Choice: ",Choice)

#---------------------------------
# The where clause in arrays 
#---------------------------------
# Create a 3x3 array of random numbers 
Arr3 = np.random.randn(3,3)
Arr3
# Replace positive numbers with +2 and _ve nos with -2 
Arr3_Chg = np.where(Arr3 > 0, 2, -2)
# Print changed Array 
print("Chg Array is : ", Arr3_Chg)


#-------------------------------------
# Basic Statistical Functions 
#-------------------------------------
Arr_Stat = np.array([[1,2,3],[4,5,6],[7,8,9]])
# Mean
Arr_Mean = np.mean(Arr_Stat) 
print("Mean of Array is: ",Arr_Mean)
# Sum 
Arr_Sum = np.sum(Arr_Stat)
print("Sum of Array is:",Arr_Sum)

#--------------------------------------------
# Stat Operations along a particular direction
#--------------------------------------------

# Get Column means (0th dimension or axis )
Arr_Mean_Cols = Arr_Stat.mean(0)
print("Col_Means = :", Arr_Mean_Cols)

#rows?


#------------------------------------------------
# methods for Boolean Arrays 
#------------------------------------------------
# Create an array of 10 Random nos 
Arr10 = np.random.randn(10)
# Find out how many are positive numbers 
Poscnt = (Arr10 > 0).sum()
print("nos of pos numbers in Arr10 is : ",Poscnt)
print("Arr10 is : ",Arr10)

#--------------------------------------------
#   Sorting 
#--------------------------------------------
Arr10_Sorted = sorted(Arr10) 
print("Unsorted Array is ",Arr10)
print("Sorted Array is ",Arr10_Sorted)
#-------------------------------------------------
#  Unique  and other Array set operations 
#-------------------------------------------------
# Create a Names Array 
Names_Arr = np.array(["Jim","Bob","Roy","Rajesh","Roy","Kumar","Jeet","Roy","Joy"])
# Get Unique Names 
Names_Arr_Unq = np.unique(Names_Arr)
print("Unique Names are ",Names_Arr_Unq)
#-----------------------------------------------------
# Linear Algebra Modules in Python - Examples 
#-----------------------------------------------------
# Matrix Multiplication Dot Product 
A_Mat = np.array([[1,2],[3,4]])
B_Mat = np.array([[1,2],[3,4]])
Dot_Prod_A_B = np.dot(A_Mat,B_Mat)
print("Dot Product A and B ",Dot_Prod_A_B)

# Trace of a matrix ( Sum of diag elements )
Trace_A = np.trace(A_Mat)
print("Trace of A Mat is ", Trace_A)




#-----------------------------------------------------
# Random number generation 
#-----------------------------------------------------
# Generate a 4 by 4  Array of random numbers following 
# Normal distribution 
#------------------------------------------------------
Arr4by4 = np.random.normal(loc=5,scale=1,size=(4,4))
Arr4by4
#-------------------------------------------------------
# Case Study on Random Walk 
#------------------------------------------------------
# Notes : we are generating a 100 step random walk and will plot it 
#----------------------------------------------------------------
# Import Plot library 
import  matplotlib.pyplot  as plt
#import random 
pos = 0 
walk = [pos]
steps = 100
for i in np.arange(100):
    step = 1 if np.random.randint(0,2) else -1 
    pos = pos + step 
    walk.append(pos)
    
plt.plot(walk)   

###########################################################################################
################################################################################################## pandas

#---------------------------------------------
# 1D Data Structure - the series (index to the values)
#-------------------------------------------
import pandas as pd
#import scipy as sp

Series1  = pd.Series([1, 2, 3, 4])
print("Series 1 is : ",Series1)
# Extracting the Index and Values separately from a Series 
Series1_Val = Series1.values
Series1_Idx = Series1.index
print("Series values ",Series1_Val)
print("Series Indices ",Series1_Idx)

#-------------------------------------------
# Specifying a Series with Indices pre defined
#-----
Series2 = pd.Series([1,2,3,4], index=['a','b','c','d'])
print("Series2 = ",Series2)

#-------------------------------------------
# Converting Dictionary to Series 
#------------------------------------------
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
Series_State = pd.Series(sdata)

Series_State

# Now consider passing Only key Values  with A Invalid Key 

States1 = ['Ohio','Texas','Oregon','California']
Series_State2 = pd.Series(sdata,index=States1)
# Check Values now with this Key 
print("Values based on new key set ",Series_State2.values)
# Observe the NA in the position of the invalid key 
# ----------------------------------------------------------------
# Detecting missing data in series
#-----------------------------------------------------------------
NullCheck = pd.isnull(Series_State2)
print("NullCheck is :  " ,NullCheck)
#-------------------------------------------------------------
#  DATA FRAMES
#-------------------------------------------------------------

# Create a Data Frame from a Equal length list or numpy arrays 
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

# Check data type to confirm its a dictionary 
type(data)
# Convert to Dataframe 
df = pd.DataFrame(data)

# Display the DataFrame 
df # Note column names are in alphabetical order 

# Customize column sequence 
df_new = pd.DataFrame(data, columns=['year', 'state', 'pop'])
df_new

# Retrieving col information 
State_Data = df_new.state

# Display Retrieved Col 
State_Data 
type(State_Data)
print(State_Data)
# Check the indices for this dataframe
df_new.index

# Print data for index position 2 
df_new.ix[2]
df_new.iloc[2]
df_new.loc[2]

# Re_set data values - Year col all set to a scalar 
df_new['year'] = 2001

# re check data in this data frame : All year values changed to 2001 
df_new

# Create a New column 
df_2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'])

# display new data frame 
df_2

# Fill Data in specified index positions in New Col 
val = pd.Series([-1.2, -1.5, -1.7], index=[0, 1, 2])
df_2['debt'] = val 

# Display Dataframe again 
df_2 
# Create a Computed Column 
# Just assign a column which does not exist 
# eg create a col which is true f state is Eastern State of Ohio 
df_2["Eastern"] = df_2.state == 'Ohio'

# Display  again 
df_2
# Delete a column 
del df_2["Eastern"]
# Display again 
df_2

#???????????-------------------------------------
# Creating a Data Frame from a nested dict of dicts 
#???????????-------------------------------------------------------------
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
        'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
df_3 = pd.DataFrame(pop) 
# Display 
df_3 
####
# We can transpose as well 
df_3_T = df_3.T 
# Display Transposed 
df_3.T 


#------------------------------------------
# Dropping entries from axis 
#-------------------------------------------
# eg with 1D data i.e. series 
#---------------------------------
# Create Series 
Ser1  = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
# Drop data for index value = c 
New_Ser = Ser1.drop('c')
# Display after drop 
New_Ser

#----------------------------------
#  With 2 D data i.e. dataframes
#----------------------------------
# Create data Frame Note: Matrix fills By Row as default 
DF = pd.DataFrame(np.arange(16).reshape((4, 4)),
  index=['Ohio', 'Colorado', 'Utah', 'New York'],
  columns=['one', 'two', 'three', 'four'])
  
 # Display DF 
DF  
# drop rows 
DF_1 = DF.drop(['Colorado', 'Ohio'])
# Display DF
DF_1
# Drop columns : note columns are treated as axis 1 
DF_2 = DF.drop('two',axis=1)
# Display 
DF_2

#------------------------------------------------------------
#  Subselect data using indexing  - 1D Array 
#------------------------------------------------------------
# Create a numeric array of size 6 
Ser1 =  pd.Series(np.arange(5.),index=['a', 'b', 'c', 'd','e'])
# Display the Series 
Ser1
# Index by Index Name directly
Ser1_Idx_b = Ser1['b']
# Display extract
Ser1_Idx_b
# Use a Index range to Redefine Series elements 
Ser1['b':'d'] = 55
# Display after Re indexing 
Ser1
#-------------------------------------------------------------
#  Subselect data using indexing - 2D Array 
#-------------------------------------------------------------
# Create a DF : We use the DF created already above 
# Check DF 
DF
# Slice Two columns from DF 
# Note the Double Subscripting  to refer to columns 
DF_subset = DF[['two','three']]
# Display 
DF_subset 
#-------------------------------
# Replace all data  values less than 5 with zeros 
#--
DF[DF < 5] = 0
# Check replacement 
DF
#-------------------------------------------------------
# Use Special indexing ix  to subselect rows then  cols 
#-------------------------------------------------------
DF_One_row_2Cols = DF.loc['Colorado', ['two', 'three']]
# Display 
DF_One_row_2Cols 

# Row pos 2 col pos 3
DF_Row2_Col3 = DF.iloc[1,2]
# Display 
DF_Row2_Col3
#---------------------------------------------------------
#  Union of Series 
#---------------------------------------------------------
# Def Series 1
Ser_1 =  pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
# Def Series 2
Ser_2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])

# Union 
Ser_New = Ser_1 + Ser_2 
# Display 
Ser_New 
#----------------------------------------------------------
#  Union of Data Frame 
#----------------------------------------------------------
# Def data Frame 1 
#----------------------------------------------------------
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
          index=['Ohio', 'Texas', 'Colorado'])
# Display 
df1
#----------------------------------------------------------
# Def data Frame 2 
#----------------------------------------------------------
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
 index=['Utah', 'Ohio', 'Texas', 'Oregon'])

# Display 
df2
# Union 

df_new = df1 + df2
# Display 
df_new
# Observe all non match col and index pos are filled with NAn
#-----------------------------------------------------
#--    Add with fill value for Nans
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))

#-----------
#  Display 
df1
df2
#---------------------------
# Use fill value of 0 to fill up cases of no overlap
df_new2 = df1.add(df2, fill_value=0)
# Display 
df_new2

#---------------------------------------------------------------
#  OPerations between DF and Series 
#------------------------------------------------------------
# Define a 2D DF 
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)))
# Display 
df1
#----------------------
# Extract 1st Row 
df_1st_row = df1.iloc[0]
# Display 
df_1st_row
# Subtract 1st Row from the 2D Array 
#------- Note that the operations will apply to all rows 
# Also called broadcast  method
# Index match of cols : broadcast down the rows 
df_subtract = df1 - df1.iloc[0]
# Display 
df_subtract

#--------------------------------------------------------------------
# Applying Functions to Data Frame cols or Rows 
#-----------------------------------------------------------------
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
# Display 
df1
# ----------------------------------------------------------------------
#  default - summary applied over cols 
#-------------------
MaxCol = np.max(df1)
MaxCol
 #  summary applied over rows 
MaxRow = np.max(df1,axis=1)
MaxRow
#-------------------------------------
# Sorting of Indices 
#--------------------------------------
# 1)  sorting lexicographically by index 
#----------------------------------------------
# define series  
df1 = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
#  Display 
df1 
# Sort on index 
df1_sorted = df1.sort_index()
df1_sorted
#-------------------------------------------------------------------
# Sorting for 2D  Frame 
# Create a 2D DF
DF2 = pd.DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],
 columns=['d', 'a', 'b', 'c'])
#-------------------------------------------------------------------
# Display 
#------------------------------------------------------------------
DF2
# Sort Index
DF2_Srted = DF2.sort_index()
# Display Sorted ( row index by default chosen )
DF2_Srted 
# sort by col index 
DF2_col_srt = DF2.sort_index(axis=1)
# Display 
DF2_col_srt

#--------------------------------------------------------------------
# Descriptive Statistics 
#--------------------------------------------------------------------
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
      [np.nan, np.nan], [0.75, -1.3]],
      index=['a', 'b', 'c', 'd'],
      columns=['one', 'two'])

# Display 
df
# Compute Sum : default col is summed 
DFSUM = df.sum()
DFSUM
# row sum 
DF_Row_SUM = df.sum(axis=1)
DF_Row_SUM

#--------------------------- Summary Statistics of each col 
# Applied to numeric data it produces summary statistics 
df.describe()

# Create non Numeric Series 
Ser_1 = pd.Series(['a', 'a', 'b', 'c'] * 4)
Ser_1
# Applied to Non Numeric data : different type of summary is provided 
Ser_1.describe()

#------------------------------------------------------------
# Correlation - Studies ------------------------------------- 
#------------------------------------------------------------
# Create a Series 
Ser1 = pd.Series([10,12,27,35,42])
# Create a Perfectly correlated Series to Ser1
Ser2 = Ser1 + 10 

# Create a Not so perfectly correlated series this time 
len1 = Ser1.size 
Ser3 = pd.Series()
for i in range(len1):
   Ser3 = Ser3.set_value(i,np.random.randint(0,5))
# Initialize New Series 
Ser4 = pd.Series()   
# Series 4 is NOT perfectly correlated with Series 1 
Ser4 = Ser1 + Ser3    

# Check Correlation between Series 1 and Series 2  
Ser1.corr(Ser2)

# Check Correlation between Series 1 and Series 4
Ser1.corr(Ser4)

dfnew= pd.DataFrame()
#now Create a Data Frame using all the Series we have 
dfnew = pd.concat([Ser1,Ser2,Ser3,Ser4],axis=1,keys =['S1','S2','S3','S4'])
# Display combined  DataFrame
dfnew

#Correl on the DF returns a Corr Matrix 
dfnew.corr()

#------------------------------------------------------------
# Unique Values 
#-----------------------------------------------------------
# Create a Series with Duplicate values 
Ser1 = pd.Series(['a','b','c','c','e','f','d','a','e','f','c'])
# extract Unique Values
Unique_Ser1 = Ser1.unique()
Unique_Ser1
# Get Frequency Table # default sort is Desc order of freq 
Ser1_Freq = Ser1.value_counts()
Ser1_Freq
#
type(Ser1_Freq)


#----------------------------------------------------------
#-----------------  Handling Missing Data -----------------
#----------------------------------------------------------
# Create String NAN 
string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
# test for Null  in Boolean form 
string_data.isnull()
#Store only NON Null string values 
No_Null_string = string_data[string_data.notnull()]
No_Null_string
# Another way to filter out Null values is 
string_data1 = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data1_Nonull = string_data1.dropna()
string_data1_Nonull

#---------------------------------------------------------------
#  Nulls with respect to dataframes 
#---------------------------------------------------------------
from numpy import nan as NA
# Create DF with NA's 
DFnew = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
                 [NA, NA, NA], [NA, 6.5, 3.]])
DFnew    
# Drop ALL Rows which has a missing value in ANY column 
DFnew_Clean = DFnew.dropna()
DFnew_Clean
# Drop ONLY those rows which have ALL cols as NA's 
DFnew_Semi_Clean = DFnew.dropna(how='all')
DFnew_Semi_Clean
# Create a Column of only NA's 
DFnew[3] = NA
DFnew
# Drop That Col which has ALL rows as NA's 
DFnew_Semi_Col_Clean =  DFnew.dropna(how='all',axis=1)
DFnew_Semi_Col_Clean

#-----------------------------------------------------------------
# Filling in Missing data     
#-----------------------------------------------------------------
# Fill missing values with Zeros 
DFnew_filled = DFnew.fillna(0)
DFnew_filled
# Using mean to fill up NA values in a Series 
Ser1 = pd.Series([10,10,10,NA,NA])
Ser1_Mean_fill = Ser1.fillna(Ser1.mean())
Ser1_Mean_fill
#------------------------------------------------------------------
# Hierarchial Indexing 
#------------------------------------------------------------------
# CreateIndex
Ser1 = pd.Series(np.random.randn(10),
       index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
         [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
    
Ser1
# Access with Outer Level 
Ser1['b']
# Access with Inner level    
Ser1[:,3]    
#-------------------------------------------------------------------
#  Using a DF 's columns 
#--------------------------------------------------------------------
DF1 = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                'd': [0, 1, 2, 0, 1, 2, 3]})
# Display 
DF1

# Set Index to one or more columns 
DF2 = DF1.set_index(['c', 'd'])
# Display 
DF2
# Reset index just does the reverse
DF3 = DF2.reset_index()
# Display 
DF3
#------------------------------------------------------------------------------------
######################################################################################################
########################################################################################################