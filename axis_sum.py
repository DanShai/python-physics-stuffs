import numpy as np


a = np.arange(8).reshape((2,2,2))
print (a)
b = a.sum(axis=0)
print ('b', b)
c = a.sum(axis=1)
print ('c',c)
d = a.sum(axis=2)
print ('d', d)

'''
a (2,2,2) => 2 blocks of 2x2
a =[
            [[0 1]
            [2 3]]

            [[4 5]
            [6 7]]
    ]
axis 0 : 
        b = [[ 4  6]    block1 + block2 => sum of all blocks 
            [ 8 10]]
axis 1 :
        c = [[ 2  4]    rowi = sum of bi_rows 
            [10 12]]
axis 2 :
        d = [[ 1  5]    rowi = sum of bi_cols then transpose
            [ 9 13]]

'''

a = np.arange(6).reshape((2,3))
print ('a',a)
b = a.sum(axis=0)
print ('b', b)
c = a.sum(axis=1)
print ('c',c)
d = a.sum(axis=-1)
print ('d', d)

'''
a = [
        [0 1 2]
        [3 4 5]
    ]
axis 0:
b  =    [3 5 7]  look down :    sum of rows
axis 1:
c =     [ 3 12]  look right:   sum of cols then transpose
axis -1:
d =     [ 3 12]    last axis same as 1

'''