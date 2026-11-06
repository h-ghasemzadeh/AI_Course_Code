# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:11:24 2024

@author: Hasan Ghasemzadeh
Magic matrix recognization
"""

import numpy as np
maindia = 0
subdia = 0

msum = []

n = int(input('enter matrix dimension: '))
m = np.zeros( (n, n) )
 
for i in range(0, n):
    for j in range(0, n):
        m [i] [j] = int(input(f'for i = {i} and j = {j} enter m[i,j]: '))
print(m)

''' just for vscode the line below is necessary'''
np.set_printoptions(legacy='1.25')

for i in range(0,n):
    
    msum.append(sum(m[i]))
    msum.append(sum(m[:,i]))
    
    maindia += m [i] [i]
    subdia += m [i] [(len(m)-1-i)]
    
msum.append(maindia)
msum.append(subdia)

print(msum)

if len(set(msum)) != 1 :
   print('It is not magic')
   
else:
    print(f'congradulation it is magic Diameter sum={set(msum)}')
        
    if np.linalg.det(m) != 0 :
        print('congradulation it is Realy magic')
        print(f' matrix determinant {np.linalg.det(m)}')
        print(f' matrix invers {np.linalg.inv(m)}')
