import numpy as np
c = 0
d = 0
m = []
l1 = []
'''
n = int(input('enter matrix dimension: '))

for i in range(0, n):
    for j in range(0, n):
        m.append(int(input(f'for i = {i} and j = {j} enter m[i,j]: ')))
print(m)
'''

A = np.array([[1, 1], [1, 1]])


print("A[:,0] =",A[:,0]) # First Column

l1.append(sum(A[:,0]))
l1.append(sum(A[:,1]))

for i in range(0,2):
    
    l1.append(sum(A[i]))

print(l1)

if len(set(l1)) != 1 :
   print('not ok for row and column')
   
else:
    for i in range(len(A)):
        
        c += A [i] [i]
        d += A [i] [(len(A)-1-i)]
        
        
    l1.append(c)
    l1.append(d)
 
if len(set(l1)) == 1 :
   print('congradulation it is magic')
   
else: 
    print('it is not magic')
    