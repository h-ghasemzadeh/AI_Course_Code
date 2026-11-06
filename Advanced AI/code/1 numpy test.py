

import numpy as np
    
a = 5
b = 6
c = 0
p = 0
pp = 0
list_int=[]
list_intc=[]
y=[]
z= 0
list_class_1 = [1, 2, 3]


    
for i in range(1, 5):
    b , a = b+a , b
    print(f'a is {a}')
    print(f'b is {b}')
    
    for j in range(1, 2):
        c += 1
        print(f'c is {c}')
        
matrix_A = np.array([[1, 2], [3, 4]])

matrix_B = np.array([[5, 6], [7, 8]])

# Element-wise addition

result_addition = matrix_A + matrix_B

# Element-wise multiplication

result_multiplication = matrix_A * matrix_B
re_reverse = np.linalg.inv(matrix_A)

for i in matrix_A:
    x=sum(i)
    list_int.append(x)
if len(set(list_int)) == 1 :
    print ('row is ok')
else : 
    print ('row is not ok')
for i in matrix_A:
    for j in i:
        y = sum(i[j])
        list_intc.append(y)
        
    if len(set(list_intc)) == 1 :
      print ('column is ok')
    else : 
      print ('column is not ok')      
        
"""       
    for m in range(0, 2):
        p += matrix_A[k][m]
    pp = p
    pp += pp 
    if pp == 2*p : 
       print('column is ok')
    else:
       print('column is not ok')
"""
n = input('enter matrix dimension: ')