import os
os.system("cls")
import numpy as np

# Creating a 3x3 matrix
matrix = np.array([[6, 1, 8], [7, 5, 3],[2,9,4]]) 

for i in matrix:
    print(i)

print('****************************************')

row_sum=np.sum(matrix, axis=1) # axis=0 is minig sum of row
print(f'row sum is : {row_sum}')

print('****************************************')

col_sum=np.sum(matrix, axis=0) # axis=0 is minig sum of col
print(f'col sum is : {col_sum}')

print('****************************************')

main_diag=np.trace(matrix) # main_diag is (Ghoter asli)
print(f'main diag is : {main_diag}')


print('****************************************')

second_diag=np.trace(np.fliplr(matrix)) # main_diag is (Ghoter asli)
print(f'second diag is : {second_diag}')

print('****************************************')

if all(row_sum==col_sum):
    if main_diag==second_diag:
        print('***moraba jadooi (((AST)))***')
    else:
        print('***moraba jadooi (((NIST)))***')