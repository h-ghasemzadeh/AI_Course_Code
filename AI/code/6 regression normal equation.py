# Using Normal Equation in regression 
import numpy as np

# تعریف ماتریس X
X = np.array([
    [1, 210.4, 5, 1, 45],
    [1, 141.6, 3, 2, 40],
    [1, 153.4, 3, 2, 30],
    [1, 85.2, 2, 1, 36],
    [1, 300, 4, 1, 38],
    [1, 280, 5, 3, 20]
])
rank_X = np.linalg.matrix_rank(X)
print("Ranx X=", rank_X )
# محاسبه ترنسپوز ماتریس X
X_transpose = X.T

# نمایش نتیجه
print("ماتریس اصلی:")
print(X)
print("ترنسپوز ماتریس:")
print(X_transpose)

A = np.dot(X_transpose,X )
print(A)

if np.linalg.det(A) != 0:
    # محاسبه معکوس ماتریس
    inverse_A = np.linalg.inv(A)
    print("معکوس ماتریس حاصل‌ضرب:")
    print(inverse_A)
else:
    print("ماتریس معکوس‌پذیر نیست.")

C = np.dot(inverse_A, X_transpose)
print(C)
y =np.array([46,23.2,31.5,17.8,54,60])

Tetha = np.dot(C,y)
print(Tetha)

x =np.array([[1],[200],[3],[2],[10]])  # can be writen as below
# x =np.array([1,200,3,2,10])          # just in one dimension 
hx = np.dot(Tetha,x)
print("h(x):",hx)

