import numpy as np

# 1. تجزیه LU (پایین مثلثی L و بالا مثلثی U)
def lu_decomposition(A, n):
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # U matrix calculation
        for k in range(i, n):
            sum_U = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_U

        # L matrix calculation
        for k in range(i, n):
            if i == k:
                L[i][i] = 1  # Diagonal as 1 for L
            else:
                sum_L = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - sum_L) / U[i][i]
    
    return L, U

# 2. پیش‌حل (Forward Substitution) برای حل L * y = b
def forward_substitution(L, b, n):
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y

# 3. پس‌حل (Backward Substitution) برای حل U * x = y
def backward_substitution(U, y, n):
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    return x

# Main function to solve the system of equations
def main():
    n = int(input("Enter the number of equations (n): "))

    # دریافت ماتریس ضرایب و بردار نیروها
    A = []
    print(f"Enter the {n}x{n} coefficient matrix row by row:")
    for i in range(n):
        row = list(map(float, input(f"Enter row {i+1}: ").split()))
        A.append(row)
    A = np.array(A)

    b = list(map(float, input("Enter the force vector: ").split()))
    b = np.array(b)

    # تجزیه LU
    L, U = lu_decomposition(A, n)
    
    print("Lower Triangular Matrix L:")
    print(L)
    print("Upper Triangular Matrix U:")
    print(U)

    # پیش‌حل: حل L * y = b
    y = forward_substitution(L, b, n)

    # پس‌حل: حل U * x = y
    x = backward_substitution(U, y, n)

    print("Solution vector x:")
    print(x)

if __name__ == "__main__":
    main()
