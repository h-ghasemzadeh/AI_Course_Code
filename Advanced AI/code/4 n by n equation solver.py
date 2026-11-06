import numpy as np

# 1. خواندن ماتریس ضرایب n در n
def read_matrix(n):
    print(f"Enter the {n}x{n} coefficient matrix row by row:")
    matrix = []
    for i in range(n):
        row = list(map(float, input(f"Enter row {i+1}: ").split()))
        matrix.append(row)
    return np.array(matrix)

# 2. خواندن بردار نیروها
def read_vector(n):
    print(f"Enter the {n}-dimensional force vector:")
    vector = list(map(float, input().split()))
    return np.array(vector)

# 3. محاسبه دترمینان ماتریس
def calculate_determinant(matrix):
    return np.linalg.det(matrix)

# 4. محاسبه معکوس ماتریس در صورت غیر صفر بودن دترمینان
def calculate_inverse(matrix, determinant):
    if determinant != 0:
        return np.linalg.inv(matrix)
    else:
        print("The determinant is zero, the matrix is singular and has no inverse.")
        return None

# 5. حل دستگاه معادله خطی
def solve_system(matrix, vector):
    try:
        solution = np.linalg.solve(matrix, vector)
        return solution
    except np.linalg.LinAlgError:
        print("The system cannot be solved.")
        return None

# Main function to run the solver
def main():
    n = int(input("Enter the number of equations (n): "))

    # خواندن ماتریس ضرایب و بردار نیروها
    matrix = read_matrix(n)
    vector = read_vector(n)

    # محاسبه دترمینان
    determinant = calculate_determinant(matrix)
    print(f"Determinant of the matrix: {determinant}")

    # محاسبه معکوس در صورت امکان
    inverse_matrix = calculate_inverse(matrix, determinant)
    if inverse_matrix is not None:
        print("Inverse of the matrix:")
        print(inverse_matrix)

        # حل دستگاه معادله
        solution = solve_system(matrix, vector)
        if solution is not None:
            print("Solution to the system of equations:")
            print(solution)

if __name__ == "__main__":
    main()
