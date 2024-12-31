import numpy as np

def is_diagonally_dominant(matrix):
    """Check if the matrix is diagonally dominant."""
    n = len(matrix)
    for i in range(n):
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if i != j)
        if abs(matrix[i][i]) <= row_sum:
            return False
    return True


def make_diagonally_dominant(matrix, vector):
    """Attempt to make the matrix diagonally dominant by row swapping."""
    n = len(matrix)
    for _ in range(n):
        for i in range(n):
            for j in range(i + 1, n):
                new_matrix = matrix.copy()
                new_vector = vector.copy()
                new_matrix[[i, j]] = new_matrix[[j, i]]
                new_vector[[i, j]] = new_vector[[j, i]]
                if is_diagonally_dominant(new_matrix):
                    return new_matrix, new_vector
    return matrix, vector


def jacobi_method(matrix, vector, tolerance=1e-5, max_iterations=100):
    n = len(matrix)
    x = np.zeros(n)
    prev_x = x.copy()

    if not is_diagonally_dominant(matrix):
        matrix, vector = make_diagonally_dominant(matrix, vector)
        if not is_diagonally_dominant(matrix):
            print("The matrix is not diagonally dominant and cannot be made so. Results may not converge.")

    print("Jacobi Method:")
    for iteration in range(1, max_iterations + 1):
        for i in range(n):
            sum_ = sum(matrix[i][j] * prev_x[j] for j in range(n) if j != i)
            x[i] = (vector[i] - sum_) / matrix[i][i]
        print(f"Iteration {iteration}: {x}")
        if np.linalg.norm(x - prev_x, ord=np.inf) < tolerance:
            print(f"Converged in {iteration} iterations.")
            return x
        prev_x = x.copy()

    print("The system did not converge.")
    return x


def gauss_seidel_method(matrix, vector, tolerance=1e-5, max_iterations=100):
    n = len(matrix)
    x = np.zeros(n)

    if not is_diagonally_dominant(matrix):
        matrix, vector = make_diagonally_dominant(matrix, vector)
        if not is_diagonally_dominant(matrix):
            print("The matrix is not diagonally dominant and cannot be made so. Results may not converge.")

    print("Gauss-Seidel Method:")
    for iteration in range(1, max_iterations + 1):
        x_new = x.copy()
        for i in range(n):
            sum_ = sum(matrix[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (vector[i] - sum_) / matrix[i][i]
        print(f"Iteration {iteration}: {x_new}")
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            print(f"Converged in {iteration} iterations.")
            return x_new
        x = x_new

    print("The system did not converge.")
    return x


def main():
    matrix_a = np.array([[4, 2, 0], [2, 10, 4], [0, 4, 5]], dtype=float)
    vector_b = np.array([2, 6, 5], dtype=float)

    print("Choose a method to solve the system:")
    print("1. Jacobi Method")
    print("2. Gauss-Seidel Method")
    choice = int(input("Enter your choice (1 or 2): "))

    if choice == 1:
        solution = jacobi_method(matrix_a, vector_b)
    elif choice == 2:
        solution = gauss_seidel_method(matrix_a, vector_b)
    else:
        print("Invalid choice.")
        return

    print("\nSolution:", solution)


if __name__ == "__main__":
    main()
