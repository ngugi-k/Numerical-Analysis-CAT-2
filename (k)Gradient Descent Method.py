import numpy as np

def gradient_descent(learning_rate=0.1, initial_guess=(0, 0), max_iterations=1000, tolerance=1e-6):
    def f(x, y):
        return x**2 + y**2 - x*y + x - y + 1

    def gradient_f(x, y):
        df_dx = 2*x - y + 1
        df_dy = 2*y - x - 1
        return np.array([df_dx, df_dy])

    x, y = initial_guess
    for i in range(max_iterations):
        grad = gradient_f(x, y)
        new_x = x - learning_rate * grad[0]
        new_y = y - learning_rate * grad[1]
        
        if np.sqrt((new_x - x)**2 + (new_y - y)**2) < tolerance:
            break
        
        x, y = new_x, new_y

    return (x, y), f(x, y), i+1

# Run the gradient descent
result, min_value, iterations = gradient_descent()

print(f"Minimum found at: x = {result[0]:.6f}, y = {result[1]:.6f}")
print(f"Minimum value: f(x,y) = {min_value:.6f}")
print(f"Number of iterations: {iterations}")