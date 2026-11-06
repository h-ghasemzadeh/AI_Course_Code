import matplotlib.pyplot as plt

# Data from the table
x = [21, 35, 7, 33, 5, 14, 25]  # Income
y = [15, 21, 2, 13, 3, 8, 9]    # Food Expenditure

# Number of data points
N = len(x)

# Calculating products of x and y, and squares of x
xy = [x[i] * y[i] for i in range(N)]
x_squared = [x[i] ** 2 for i in range(N)]

# Calculating the sums
sum_x = sum(x)
sum_y = sum(y)
sum_xy = sum(xy)
sum_x_squared = sum(x_squared)

# Calculating the slope (B1) and intercept (B0)
B1 = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - (sum_x) ** 2)
B0 = (sum_y - B1 * sum_x) / N

# Plotting the scatter plot
plt.scatter(x, y, color="blue", label="Data points")

# Calculating the regression line values
x_values = range(min(x), max(x) + 1)
y_values = [B0 + B1 * xi for xi in x_values]

# Plotting the regression line
plt.plot(x_values, y_values, color="red", label="Regression line")

# Adding labels and title
plt.xlabel("Income")
plt.ylabel("Food Expenditure")
plt.title("Income vs. Food Expenditure with Regression Line")
plt.legend()

# Annotating the slope and intercept
plt.text(min(x), max(y) - 1, f"Slope (B1): {B1:.2f}", color="red")
plt.text(min(x), max(y) - 3, f"Intercept (B0): {B0:.2f}", color="red")

# Displaying the plot
plt.show()
