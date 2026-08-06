import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Salary_Data.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Display available columns
print("Columns in dataset:", data.columns.tolist())

# Check if required columns exist
if "Experience" not in data.columns or "Salary" not in data.columns:
    print("\nError: Required columns 'Experience' and 'Salary' were not found.")
    print("Available columns are:", data.columns.tolist())
    exit()

# Extract independent and dependent variables
experience = data["Experience"].to_numpy()
salary = data["Salary"].to_numpy()

# Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(experience, salary, color='blue', s=70, label='Original Data')
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.grid(True)
plt.legend()
plt.show()

# Calculate Means
mean_x = np.mean(experience)
mean_y = np.mean(salary)

# Calculate Regression Coefficients
b1 = np.sum((experience - mean_x) * (salary - mean_y)) / np.sum((experience - mean_x) ** 2)
b0 = mean_y - b1 * mean_x

# Print Regression Equation
print("\nRegression Coefficients")
print("-----------------------")
print("Intercept (b0):", round(b0, 4))
print("Slope (b1):", round(b1, 4))

print("\nRegression Equation")
print("-------------------")
print(f"Salary = {b0:.4f} + ({b1:.4f}) × Experience")

# Predicted values
predicted = b0 + b1 * experience

# Sort values for regression line
sorted_index = np.argsort(experience)
experience_sorted = experience[sorted_index]
predicted_sorted = predicted[sorted_index]

# Plot Regression Line
plt.figure(figsize=(8,6))
plt.scatter(experience, salary, color='blue', s=70, label='Original Data')
plt.plot(experience_sorted, predicted_sorted, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Linear Regression: Experience vs Salary")
plt.grid(True)
plt.legend()
plt.show()

# Interpretation
print("\nInterpretation")
print("--------------")

if b1 > 0:
    print("The slope is positive.")
    print("As experience increases, salary tends to increase.")
elif b1 < 0:
    print("The slope is negative.")
    print("As experience increases, salary tends to decrease.")
else:
    print("No linear relationship exists between experience and salary.")

# Prediction for a new experience value
new_experience = 3.5
predicted_salary = b0 + b1 * new_experience

print("\nPrediction")
print("----------")
print(f"Predicted Salary for {new_experience} years of experience = {predicted_salary:.2f}")