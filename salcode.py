
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_Data.csv")

# -----------------------------
# Outlier Detection using IQR
# -----------------------------

print("\nOutlier Detection (IQR Method)")
print("------------------------------")

for column in ["Experience", "Salary"]:

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    print(f"\nColumn: {column}")
    print(f"Lower Bound = {lower_bound:.2f}")
    print(f"Upper Bound = {upper_bound:.2f}")
    print(f"Number of Outliers = {len(outliers)}")

    if len(outliers) > 0:
        print(outliers[[column]])
    else:
        print("No outliers found.")

data.columns = data.columns.str.strip()

print("Columns in dataset:", data.columns.tolist())

if "Experience" not in data.columns or "Salary" not in data.columns:
    print("\nError: Required columns 'Experience' and 'Salary' were not found.")
    print("Available columns are:", data.columns.tolist())
    exit()

experience = data["Experience"].to_numpy()
salary = data["Salary"].to_numpy()

plt.figure(figsize=(8,6))
plt.scatter(experience, salary, color='blue', s=70, label='Original Data')
plt.xlabel("Experience (X)")
plt.ylabel("Salary (Y)")
plt.title("Experience vs Salary")
plt.grid(True)
plt.legend()
plt.show()

mean_x = np.mean(experience)
mean_y = np.mean(salary)

b1 = np.sum((experience - mean_x) * (salary - mean_y)) / np.sum((experience - mean_x) ** 2)
b0 = mean_y - b1 * mean_x

print("\nRegression Coefficients")
print("-----------------------")
print("Intercept (b0):", round(b0, 4))
print("Slope (b1):", round(b1, 4))

print("-------------------")
print(f"Y = {b0:.4f} + ({b1:.4f})X")

predicted = b0 + b1 * experience

sorted_index = np.argsort(experience)
experience_sorted = experience[sorted_index]
predicted_sorted = predicted[sorted_index]

plt.figure(figsize=(8,6))
plt.scatter(experience, salary, color='blue', s=70, label='Original Data')
plt.plot(experience_sorted, predicted_sorted, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Experience (X)")
plt.ylabel("Salary (Y)")
plt.title("Linear Regression: Experience vs Salary")
plt.grid(True)
plt.legend()
plt.show()

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

new_experience = 5
predicted_salary = b0 + b1 * new_experience

print("\nPrediction")
print("----------")
print(f"Predicted Salary for Experience {new_experience} = {predicted_salary:.2f}")



































