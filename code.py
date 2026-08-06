

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("dataset.csv")


data.columns = data.columns.str.strip()


print("Columns in dataset:", data.columns.tolist())


if "Age" not in data.columns or "Glucose" not in data.columns:
    print("\nError: Required columns 'Age' and 'Glucose' were not found.")
    print("Available columns are:", data.columns.tolist())
    exit()

age = data["Age"].to_numpy()
glucose = data["Glucose"].to_numpy()


plt.figure(figsize=(8,6))
plt.scatter(age, glucose, color='blue', s=70, label='Original Data')
plt.xlabel("Age (X)")
plt.ylabel("Glucose Level (Y)")
plt.title("Age vs Glucose Level")
plt.grid(True)
plt.legend()
plt.show()



mean_x = np.mean(age)
mean_y = np.mean(glucose)


b1 = np.sum((age - mean_x) * (glucose - mean_y)) / np.sum((age - mean_x) ** 2)
b0 = mean_y - b1 * mean_x

print("\nRegression Coefficients")
print("-----------------------")
print("Intercept (b0):", round(b0, 4))
print("Slope (b1):", round(b1, 4))


print("-------------------")
print(f"Y = {b0:.4f} + ({b1:.4f})X")

predicted = b0 + b1 * age


sorted_index = np.argsort(age)
age_sorted = age[sorted_index]
predicted_sorted = predicted[sorted_index]

plt.figure(figsize=(8,6))
plt.scatter(age, glucose, color='blue', s=70, label='Original Data')
plt.plot(age_sorted, predicted_sorted, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Age (X)")
plt.ylabel("Glucose Level (Y)")
plt.title("Linear Regression: Age vs Glucose Level")
plt.grid(True)
plt.legend()
plt.show()



print("\nInterpretation")
print("--------------")

if b1 > 0:
    print("The slope is positive.")
    print("As age increases, glucose level tends to increase.")
elif b1 < 0:
    print("The slope is negative.")
    print("As age increases, glucose level tends to decrease.")
else:
    print("No linear relationship exists between age and glucose level.")


new_age = 55
predicted_glucose = b0 + b1 * new_age

print("\nPrediction")
print("----------")
print(f"Predicted Glucose Level for Age {new_age} = {predicted_glucose:.2f}")
