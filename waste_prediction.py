
# Simple Linear Regression Without sklearn for Waste Management Prediction

# Sample data: days and corresponding waste level in kg
days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
waste_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Step 1: Calculate means
n = len(days)
mean_x = sum(days) / n
mean_y = sum(waste_levels) / n

# Step 2: Calculate slope (m) and intercept (c) of y = mx + c
numerator = sum((days[i] - mean_x) * (waste_levels[i] - mean_y) for i in range(n))
denominator = sum((days[i] - mean_x) ** 2 for i in range(n))

m = numerator / denominator
c = mean_y - m * mean_x

# Step 3: Predict waste level for a future day
future_day = 12
predicted_waste = m * future_day + c

print(f"Predicted waste level on day {future_day}: {predicted_waste:.2f} kg")
