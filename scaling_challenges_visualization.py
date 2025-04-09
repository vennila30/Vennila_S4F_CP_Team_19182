
import matplotlib.pyplot as plt

# Define potential challenges and their hypothetical impact scores (1-10)
challenges = {
    "Data Variability Across Centers": 9,
    "Hardware Compatibility": 7,
    "Model Generalization to New Environments": 8,
    "Internet/Network Infrastructure": 6,
    "Staff Training and Adoption": 7,
    "Cost of Deployment and Maintenance": 8,
    "Real-time Processing Constraints": 6,
    "Waste Category Ambiguity": 7
}

# Sort challenges by impact score
sorted_challenges = dict(sorted(challenges.items(), key=lambda item: item[1], reverse=True))

# Plot the challenges
plt.figure(figsize=(10, 6))
plt.barh(list(sorted_challenges.keys()), list(sorted_challenges.values()), color='green')
plt.xlabel('Impact Score (1-10)')
plt.title('Challenges in Scaling Waste Sorting System')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
