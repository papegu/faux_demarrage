import matplotlib.pyplot as plt

# Data
years = list(range(2000, 2018))
ssrn = [4.83, -2.83, -47.83, 80.08, -29.08, 17.75, 3.92, -19.75, 52.08, 32.67,
        90.08, -29.75, 103, 24.25, 50.42, 104, 88, 49.08]

# Plot
plt.figure(figsize=(10,5))
plt.plot(years, ssrn, marker='o', linestyle='-', color='b', label='SSRN')

# Customization
plt.title("Evolution of the Standardized Precipitation Index (SSRN) in the Sahel Region")
plt.xlabel("Year")
plt.ylabel("SSRN Index")
plt.grid(True)
plt.axhline(0, color='grey', linestyle='--')  # zero reference line
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
