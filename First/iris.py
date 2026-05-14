"""
DAY 2: Iris Flower Classifier
Learn: Features, training vs testing, making predictions on new flowers
"""

# 1. IMPORT TOOLS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Different from yesterday!
from sklearn.metrics import accuracy_score

# 2. LOAD DATA
iris = load_iris()
print("\n🌸 IRIS DATASET INFO 🌸")
print(f"Flower measurements: {iris.feature_names}")
print(f"Flower species: {iris.target_names}")
print(f"Total samples: {len(iris.data)}")

# 3. EXPLORE THE DATA
X = iris.data      # Measurements (150 flowers × 4 measurements)
y = iris.target    # Species (0=setosa, 1=versicolor, 2=virginica)

print(f"\nFirst 5 flowers:")
for i in range(5):
    print(f"  Flower {i+1}: {X[i]} → Species {y[i]} ({iris.target_names[y[i]]})")

# 4. SPLIT DATA (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining on {len(X_train)} flowers")
print(f"Testing on {len(X_test)} flowers")

# 5. CREATE & TRAIN MODEL (Decision Tree - easier to understand than neural net)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
# max_depth=3 means tree asks up to 3 questions before deciding

print("\n🌳 Training the decision tree...")
model.fit(X_train, y_train)

# 6. TEST THE MODEL
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n✅ Test accuracy: {accuracy*100:.2f}%")

# 7. SHOW WHERE IT WAS WRONG
print("\n🔍 Detailed results:")
for i in range(len(X_test)):
    true_species = iris.target_names[y_test[i]]
    pred_species = iris.target_names[predictions[i]]
    if true_species != pred_species:
        print(f"  ❌ Flower {i+1}: Was {true_species}, but AI said {pred_species}")
    else:
        print(f"  ✅ Flower {i+1}: {true_species} (correct)")

print("\n🎉 Basic classifier done!")

# ========== VISUALIZATION SECTION ==========
print("\n📊 Creating visualizations...")

# Create a figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- PLOT 1: Petal Length vs Petal Width (best features) ---
colors = ['red', 'green', 'blue']
species_names = iris.target_names

for species_id in range(3):
    # Get indices where species matches
    indices = y == species_id
    # Plot petal length (x) vs petal width (y)
    ax1.scatter(X[indices, 2], X[indices, 3],  # [:,2]=petal length, [:,3]=petal width
                c=colors[species_id], 
                label=species_names[species_id],
                alpha=0.7, s=60)  # alpha=transparency, s=size

ax1.set_xlabel('Petal length (cm)')
ax1.set_ylabel('Petal width (cm)')
ax1.set_title('Iris Species by Petal Size')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- PLOT 2: Show decision boundaries (how AI separates species) ---
# We'll use just petal length and petal width for visualization

# Create a grid of points
x_min, x_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
y_min, y_max = X[:, 3].min() - 0.5, X[:, 3].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Train a new model on just 2 features for visualization
from sklearn.tree import DecisionTreeClassifier
model_2d = DecisionTreeClassifier(max_depth=3, random_state=42)
model_2d.fit(X[:, [2, 3]], y)  # Train on petal length & width only

# Predict on the entire grid
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
ax2.contourf(xx, yy, Z, alpha=0.3, colors=['red', 'green', 'blue'])
# Plot the actual data points
for species_id in range(3):
    indices = y == species_id
    ax2.scatter(X[indices, 2], X[indices, 3],
                c=colors[species_id], 
                label=species_names[species_id],
                edgecolors='black', linewidth=0.5, s=60)

ax2.set_xlabel('Petal length (cm)')
ax2.set_ylabel('Petal width (cm)')
ax2.set_title('Decision Boundaries (How AI Separates Species)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_visualization.png', dpi=150)
plt.show()

print("✅ Visualizations saved as 'iris_visualization.png'")