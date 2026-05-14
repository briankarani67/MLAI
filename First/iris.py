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