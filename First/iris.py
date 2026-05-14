

# IMPORT TOOLS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Different from yesterday!
from sklearn.metrics import accuracy_score
import joblib  
import os

# LOAD DATA
iris = load_iris()
print("\n IRIS DATASET INFO ")
print(f"Flower measurements: {iris.feature_names}")
print(f"Flower species: {iris.target_names}")
print(f"Total samples: {len(iris.data)}")


X = iris.data      
y = iris.target   

print(f"\nFirst 5 flowers:")
for i in range(5):
    print(f"  Flower {i+1}: {X[i]} → Species {y[i]} ({iris.target_names[y[i]]})")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining on {len(X_train)} flowers")
print(f"Testing on {len(X_test)} flowers")


model = DecisionTreeClassifier(max_depth=3, random_state=42)
# max_depth=3 means tree asks up to 3 questions before deciding

print("\n Training the decision tree...")
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n Test accuracy: {accuracy*100:.2f}%")


print("\n Detailed results:")
for i in range(len(X_test)):
    true_species = iris.target_names[y_test[i]]
    pred_species = iris.target_names[predictions[i]]
    if true_species != pred_species:
        print(f"   Flower {i+1}: Was {true_species}, but AI said {pred_species}")
    else:
        print(f"  Flower {i+1}: {true_species} (correct)")

print("\n Basic classifier done!")


print("\n Creating visualizations...")

# Create a figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


colors = ['red', 'green', 'blue']
species_names = iris.target_names

for species_id in range(3):
    # Get indices where species matches
    indices = y == species_id
    # Plot petal length (x) vs petal width (y)
    ax1.scatter(X[indices, 2], X[indices, 3],  
                c=colors[species_id], 
                label=species_names[species_id],
                alpha=0.7, s=60)  

ax1.set_xlabel('Petal length (cm)')
ax1.set_ylabel('Petal width (cm)')
ax1.set_title('Iris Species by Petal Size')
ax1.legend()
ax1.grid(True, alpha=0.3)



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

print("Visualizations saved as 'iris_visualization.png'")



print(" IRIS FLOWER PREDICTOR")


while True:
    print("\nEnter flower measurements (or 'quit' to exit):")
    
    try:
       
        sepal_len = input("  Sepal length (cm) [4.0-8.0]: ")
        if sepal_len.lower() == 'quit':
            break
            
        sepal_wid = input("  Sepal width (cm) [2.0-4.5]: ")
        if sepal_wid.lower() == 'quit':
            break
            
        petal_len = input("  Petal length (cm) [1.0-7.0]: ")
        if petal_len.lower() == 'quit':
            break
            
        petal_wid = input("  Petal width (cm) [0.1-2.5]: ")
        if petal_wid.lower() == 'quit':
            break
        
        # Convert to numbers
        measurements = [[float(sepal_len), float(sepal_wid), 
                        float(petal_len), float(petal_wid)]]
        
        # Make prediction
        predicted_id = model.predict(measurements)[0]
        predicted_species = iris.target_names[predicted_id]
        
        # Get prediction confidence (probability)
        probabilities = model.predict_proba(measurements)[0]
        confidence = max(probabilities) * 100
        
        print(f"\n AI Prediction: {predicted_species.upper()}")
        print(f"   Confidence: {confidence:.1f}%")
        
       
        print("   Breakdown:")
        for i, species in enumerate(iris.target_names):
            bar = "█" * int(probabilities[i] * 20)
            print(f"     {species:12s}: {bar:20s} {probabilities[i]*100:.1f}%")
        
        
        if confidence < 70:
            print("\n   Low confidence! Measurements might be between species.")
            
    except ValueError:
        print Please enter valid numbers!")
    except KeyboardInterrupt:
        break

print("\n Thanks for using Iris Classifier!")


save_model = input("\n Save this model for later? (yes/no): ")

if save_model.lower() == 'yes':
    model_filename = 'iris_model.joblib'
    joblib.dump(model, model_filename)
    print(f"✅ Model saved as '{model_filename}'")
    
    
    import pickle
    with open('iris_info.pkl', 'wb') as f:
        pickle.dump({'feature_names': iris.feature_names,
                    'target_names': iris.target_names}, f)
    print(" Info saved as 'iris_info.pkl'")

# Show how to load it later
print("\n To load this model in the future:")
print("  import joblib")
print("  model = joblib.load('iris_model.joblib')")