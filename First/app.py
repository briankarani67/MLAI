import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

digits = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Digit: {digits.target[i]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_digits.png')  # ← This will appear on GitHub
plt.show()


X = digits.data  
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)

print("Training the model...")
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print(f"\n✅ Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

predictions = model.predict(X_test)


print("\nClassification Report:")
print(classification_report(y_test, predictions))


conf_matrix = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(conf_matrix, cmap='Blues')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')

for i in range(10):
    for j in range(10):
        ax.text(j, i, conf_matrix[i, j], ha='center', va='center')
plt.colorbar(im)
plt.savefig('confusion_matrix.png')  
plt.show()

print("\n🎉 Done! Check sample_digits.png and confusion_matrix.png")