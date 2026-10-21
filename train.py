from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path



"""Train an Iris classifier, report its accuracy, and save it to models/."""
iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
    stratify=iris.target,
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.2%}")

model_dir = Path(__file__).resolve().parent / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "iris_model.joblib"
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")


