from pathlib import Path
import joblib

MODEL_PATH = Path("models/intent_pipeline.joblib")

model = joblib.load(MODEL_PATH)

print("Insurance Intent Classifier")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        print("Please enter a question.\n")
        continue

    predicted_intent = model.predict([question])[0]

    probabilities = model.predict_proba([question])[0]
    confidence = max(probabilities)

    print("Predicted intent:", predicted_intent)
    print(f"Model confidence: {confidence:.2%}\n")