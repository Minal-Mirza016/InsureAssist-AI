import pandas as pd

# Load the dataset
df = pd.read_csv("training/intent_dataset.csv")

# Show the first five rows
print("\nFIRST FIVE EXAMPLES")
print(df.head())

# Show dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Count examples in each category
print("\nEXAMPLES PER INTENT")
print(df["intent"].value_counts())

# Check for missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Check for duplicate questions
print("\nDUPLICATE QUESTIONS")
print(df["text"].duplicated().sum())