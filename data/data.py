import pandas as pd

train_data = pd.read_csv("data/train.csv")
test_labeled_data = pd.read_csv("data/test_labeled.csv")
test_unlabeled_data = pd.read_csv("data/test-unlabeled.csv")

print(train_data.head())  # Check if data loaded correctly
