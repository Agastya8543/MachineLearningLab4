import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

dataframe=pd.read_excel("embeddingsdata.xlsx")

model_entropy = DecisionTreeClassifier(criterion="entropy")

binary_dataframe = dataframe[dataframe['Label'].isin([0, 1])]
X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_entropy.fit(X_train, y_train)

train_accuracy = model_entropy.score(X_train, y_train)

test_accuracy = model_entropy.score(X_test, y_test)

print(f"Training Set Accuracy (criterion='entropy'): {train_accuracy}")
print(f"Test Set Accuracy (criterion='entropy'): {test_accuracy}")

plt.figure(figsize=(20, 10))
plot_tree(model_entropy, filled=True, feature_names=['embed_0', 'embed_1'], class_names=['No', 'Yes'], rounded=True)
plt.show()
