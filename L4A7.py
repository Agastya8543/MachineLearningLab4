import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

dataframe=pd.read_excel("embeddingsdata.xlsx")

binary_dataframe = dataframe[dataframe['Label'].isin([0, 1])]
X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

decision_tree = DecisionTreeClassifier(max_depth=5)
decision_tree.fit(X_train, y_train)

y_dt = decision_tree.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_dt)
confusion_matrix_dt = confusion_matrix(y_test, y_dt)
classification_report_dt = classification_report(y_test, y_dt)

print("Decision Tree Accuracy:", accuracy_dt)
print("Decision Tree Confusion Matrix:\n", confusion_matrix_dt)
print("Decision Tree Classification Report:\n", classification_report_dt)

random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
random_forest.fit(X_train, y_train)

y_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_rf)
confusion_matrix_rf = confusion_matrix(y_test, y_rf)
classification_report_rf = classification_report(y_test, y_rf)

print("\nRandom Forest Accuracy:", accuracy_rf)
print("Random Forest Confusion Matrix:\n", confusion_matrix_rf)
print("Random Forest Classification Report:\n", classification_report_rf)