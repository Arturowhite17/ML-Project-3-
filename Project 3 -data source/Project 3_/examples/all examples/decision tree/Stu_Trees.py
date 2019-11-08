from sklearn import tree
import pandas as pd
import os

df = pd.read_csv(os.path.join("..", "Resources", "diabetes.csv"))
df.head()

target = df["Outcome"]
target_names = ["negative", "positive"]

data = df.drop("Outcome", axis=1)
feature_names = data.columns
data.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf = rf.fit(X_train, y_train)
rf.score(X_test, y_test)

sorted(zip(rf.feature_importances_, feature_names), reverse=True)