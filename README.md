# Expresso Churn Prediction - ML Prediction App.
# 🎯 What You're Aiming For.
This project demonstrates how to predict which clients are most likely to churn using the 'Expresso churn' dataset provided as part of the Expresso Churn Prediction Challenge hosted by Zindi platform.

➡️ Dataset Link.
[Dataset Link](https://drive.google.com/file/d/12_KUHr5NlHO_6bN5SylpkxWc-JvpJNWe/view)

➡️ Columns Explanation.
[Columns Explanation Link](https://drive.google.com/file/d/13hpZ51H7Ty9pKpRSrBvXC1OSDLrF2jHB/view)

# ℹ️ Instructions.
1. Install the necessary packages:
2. Import your data and perform basic data exploration:
3. Display general information about the dataset:
4. Create a pandas profiling report to gain insights into the dataset:
5. Handle missing and corrupted values:
6. Remove duplicates, if they exist:
7. Handle outliers, if they exist:
8. Encode categorical features:
9. Based on the previous data exploration, train and test a machine learning classifier:
10. Create a Streamlit application (locally) and add input fields for your features and a validation button at the end of the form:
11. Deploy your application on Streamlit share:
  - Create a GitHub and a Streamlit Share account.
  - Create a new git repository.
  - Upload your local code to the newly created git repository.
  - Log in to your Streamlit account and deploy your application from the git repository.

# 🛠️ Tools Used.
- Anaconda
- Jupyter Nootbook.
- Python.
- VSCode.
- GitHub.
- Google Chrome.
- Streamlit.

## Example Code.
pip install pandas scikit-learn streamlit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

```
data = pd.read_csv('path_to_your_dataset.csv')
print(data.head())
print(data.info())
le = LabelEncoder()
data['encoded_column'] = le.fit_transform(data['categorical_column'])
features = data.drop('target_column', axis=1)
label = data['target_column']
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
