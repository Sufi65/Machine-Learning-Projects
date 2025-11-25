import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Synthetic dataset
data = pd.DataFrame({
    'Income': [3000,4000,5000,6000,2000,2500,7000,8000],
    'Age': [25,35,45,52,23,29,48,57],
    'LoanAmount': [200,150,300,250,100,120,350,400],
    'Eligible': [1,1,1,1,0,0,1,1]
})

X = data[['Income','Age','LoanAmount']]
y = data['Eligible']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,pred))
