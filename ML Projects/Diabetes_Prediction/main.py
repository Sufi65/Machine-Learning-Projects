import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.DataFrame({
    'Glucose':[120,130,85,90,200,180,140,160],
    'BMI':[35,30,25,28,45,50,33,37],
    'Age':[30,40,22,25,50,55,42,48],
    'Outcome':[1,1,0,0,1,1,1,1]
})

X = data[['Glucose','BMI','Age']]
y = data['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,pred))
