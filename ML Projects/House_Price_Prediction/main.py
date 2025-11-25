import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    'Area':[800,1200,1500,600,2000,2500,1800,1600],
    'Bedrooms':[2,3,3,1,4,4,3,3],
    'Age':[10,5,8,20,2,1,4,6],
    'Price':[100000,150000,180000,80000,250000,300000,220000,200000]
})

X = data[['Area','Bedrooms','Age']]
y = data['Price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

print("Predictions:", model.predict(X_test))
