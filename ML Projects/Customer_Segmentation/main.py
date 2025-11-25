import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Age':[25,34,45,23,51,62,43,36],
    'AnnualIncome':[30000,40000,50000,25000,60000,75000,52000,43000]
})

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data)

print(data)
