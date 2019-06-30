import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,confusion_matrix
data = pd.read_csv("C:\Users\mathe\Desktop\dm\\cleaned1516.csv")
data.head()
#data.info()
#data.describe()
#sdata.columns

x=data.iloc[:, 3:-1]
y=data.iloc[:,1]

print x
print y

testdata=pd.read_csv("C:\Users\mathe\Desktop\dm\Preprocessed_data\\testset_2017.csv")
x_test=testdata.iloc[:, 3:-1]
y_test=testdata.iloc[:,1]

Regobj = LinearRegression()

Regobj.fit(x,y)

prediction = Regobj.predict(x_test)

#plt.plot(x,y, color="red")
print y_test[0:5]
print (prediction) [0:5]
plt.scatter(y_test,Regobj.predict(x_test))
plt.show()

# prediction2 = Regobj.predict(x)

# print r2_score(y,prediction2)


# sbn.pairplot(data)
# plt.show()
