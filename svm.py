import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.svm import SVR # "Support Vector Classifier" 
from sklearn.preprocessing import StandardScaler

clf = SVR(kernel='linear') 
  
# reading csv file and extracting class column to y. 
d = pd.read_csv("C:\Users\mathe\Desktop\dm\Preprocessed_data\\2015-2016_merged-csv_mod.csv") 
#a = np.array(d) 

x=d.iloc[:, 3:-1]
y=d.iloc[:,1]

#x.shape  
#print (x),(y) 

scaler = StandardScaler()

x1 = scaler.fit(x)

clf = SVR(kernel='linear') 
  
# fitting x samples and y classes 
clf.fit(x1, y) 
print clf.predict(x1)

plt.scatter(y,clf.predict(x1))
plt.show()