import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, confusion_matrix

data = pd.read_csv("C:\Users\mathe\Desktop\dm\Preprocessed_data\\2015-2016_merged-csv_mod.csv")
x=data.iloc[:, 3:-1]
y=data.iloc[:,1]

testdata=pd.read_csv("C:\Users\mathe\Desktop\dm\Preprocessed_data\\testset_2017.csv")
x_test=testdata.iloc[:, 3:-1]
y_test=testdata.iloc[:,1]

rf = RandomForestRegressor(n_estimators=450)
rf.fit(x,y)
pred=rf.predict(x_test)

plt.scatter(y_test,pred)
plt.show()

# cm = confusion_matrix(y, pred)

# im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
# ax.figure.colorbar(im, ax=ax)

print r2_score(y,pred)
