import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int') # converting into numbers
X = X.astype('int') # converting into numbers 
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train) # give training set as input 

inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]

b = log_reg.predict_proba(final)


pickle.dump(log_reg,open('model.pkl','wb')) # import it into a pickle file so you're not running the model everytime the server loads
model=pickle.load(open('model.pkl','rb')) # reboot the model