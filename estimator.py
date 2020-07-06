import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("time_series_covid_19_confirmed.csv")

days = data[data["Province/State"] == "Hubei"]

days = days.drop(columns=[
    "Province/State",
    "Country/Region",
    "Lat",
    "Long"
])
x = np.asanyarray(days) 
x = np.squeeze(x)

plt.plot(x)
plt.xlabel('Tiempo')
plt.ylabel('Contagios')
plt.title('Razón de contagios')
plt.show()
p = 3
plt.scatter(x[p:],x[:-p])
plt.title('autocorrelacion')
plt.show()

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(x)
plt.show()

dat = pd.DataFrame({"Contagios": x})
for i in range(0, 3):
    dat = pd.concat([dat, dat.Contagios.shift(-i)], axis=1)
dat = dat[:-p]


x = np.asanyarray(dat.iloc[:, 1:])
y = np.asanyarray(dat.iloc[:, 0])

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

model = Pipeline([
    ('scaler', StandardScaler()),
    ('lin_reg', DecisionTreeRegressor(min_samples_leaf=0.0005, max_depth=3))
])

model.fit(xtrain,ytrain)

print('Train: ',model.score(xtrain,ytrain))
print('Test: ',model.score(xtest,ytest))

plt.close('all')
plt.title("Resultado aplicando Árboles de Decisión")
plt.plot(y, 'bo')
plt.plot(model.predict(x), 'r-')
plt.show()

seven_days_before  = np.linspace(x.min(), x.max(), 7)

print(seven_days_before)
print(model.predict( [seven_days_before] ))
