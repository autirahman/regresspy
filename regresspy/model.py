from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_iris
from regresspy.regression import Regression
from regresspy.loss import rmse

iris = load_iris()
# We will use sepal length to predict sepal width
X = iris.data[:, 0].reshape(-1, 1)
Y = iris.data[:, 1].reshape(-1, 1)

stc_grd_dcnt = SGDRegressor(max_iter= 100, learning_rate= 'constant', eta0= 0.001)
stc_grd_dcnt.fit(X, Y.reshape(-1))
sto_chas_grad_prdct = stc_grd_dcnt.predict(X)
sto_rmse = rmse(sto_chas_grad_prdct, Y)
print('Stochastic Gradient Descent Regressor RMSE value:', str(sto_rmse))

rg_value = Regression(epochs= 100, learning_rate= 0.0001)
rg_value.fit(X, Y)
reg_prediction = rg_value.predict(X)
rg_rmse = rg_value.score(reg_prediction, Y)
print('RMSE value of class: ', str(rg_rmse))