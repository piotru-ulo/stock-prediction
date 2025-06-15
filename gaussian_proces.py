import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class GaussianProcessStockPredictor:
    def __init__(self, data):
        self.data = data.copy()
        self.X = np.arange(len(data)).reshape(-1, 1)
        self.y = data['close'].values
        self.gpr = None
        self.kernel_list = [
            RBF(length_scale=5.0) + WhiteKernel(noise_level=0.5),
            RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0),
            Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=1.0),
            ConstantKernel(1.0) * RBF(length_scale=20.0) + WhiteKernel(noise_level=0.5),
            Matern(length_scale=5.0, nu=0.5) + WhiteKernel(noise_level=0.5),  
            Matern(length_scale=15.0, nu=2.5) + WhiteKernel(noise_level=1.0),
            ConstantKernel(0.5) * RBF(length_scale=15.0) + WhiteKernel(noise_level=0.2),
            ConstantKernel(1.0) * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=0.3),
            RBF(length_scale=3.0) + RBF(length_scale=15.0) + WhiteKernel(noise_level=0.5),
            ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=0.5) + 
            0.5 * RBF(length_scale=50.0),
        ]

    def _train_on_data(self, X_train, y_train, **gpr_params):
        best_score = float('inf')
        best_model = None

        for kernel in self.kernel_list:
            model = GaussianProcessRegressor(kernel=kernel, **gpr_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            mse = np.mean((y_train - y_pred)**2)

            if mse < best_score:
                best_score = mse
                best_model = model

        return best_model

    def predict_n_days(self, n_days, **gpr_params):
        preds = []
        stds = []
        dates = []

        X_train = self.X
        y_train = self.y
        next_idx = len(self.X)

        for i in range(n_days):
            # Fit model
            self.gpr = self._train_on_data(X_train, y_train, **gpr_params)

            # Predict next day
            X_next = np.array([[next_idx]])
            y_pred, y_std = self.gpr.predict(X_next, return_std=True)

            preds.append(y_pred[0])
            stds.append(y_std[0])
            dates.append(self.data.index[-1] + pd.Timedelta(days=i+1))

            y_actual = y_pred[0] + np.random.normal(0, y_std[0])

            # Update training data
            X_train = np.vstack([X_train, X_next])
            y_train = np.append(y_train, y_actual)

            next_idx += 1

        return dates, preds, stds

    def plot_predictions(self, dates, preds, stds, ax):
        preds = np.array(preds).ravel()
        stds = np.array(stds).ravel()

        ax.plot(dates, preds, 'b-', label='Predykcja GPR')

        ax.fill_between(
            dates,
            preds - 1.645 * stds,
            preds + 1.645 * stds,
            color='blue', alpha=0.2,
            label='90% przedział ufności'
        )
