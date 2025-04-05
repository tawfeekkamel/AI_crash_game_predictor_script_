from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class Process:
    def __init__(self):
        self.df = pd.read_csv('data - Copy.csv')
        self.y = self.df['Multiplier(Crash)']
        self.X = self.df.drop(columns=['Multiplier(Crash)', 'Time'])
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, test_size=0.9, random_state=123)
        self.forest_reg = RandomForestRegressor()
        self.nn_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=3000, random_state=123)
        self.ridge_reg = Ridge(alpha=1.0)  # Adding Ridge Regression

class Train:
    def __init__(self):
        self.pr = Process()
        self.train_X = self.pr.train_X
        self.train_y = self.pr.train_y
        self.nn_reg = self.pr.nn_reg
        self.forest_reg = self.pr.forest_reg
        self.ridge_reg = self.pr.ridge_reg
        self.forest_reg.fit(self.train_X, self.train_y)
        self.nn_reg.fit(self.train_X, self.train_y)
        self.ridge_reg.fit(self.train_X, self.train_y)  # Training Ridge Regression

class Acc:
    def __init__(self):
        self.tr = Train()
        self.pr = Process()
        self.forest_reg_pred = self.tr.forest_reg.predict(self.pr.test_X)
        self.nn_reg_pred = self.tr.nn_reg.predict(self.pr.test_X)
        self.ridge_reg_pred = self.tr.ridge_reg.predict(self.pr.test_X)  # Ridge Regression predictions
        self.forest_reg_score = str(r2_score(self.pr.test_y, self.forest_reg_pred) * 100)[:4]
        self.nn_reg_score = str(r2_score(self.pr.test_y, self.nn_reg_pred) * 100)[:4]
        self.ridge_reg_score = str(r2_score(self.pr.test_y, self.ridge_reg_pred) * 100)[:4]  # Ridge Regression score
        print(f"The accuracy for RandomForestRegressor is {self.forest_reg_score}%.")
        print(f"The accuracy for MLPRegressor is {self.nn_reg_score}%.")
        print(f"The accuracy for Ridge Regression is {self.ridge_reg_score}%.")

# Implementing cross-validation
def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean cross-validation score: {scores.mean()}")

# Example usage
process = Process()
cross_validate_model(process.forest_reg, process.X, process.y)
cross_validate_model(process.nn_reg, process.X, process.y)
cross_validate_model(process.ridge_reg, process.X, process.y)
ac=Acc()