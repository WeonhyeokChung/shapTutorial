# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


###

import pandas as pd
import numpy as np
import shap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if you don't need to display the plot interactively

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

shap.initjs()
mpl.rcParams['axes.unicode_minus'] = False

# 캐글 데이터 URL: https://www.kaggle.com/datasets/sazid28/advertising.csv
df = pd.read_csv('data/Advertising.csv', dtype={'sales': np.float64})
df.drop('Unnamed: 0', axis=1, inplace=True)
df.info()

# Create train test  split.
Y = df['sales']
X = df[['TV', 'radio', 'newspaper']]
# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fit Random Forest Regressor Model
rf = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=20)
rf.fit(X_train, Y_train)

# Predict
Y_predict = rf.predict(X_test)

# RMSE
print('RMSE: ', mean_squared_error(Y_test, Y_predict) ** (0.5))

# create explainer model by passing trained model to shap
explainer = shap.TreeExplainer(rf)

# Get SHAP values for the training data from explainer model
shap_values_train = explainer.shap_values(X_train)

# Get SHAP values for the test data from explainer model
shap_values_test = explainer.shap_values(X_test)

# Create a dataframe of the shap values for the training set and the test set
df_shap_train = pd.DataFrame(shap_values_train, columns=['TV_Shap', 'radio_Shap', 'newspaper_Shap'])
df_shap_test = pd.DataFrame(shap_values_test, columns=['TV_Shap', 'radio_Shap', 'newspaper_Shap'])

# Base Value: Training data
print('base array: ', '\n', shap_values_train.base[0], '\n')
print('basevalue: ', shap_values_train.base[0][3])

# Base Value: Test data
print('base array: ', '\n', shap_values_test.base[0], '\n')
print('basevalue: ', shap_values_test.base[0][3])

# Base Value is is calculated from the training dataset so its same for test data
base_value = shap_values_train.base[0][3][0]

# Create a  new column for base value
df_shap_train['BaseValue'] = base_value

# Add shap values and Base Value
df_shap_train['(ShapValues + BaseValue)'] = df_shap_train.iloc[:, 0] + df_shap_train.iloc[:, 1] + df_shap_train.iloc[:,
                                                                                                  2] + base_value

# Also create a new columns for the prediction values from training dataset
df_shap_train['prediction'] = pd.DataFrame(list(rf.predict(X_train)))

# Note: Prediction Column is added to compare the values of Predicted with the sum of Base and SHAP Values

df_shap_train.head()

#피쳐별 평균 임팩트 확인 - 각 피쳐에 해당하는 SHAP 값의 집계
shap.summary_plot(shap_values_train,
                  X_train,
                  plot_type='bar',
                  color=['#093333', '#ffbf00', '#ff0000'])

# 트레이닝 데이터의 SHAP Summary plot
shap.summary_plot(shap_values_train, X_train)

# 테스트 데이터의 SHAP Summary plot
shap.summary_plot(shap_values_test, X_test)

# 트레이닝 데이터의 TV 광고비 지출과 SHAP Values plot
shap.dependence_plot("TV", shap_values_train, X_train, interaction_index=None)

# 테스트 데이터의 TV 광고비 지출과 SHAP Values plot
shap.dependence_plot("TV", shap_values_test, X_test, interaction_index=None)

# 트레이닝 데이터의 Radio 광고비 지출과 SHAP Values plot
shap.dependence_plot("radio", shap_values_train, X_train, interaction_index=None)

# 테스트 데이터의 Radio 광고비 지출과 SHAP Values plot
shap.dependence_plot("radio", shap_values_test, X_test, interaction_index=None)

# TV 광고의 임계 지출액 탐색
fig, ax = plt.subplots(1, 1, figsize=(10,7))
ax.hlines(0, 0, 135, colors='orange')
ax.vlines(135, -10, 0, colors='orange')
shap.dependence_plot("TV", shap_values_train, X_train, interaction_index=None, ax=ax)

# SHAP interaction values
shap_interaction_values = shap.TreeExplainer(rf).shap_interaction_values(X_train)

# Interaction Summary Plot
shap.summary_plot(shap_interaction_values, X_train)

# TV와 Radio 간 Dependence Plot
shap.dependence_plot("TV", shap_values_train, X_train, interaction_index='auto')

# TV와 Radio 간 Dependence Plot 영역 구분
fig, ax = plt.subplots(1, 1, figsize=(10,7))
ax.vlines(150, -10, 6, colors='orange')
rect = mpl.patches.Rectangle((215, 1.5), 20, 4,
                              edgecolor='red',
                              facecolor='white',
                              lw=4,
                              alpha=0.5)
ax.add_patch(rect)

shap.dependence_plot("TV", shap_values_train, X_train, interaction_index='auto', ax=ax)

# force Plot을 사용한 로컬 해석

i = 0  # 첫번째 SHAP Value에 대한 인덱스
j = 134  # 트레[인] 데이터의 해당 인스턴스에 대한 인덱스 값

shap.force_plot(explainer.expected_value,
                shap_values_train[i],
                features=X_train.loc[j],
                feature_names=X_train.columns,
                matplotlib=True)  # If you prefer to use matplotlib.


