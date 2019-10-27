import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# Import data set from URL and store data in df variable and print the correlation between variables
path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
print(df.corr())
print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())
# Linear correlation of engine size, highway mpg, peak rpm and stoke as a potential predictor
# of price variable
print('Correlation between engine size and price: ')
print(df[["engine-size", "price"]].corr())
print('Correlation between highway mpg and price: ')
print(df[['highway-mpg', 'price']].corr())
print('Correlation between peak rpm and price: ')
print(df[['peak-rpm','price']].corr())
print('Correlation between stroke and price: ')
print(df[['stroke','price']].corr())
# Boxplot plot to visualize categorical variables like body style, engine location and drive
# wheels in relationship with price
#sns.boxplot(x="body-style", y="price", data=df)
#sns.boxplot(x="engine-location", y="price", data=df)
#sns.boxplot(x="drive-wheels", y="price", data=df)
# Create a data frame with the value count of variable engine location and drive wheels
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'\
# Group the drive wheels, body style and price variables
df['drive-wheels'].unique()
df_gpt = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gpt.groupby(['drive-wheels','body-style'],as_index=False).mean()
print('Average price for drive wheel and body type type: ')
print(df_gpt)
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0)
print('Pivoted average price for drive wheel and body type type: ')
print(grouped_pivot)
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')
row_labels = grouped_pivot.columns.levels[1]        # row label
col_labels = grouped_pivot.index                    # col label
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)     # Center x axis label
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)     # Center y axis label
ax.set_xticklabels(row_labels, minor=False)     # Insert x axis label
ax.set_yticklabels(col_labels, minor=False)     # Insert y axis label
plt.xticks(rotation=90)         # Rotate label 90 degrees if it's too long
fig.colorbar(im)
# Pearson Correlation. 1 = total linear correlation, 0 = no linear correlation,
# -1 = total negative correlation
print('Pearson Correlation')
print(df.corr())
# P values. < 0.0001 indicates strong evidence that correlation is significant, < 0.05 indicates
# there is moderate evidence that correlation is significant, < 0.1 indicates weak evidence that
# correlation is significant, > 0.1 indicates there is no evidence that correlation is significant
# Pearson coefficient and p value between various variables and price
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value)
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
# ANOVA (analysis of variance) returns f test score (difference between means) and p value
# (how statistically significant is the calculated score value)
grouped_test2=df_gpt[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.get_group('4wd')['price']
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])
print("ANOVA results for fwd, rwd and 4wd: F=", f_val, ", P =", p_val)
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])
print("ANOVA results fwd and rwd: F=", f_val, ", P =", p_val)
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])
print("ANOVA results 4wd and rwd: F=", f_val, ", P =", p_val)
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])
print("ANOVA results 4wd and fwd: F=", f_val, ", P =", p_val)
