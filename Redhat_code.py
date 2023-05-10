#importing libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Importing test data

import pandas as pd
test_df = pd.read_csv('/gdrive/MyDrive/act_test.csv.zip' )
test_df

#Data type of test data

test_df.dtypes

#Importing train data

import pandas as pd
train_d = '/gdrive/MyDrive/act_train.csv.zip'
train_df = pd.read_csv(train_d)
train_df

#Data type of train data

train_df.dtypes

#Importing people data

import pandas as pd
people_d = '/gdrive/MyDrive/people.csv.zip'
people_df = pd.read_csv(people_d)
people_df

#Data type of people data

people_df.dtypes

#Rename date columns in all three files
train_df.rename(columns={"date": "activ_date"},inplace=True)
people_df.rename(columns={"date": "ppl_date"},inplace=True)
test_df.rename(columns={"date": "activ_date"},inplace=True)

#Merging train and people files

tot1_df=pd.merge(people_df,train_df,on="people_id")
tot1_df.head()

#Shape of tot1
tot1_df.shape

#Merging test and people files

tot2_df=pd.merge(people_df,test_df,on="people_id")
tot2_df.head()
#Shape of tot2
tot2_df.shape

#It gives concise summary of the dataframe
tot1_df.info()

#It gives concise summary of the dataframe
tot2_df.info()

#Finding null values

def findnull(data,col):
    if data[col].isnull().any():
        print(col, data[col].isnull().sum())
    else:
        print("no null values are present in",col)



findnull(tot1_df,"char_2_y")

findnull(tot1_df,"char_1_x")

#Describing the data 
tot1_df.describe(include = 'O').transpose()

#Describing the data 
tot2_df.describe(include = 'O').transpose()

#Percentage of count of number of nulls present in columns of tot1_df data 
((tot1_df.isnull().sum() * 100 / len(tot1_df)).sort_values(ascending=False)).head(50)

#Percentage of count of number of nulls present in columns of tot2_df data 
((tot2_df.isnull().sum() * 100 / len(tot2_df)).sort_values(ascending=False)).head(50)

# delete columns contains most missing data: 
tot1_df = tot1_df.drop(['char_1_y','char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y','char_8_y','char_9_y'],axis=1)

# delete columns contains most missing data: 
tot2_df = tot2_df.drop(['char_1_y','char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y','char_8_y','char_9_y'],axis=1)

#delete people_id,activity_id
tot1_df = tot1_df.drop(['people_id','activity_id'],axis=1)
tot2_df = tot2_df.drop(['people_id','activity_id'],axis=1)

#Checking the counts of values in char_10_y for tot1_df
tot1_df.char_10_y.value_counts()

#Checking the counts of values in char_10_y for tot1_df
tot2_df.char_10_y.value_counts()

tot1_df.char_10_y.fillna('type 1',inplace=True)
tot2_df.char_10_y.fillna('type 1',inplace=True)

((tot1_df.isnull().sum() * 100 / len(tot1_df)).sort_values(ascending=False)).head(50)

((tot2_df.isnull().sum() * 100 / len(tot2_df)).sort_values(ascending=False)).head(50)

#Obtained dataframes

tot1_df.head()

tot2_df.head()

#Univariate Analysis

#Plotting a barchart for the outcome feature 
tot1_df['outcome'].value_counts().plot.bar()
tot1_df['outcome'].value_counts()

tot1_df['char_1_x'].value_counts().plot.bar()



tot1_df['char_2_x'].value_counts().plot.bar()

tot1_df['char_3_x'].value_counts().plot.bar()



tot1_df['char_4_x'].value_counts().plot.bar()

tot2_df['char_1_x'].value_counts().plot.bar()



tot2_df['char_2_x'].value_counts().plot.bar()

tot2_df['char_3_x'].value_counts().plot.bar()

tot2_df['char_4_x'].value_counts().plot.bar()

tot2_df['char_5_x'].value_counts().plot.bar()

tot2_df['char_6_x'].value_counts().plot.bar()

tot1_df_char38= tot1_df.groupby(['char_38','outcome']).char_38.count().unstack()
tot1_df_char38.plot(kind = 'bar', stacked = True, figsize=(14,9),ylabel='count of people',sort_columns="true")
print(tot1_df.groupby(['char_38','outcome']).size())

PDF of char_38

#Plotting PDF of char_38
import matplotlib.pyplot as plt
tot1_df.char_38.plot.density(color='red')
plt.title('PDF of char_38')
plt.show()
tot1_df['char_38'].describe()

#CDF of char_38

#Plotting CDF of char_38
counts, bin_edges = np.histogram(tot1_df["char_38"], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
plt.show();
cdf = np.cumsum(pdf)
plt.figure(figsize=(12,6))
plt.plot(bin_edges[1:], cdf,label='CDF')
plt.title("CDF PLot of char_38")
plt.legend()
plt.xlabel('Char_38')
plt.show();

#Bivariate Analysis

#Comparing Char_1 and Char_2(People)
pd.crosstab(tot1_df['char_2_x'], tot1_df['char_1_x']).plot(kind='bar', stacked=True)

#Comparing Char_5_x and char_2_x
pd.crosstab(tot1_df['char_2_x'], tot1_df['char_5_x']).plot(kind='bar', stacked=True,legend=True)

#Comparing Char_6_x and Char_2_x
pd.crosstab(tot1_df['char_2_x'], tot1_df['char_6_x']).plot(kind='bar', stacked=True,legend=True)

#Comparing Char_6_x and Char_5_x
pd.crosstab(tot1_df['char_5_x'], tot1_df['char_6_x']).plot(kind='bar', stacked=True)

#Comparing char_8_x and char_7_x
pd.crosstab(tot1_df['char_7_x'], tot1_df['char_8_x']).plot(kind='bar', stacked=True)

#Data visualisation through Box plot for analysing char_38 and outcome features
sns.boxplot(x = 'outcome', y = 'char_38', data = tot1_df)

#Time series analysis

#Checking the number of columns in tot1_df
tot1_df.columns

#converting the date columns
tot1_df['ppl_date'] = pd.to_datetime(tot1_df['ppl_date'])
tot1_df['activ_date'] = pd.to_datetime(tot1_df['activ_date'])

#Feature extraction of people date columns into year , month,day ,week, day of week for tot1_df
tot1_df['pplyear']= tot1_df['ppl_date'].dt.year
tot1_df['pplmonth']= tot1_df['ppl_date'].dt.month
tot1_df['pplday']= tot1_df['ppl_date'].dt.day
tot1_df['pplweek']= tot1_df['ppl_date'].dt.week
tot1_df['ppldayofweek']= tot1_df['ppl_date'].dt.dayofweek

#Feature extraction of activity date columns into year , month,day ,week, day of week for tot1_df
tot1_df['actvyear']= tot1_df['activ_date'].dt.year
tot1_df['actvmonth']= tot1_df['activ_date'].dt.month
tot1_df['actvday']= tot1_df['activ_date'].dt.day
tot1_df['actvweek']= tot1_df['activ_date'].dt.week
tot1_df['actvdayofweek']= tot1_df['activ_date'].dt.dayofweek

#Date conversion of columns in tot2_df
tot2_df['ppl_date'] = pd.to_datetime(tot2_df['ppl_date'])
tot2_df['activ_date'] = pd.to_datetime(tot2_df['activ_date'])

#Feature extraction of people date columns into year , month,day ,week, day of week for tot2_df
tot2_df['pplyear']= tot2_df['ppl_date'].dt.year
tot2_df['pplmonth']= tot2_df['ppl_date'].dt.month
tot2_df['pplday']= tot2_df['ppl_date'].dt.day
tot2_df['pplweek']= tot2_df['ppl_date'].dt.week
tot2_df['ppldayofweek']= tot2_df['ppl_date'].dt.dayofweek

#Feature extraction of people date columns into year , month,day ,week, day of week for tot2_df
tot2_df['actvyear']= tot2_df['activ_date'].dt.year
tot2_df['actvmonth']= tot2_df['activ_date'].dt.month
tot2_df['actvday']= tot2_df['activ_date'].dt.day
tot2_df['actvweek']= tot2_df['activ_date'].dt.week
tot2_df['actvdayofweek']= tot2_df['activ_date'].dt.dayofweek

# Now delete ppl_date,activ_date columns
tot1_df= tot1_df.drop(['ppl_date','activ_date'],axis=1)
tot2_df= tot2_df.drop(['ppl_date','activ_date'],axis=1)

#Checking the shape of data object types in tot1_df
tot1_df.info()

#Checking the counts of char_10_x column
tot1_df['char_10_x'].value_counts()

# Plotting bar chart to analyse pplyear
tot1_df['pplyear'].value_counts().plot.bar()
tot1_df['pplyear'].value_counts()

# Plotting bar chart to analyse pplmonth
tot1_df['pplmonth'].value_counts().plot.bar()
tot1_df['pplmonth'].value_counts()

# Plotting bar chart to analyse pplday
tot1_df['pplday'].value_counts().plot.bar()
tot1_df['pplday'].value_counts()

# Plotting bar chart to analyse pplweek
tot1_df['pplweek'].value_counts().plot.bar()
tot1_df['pplweek'].value_counts()

# Plotting bar chart to analyse ppldayofweek
tot1_df['ppldayofweek'].value_counts().plot.bar()
tot1_df['ppldayofweek'].value_counts()

# Plotting bar chart to analyse actvyear
tot1_df['actvyear'].value_counts().plot.bar()
tot1_df['actvyear'].value_counts()

# Plotting bar chart to analyse actvmonth
tot1_df['actvmonth'].value_counts().plot.bar()
tot1_df['actvmonth'].value_counts()

# Plotting bar chart to analyse actvday
tot1_df['actvday'].value_counts().plot.bar()
tot1_df['actvday'].value_counts()

#Checking number of columns for tot1_df
tot1_df.columns

#Correlation between feature

#-1 indicates a perfectly negative linear correlation between two variables. 0 indicates no linear correlation between two variables. 1 indicates a perfectly positive linear correlation between two variables.
corr = tot1_df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(0)
corr

#Heatmap

colormap = plt.cm.RdBu
plt.figure(figsize=(55,50))
mask = np.zeros_like(tot1_df.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(tot1_df.corr(), mask=mask, linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

#Converting variables to integers

# converting Boolean variables in tot1_df to 0 and 1
boolean_var = ['char_10_x','char_11','char_12','char_13','char_14','char_15','char_16','char_17','char_18','char_19','char_20','char_21','char_22','char_23','char_24','char_25','char_26','char_27','char_28','char_29','char_30','char_31','char_32','char_33','char_34','char_35','char_36','char_37']

tot1_df['char_10_x'] = tot1_df['char_10_x'].astype(int)
tot1_df['char_11'] = tot1_df['char_11'].astype(int)    
tot1_df['char_12'] = tot1_df['char_12'].astype(int)
tot1_df['char_13'] = tot1_df['char_13'].astype(int)
tot1_df['char_14'] = tot1_df['char_14'].astype(int)
tot1_df['char_15'] = tot1_df['char_15'].astype(int)
tot1_df['char_16'] = tot1_df['char_16'].astype(int)
tot1_df['char_17'] = tot1_df['char_17'].astype(int)
tot1_df['char_18'] = tot1_df['char_18'].astype(int)
tot1_df['char_19'] = tot1_df['char_19'].astype(int)    
tot1_df['char_20'] = tot1_df['char_20'].astype(int)
tot1_df['char_21'] = tot1_df['char_21'].astype(int)
tot1_df['char_22'] = tot1_df['char_22'].astype(int)
tot1_df['char_23'] = tot1_df['char_23'].astype(int)
tot1_df['char_24'] = tot1_df['char_24'].astype(int)
tot1_df['char_25'] = tot1_df['char_25'].astype(int)
tot1_df['char_26'] = tot1_df['char_26'].astype(int)
tot1_df['char_27'] = tot1_df['char_27'].astype(int)   
tot1_df['char_28'] = tot1_df['char_28'].astype(int)
tot1_df['char_29'] = tot1_df['char_29'].astype(int)
tot1_df['char_30'] = tot1_df['char_30'].astype(int)
tot1_df['char_31'] = tot1_df['char_31'].astype(int)
tot1_df['char_32'] = tot1_df['char_32'].astype(int)
tot1_df['char_33'] = tot1_df['char_33'].astype(int)
tot1_df['char_34'] = tot1_df['char_34'].astype(int)
tot1_df['char_35'] = tot1_df['char_35'].astype(int)
tot1_df['char_36'] = tot1_df['char_36'].astype(int)
tot1_df['char_37'] = tot1_df['char_37'].astype(int)

# converting Boolean variables in tot2_df to 0 and 1

tot2_df['char_10_x'] = tot2_df['char_10_x'].astype(int)
tot2_df['char_11'] = tot2_df['char_11'].astype(int)    
tot2_df['char_12'] = tot2_df['char_12'].astype(int)
tot2_df['char_13'] = tot2_df['char_13'].astype(int)
tot2_df['char_14'] = tot2_df['char_14'].astype(int)
tot2_df['char_15'] = tot2_df['char_15'].astype(int)
tot2_df['char_16'] = tot2_df['char_16'].astype(int)
tot2_df['char_17'] = tot2_df['char_17'].astype(int)
tot2_df['char_18'] = tot2_df['char_18'].astype(int)
tot2_df['char_19'] = tot2_df['char_19'].astype(int)    
tot2_df['char_20'] = tot2_df['char_20'].astype(int)
tot2_df['char_21'] = tot2_df['char_21'].astype(int)
tot2_df['char_22'] = tot2_df['char_22'].astype(int)
tot2_df['char_23'] = tot2_df['char_23'].astype(int)
tot2_df['char_24'] = tot2_df['char_24'].astype(int)
tot2_df['char_25'] = tot2_df['char_25'].astype(int)
tot2_df['char_26'] = tot2_df['char_26'].astype(int)
tot2_df['char_27'] = tot2_df['char_27'].astype(int)   
tot2_df['char_28'] = tot2_df['char_28'].astype(int)
tot2_df['char_29'] = tot2_df['char_29'].astype(int)
tot2_df['char_30'] = tot2_df['char_30'].astype(int)
tot2_df['char_31'] = tot2_df['char_31'].astype(int)
tot2_df['char_32'] = tot2_df['char_32'].astype(int)
tot2_df['char_33'] = tot2_df['char_33'].astype(int)
tot2_df['char_34'] = tot2_df['char_34'].astype(int)
tot2_df['char_35'] = tot2_df['char_35'].astype(int)
tot2_df['char_36'] = tot2_df['char_36'].astype(int)
tot2_df['char_37'] = tot2_df['char_37'].astype(int)

tot1_df.info()

tot1_df.char_1_x.value_counts()

tot1_df['char_1_x'] = tot1_df['char_1_x'].map({'type 1': 0, 'type 2': 1})
tot1_df['char_1_x'].head

tot1_df.char_5_x.value_counts()

tot1_df['char_2_x'] = tot1_df['char_2_x'].map(lambda x: x.lstrip('type '))
tot1_df['char_3_x'] = tot1_df['char_3_x'].map(lambda x: x.lstrip('type '))
tot1_df['char_4_x'] = tot1_df['char_4_x'].map(lambda x: x.lstrip('type '))
tot1_df['char_5_x'] = tot1_df['char_5_x'].map(lambda x: x.lstrip('type '))
tot1_df['char_6_x'] = tot1_df['char_6_x'].map(lambda x: x.lstrip('type '))
tot1_df['char_7_x'] = tot1_df['char_7_x'].map(lambda x: x.lstrip('type '))
tot1_df['char_8_x'] = tot1_df['char_8_x'].map(lambda x: x.lstrip('type '))
tot1_df['char_9_x'] = tot1_df['char_9_x'].map(lambda x: x.lstrip('type '))
tot1_df['char_10_y'] = tot1_df['char_10_y'].map(lambda x: x.lstrip('type '))
tot1_df['group_1'] = tot1_df['group_1'].map(lambda x: x.lstrip('group '))

tot2_df['char_1_x'] = tot2_df['char_1_x'].map({'type 2': 2, 'type 1': 1})
tot2_df['char_2_x'] = tot2_df['char_2_x'].map(lambda x: x.lstrip('type '))
tot2_df['char_3_x'] = tot2_df['char_3_x'].map(lambda x: x.lstrip('type '))
tot2_df['char_4_x'] = tot2_df['char_4_x'].map(lambda x: x.lstrip('type '))
tot2_df['char_5_x'] = tot2_df['char_5_x'].map(lambda x: x.lstrip('type '))
tot2_df['char_6_x'] = tot2_df['char_6_x'].map(lambda x: x.lstrip('type '))
tot2_df['char_7_x'] = tot2_df['char_7_x'].map(lambda x: x.lstrip('type '))
tot2_df['char_8_x'] = tot2_df['char_8_x'].map(lambda x: x.lstrip('type '))
tot2_df['char_9_x'] = tot2_df['char_9_x'].map(lambda x: x.lstrip('type '))
tot2_df['char_10_y'] = tot2_df['char_10_y'].map(lambda x: x.lstrip('type '))
tot2_df['group_1'] = tot2_df['group_1'].map(lambda x: x.lstrip('group '))

tot1_df['char_1_x'] = tot1_df['char_1_x'].astype(int)
tot1_df['char_2_x'] = tot1_df['char_2_x'].astype(int)
tot1_df['char_3_x'] = tot1_df['char_3_x'].astype(int)
tot1_df['char_4_x'] = tot1_df['char_4_x'].astype(int)
tot1_df['char_5_x'] = tot1_df['char_5_x'].astype(int)
tot1_df['char_6_x'] = tot1_df['char_6_x'].astype(int)
tot1_df['char_7_x'] = tot1_df['char_7_x'].astype(int)
tot1_df['char_8_x'] = tot1_df['char_8_x'].astype(int)
tot1_df['char_9_x'] = tot1_df['char_9_x'].astype(int)
tot1_df['char_10_y'] = tot1_df['char_10_y'].astype(int)
tot1_df['group_1'] = tot1_df['group_1'].astype(int)

tot2_df['char_1_x'] = tot2_df['char_1_x'].astype(int)
tot2_df['char_2_x'] = tot2_df['char_2_x'].astype(int)
tot2_df['char_3_x'] = tot2_df['char_3_x'].astype(int)
tot2_df['char_4_x'] = tot2_df['char_4_x'].astype(int)
tot2_df['char_5_x'] = tot2_df['char_5_x'].astype(int)
tot2_df['char_6_x'] = tot2_df['char_6_x'].astype(int)
tot2_df['char_7_x'] = tot2_df['char_7_x'].astype(int)
tot2_df['char_8_x'] = tot2_df['char_8_x'].astype(int)
tot2_df['char_9_x'] = tot2_df['char_9_x'].astype(int)
tot2_df['char_10_y'] = tot2_df['char_10_y'].astype(int)
tot2_df['group_1'] = tot2_df['group_1'].astype(int)

tot1_df.group_1.value_counts()

#One hot encoding 

activity_cat = pd.get_dummies(tot1_df['activity_category'],prefix='activity_category',drop_first=True)
tot1_df = pd.concat([tot1_df,activity_cat],axis=1)

activity_cat = pd.get_dummies(tot2_df['activity_category'],prefix='activity_category',drop_first=True)
tot2_df = pd.concat([tot2_df,activity_cat],axis=1)

tot1_df = tot1_df.drop('activity_category',axis=1)
tot2_df = tot2_df.drop('activity_category',axis=1)

tot1_df['activity_category_type 2'].value_counts()

Splitting data into training and test dataset

X = tot1_df.drop(['outcome'],axis=1)
Y = tot1_df['outcome']


X.head()

len(X. columns)

Y.value_counts()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

X_train.shape

type(X_train)

Y_train.shape

t-SNE (t-Stochastic Neighbor Embedding)

data_1000 = X_train.iloc[0:1000,:]
outcome_1000 = Y_train[0:1000]
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, outcome_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()

model = TSNE(n_components=2, random_state=0, perplexity=25,  n_iter=5000)
tsne_data = model.fit_transform(data_1000) 

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, outcome_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 25, n_iter=5000,learning_rate=800')
plt.show()

#Changing the perplexity =50 & n_iter=10000
model = TSNE(n_components=2, random_state=0, perplexity=50,  n_iter=10000)
tsne_data = model.fit_transform(data_1000) 

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, outcome_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 50, n_iter=10000')
plt.show()

#Changing the perplexity =100
model = TSNE(n_components=2, random_state=0, perplexity=100)
tsne_data = model.fit_transform(data_1000) 

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, outcome_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 100')
plt.show()

#Changing the perplexity =8
model = TSNE(n_components=2, random_state=0, perplexity=5)
tsne_data = model.fit_transform(data_1000) 

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, outcome_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 8')
plt.show()

#Changing the perplexity =3
model = TSNE(n_components=2, random_state=0, perplexity=2)
tsne_data = model.fit_transform(data_1000) 

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, outcome_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 3')
plt.show()

data_1000 = X_train.iloc[0:10000,:]
outcome_1000 = Y_train[0:10000]
model = TSNE(n_components=2, random_state=0, n_iter=5000,perplexity = 5)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, outcome_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()

data_1000 = X_train.iloc[0:10000,:]
outcome_1000 = Y_train[0:10000]

model = TSNE(n_components=2, random_state=0, n_iter=1000, perplexity =1)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, outcome_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()

data_1000 = X_train.iloc[0:10000,:]
outcome_1000 = Y_train[0:10000]

model = TSNE(n_components=2, random_state=0, n_iter=5000,perplexity = 0.9)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, outcome_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()

#Feature Selection

# With the following function we can select highly correlated features

def correlation(dataset, threshold):
    col_corr = set()  
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: 
                colname = corr_matrix.columns[i] 
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)
len(set(corr_features))

#Name of the features dropped

corr_features

X_train1= X_train.drop(corr_features,axis=1)
X_test1=X_test.drop(corr_features,axis=1)
X_test1

#Building model

#Logistic Regreassion

model=LogisticRegression()
model.fit(X_train1, Y_train)

Y_train_pred = model.predict(X_train1)
Y_test_pred = model.predict(X_test1)

model.score(X_test1,Y_test)

model.predict_proba(X_test1)

# Train set performance for accuracy, F1-score, recall and precision
model_train_accuracy = accuracy_score(Y_train, Y_train_pred) 
model_train_f1 = f1_score(Y_train, Y_train_pred, average='weighted') 
model_train_recall = recall_score(Y_train, Y_train_pred)
model_train_precision = precision_score(Y_train, Y_train_pred)

# Test set performance for accuracy, F1-score, recall and precision
model_test_accuracy = accuracy_score(Y_test, Y_test_pred) 
model_test_f1 = f1_score(Y_test, Y_test_pred, average='weighted') 
model_test_recall = recall_score(Y_test, Y_test_pred)
model_test_precision = precision_score(Y_test, Y_test_pred)



print('Model performance for Training set')
print('- Accuracy: %s' % model_train_accuracy)
print('- F1 score: %s' % model_train_f1)
print('- Recall: %s' % model_train_recall)
print('- Precision: %s' % model_train_precision)
print('----------------------------------')
print('Model performance for Test set')
print('- Accuracy: %s' % model_test_accuracy)
print('- F1 score: %s' % model_test_f1)
print('- Recall: %s' % model_test_recall)
print('- Precision: %s' % model_test_precision)

model_probs = model.predict_proba(X_test1)
model_probs = model_probs[:, 1]

model_auc = roc_auc_score(Y_test, model_probs)

print('Logistic Regression: AUROC = %.3f' % (model_auc))


SVM (Support Vector Machine) 

model1=SVC(kernel='linear')
model1.fit(X_train1[:5000], Y_train[:5000])

model1.score(X_test1,Y_test)

Y_train_pred = model1.predict(X_train1[:5000])
Y_test_pred = model1.predict(X_test1)

# Train set performance for accuracy, F1-score, recall and precision
model1_train_accuracy = accuracy_score(Y_train[:5000], Y_train_pred) 
model1_train_f1 = f1_score(Y_train[:5000], Y_train_pred, average='weighted') 
model1_train_recall = recall_score(Y_train[:5000], Y_train_pred)
model1_train_precision = precision_score(Y_train[:5000], Y_train_pred)

# Test set performance for accuracy, F1-score, recall and precision
model1_test_accuracy = accuracy_score(Y_test, Y_test_pred) 
model1_test_f1 = f1_score(Y_test, Y_test_pred, average='weighted') 
model1_test_recall = recall_score(Y_test, Y_test_pred)
model1_test_precision = precision_score(Y_test, Y_test_pred)

print('Model performance for Training set')
print('- Accuracy: %s' % model1_train_accuracy)
print('- F1 score: %s' % model1_train_f1)
print('- Recall: %s' % model1_train_recall)
print('- Precision: %s' % model1_train_precision)
print('----------------------------------')
print('Model performance for Test set')
print('- Accuracy: %s' % model1_test_accuracy)
print('- F1 score: %s' % model1_test_f1)
print('- Recall: %s' % model1_test_recall)
print('- Precision: %s' % model1_test_precision)

#Decision Tree

model2 = tree.DecisionTreeClassifier()
model2.fit(X_train1, Y_train)

model2.score(X_test1,Y_test)

model2.predict(X_test1)

Y_train_pred = model2.predict(X_train1)
Y_test_pred = model2.predict(X_test1)

# Train set performance for accuracy, F1-score, recall and precision
model2_train_accuracy = accuracy_score(Y_train, Y_train_pred) 
model2_train_f1 = f1_score(Y_train, Y_train_pred, average='weighted') 
model2_train_recall = recall_score(Y_train, Y_train_pred)
model2_train_precision = precision_score(Y_train, Y_train_pred)

# Test set performance for accuracy, F1-score, recall and precision
model2_test_accuracy = accuracy_score(Y_test, Y_test_pred) 
model2_test_f1 = f1_score(Y_test, Y_test_pred, average='weighted') 
model2_test_recall = recall_score(Y_test, Y_test_pred)
model2_test_precision = precision_score(Y_test, Y_test_pred)

print('Model performance for Training set')
print('- Accuracy: %s' % model2_train_accuracy)
print('- F1 score: %s' % model2_train_f1)
print('- Recall: %s' % model2_train_recall)
print('- Precision: %s' % model2_train_precision)
print('----------------------------------')
print('Model performance for Test set')
print('- Accuracy: %s' % model2_test_accuracy)
print('- F1 score: %s' % model2_test_f1)
print('- Recall: %s' % model2_test_recall)
print('- Precision: %s' % model2_test_precision)

model2_probs = model2.predict_proba(X_test1)
model2_probs = model2_probs[:, 1]

model2_auc = roc_auc_score(Y_test, model2_probs)

print('Decision Tree: AUROC = %.3f' % (model2_auc))


#Random forest

model3 = RandomForestClassifier(n_estimators=70)
model3.fit(X_train1, Y_train)

model3.score(X_test1, Y_test)

Y_train_pred = model3.predict(X_train1)
Y_test_pred = model3.predict(X_test1)

# Train set performance for accuracy, F1-score, recall and precision
model3_train_accuracy = accuracy_score(Y_train, Y_train_pred)
model3_train_f1 = f1_score(Y_train, Y_train_pred, average='weighted') 
model3_train_recall = recall_score(Y_train, Y_train_pred)
model3_train_precision = precision_score(Y_train, Y_train_pred)

# Test set performance for accuracy, F1-score, recall and precision
model3_test_accuracy = accuracy_score(Y_test, Y_test_pred) 
model3_test_f1 = f1_score(Y_test, Y_test_pred, average='weighted') 
model3_test_recall = recall_score(Y_test, Y_test_pred)
model3_test_precision = precision_score(Y_test, Y_test_pred)

print('Model performance for Training set')
print('- Accuracy: %s' % model3_train_accuracy)
print('- F1 score: %s' % model3_train_f1)
print('- Recall: %s' % model3_train_recall)
print('- Precision: %s' % model3_train_precision)
print('----------------------------------')
print('Model performance for Test set')
print('- Accuracy: %s' % model3_test_accuracy)
print('- F1 score: %s' % model3_test_f1)
print('- Recall: %s' % model3_test_recall)
print('- Precision: %s' % model3_test_precision)

model3_probs = model3.predict_proba(X_test1)
model3_probs = model3_probs[:, 1]

model3_auc = roc_auc_score(Y_test, model3_probs)
print('Random Forest: AUROC = %.3f' % (model3_auc))


model3_fpr, model3_tpr, _ = roc_curve(Y_test, model3_probs)

plt.plot(model3_fpr, model3_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % model3_auc)
# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()

#Building stacked model 

from sklearn.ensemble import StackingClassifier
estimator_list = [
              ('Logistic Regression,', model),
              ('SVM', model1),
              ('Decision Tree', model2),
              ('Random Forest', model3)]

stack_model = StackingClassifier(
    estimators= estimator_list, final_estimator=RandomForestClassifier()
)

stack_model.fit(X_train1[:500], Y_train[:500])


# Make predictions
Y_train_pred = stack_model.predict(X_train1[:500])
Y_test_pred = stack_model.predict(X_test1)

# Train set performance for accuracy, F1-score, recall and precision
stack_model_train_accuracy = accuracy_score(Y_train[:500], Y_train_pred) 
stack_model_train_f1 = f1_score(Y_train[:500], Y_train_pred, average='weighted')
stack_model_train_recall = recall_score(Y_train[:500], Y_train_pred)
stack_model_train_precision = precision_score(Y_train[:500], Y_train_pred)

# Test set performance for accuracy, F1-score, recall and precision
stack_model_test_accuracy = accuracy_score(Y_test, Y_test_pred) 
stack_model_test_f1 = f1_score(Y_test, Y_test_pred, average='weighted')
stack_model_test_recall = recall_score(Y_test, Y_test_pred)
stack_model_test_precision = precision_score(Y_test, Y_test_pred)

print('Model performance for Training set')
print('- Accuracy: %s' % stack_model_train_accuracy)
print('- F1 score: %s' % stack_model_train_f1)
print('- Recall: %s' % stack_model_train_recall)
print('- Precision: %s' % stack_model_train_precision)
print('----------------------------------')
print('Model performance for Test set')
print('- Accuracy: %s' % stack_model_test_accuracy)
print('- F1 score: %s' % stack_model_test_f1)
print('- Recall: %s' % stack_model_test_recall)
print('- Precision: %s' % stack_model_test_precision)


stack_model_probs = stack_model.predict_proba(X_test1)
stack_model_probs = stack_model_probs[:, 1]

stack_model_auc = roc_auc_score(Y_test, stack_model_probs)
print('Stack model: AUROC = %.3f' % (stack_model_auc))

#Table of the values of accuracy, F1-score, recall, precision and AUROC for stack model

stack_model = [stack_model_train_accuracy, stack_model_train_f1, stack_model_train_recall, stack_model_train_precision, stack_model_auc]
index = ['Accuracy','F1-score','Recall','Precision','AUROC']
pd.DataFrame(zip( index, stack_model))

#Values of accuracy, F1-score, recall and precision for train set of all the models used

Loistic_Regression = [model_train_accuracy, model_train_f1, model_train_recall, model_train_precision]
SVM = [model1_train_accuracy, model1_train_f1, model1_train_recall, model1_train_precision]
Decision_Tree = [model2_train_accuracy, model2_train_f1, model2_train_recall, model2_train_precision]
Random_Forest = [model3_train_accuracy, model3_train_f1, model3_train_recall, model3_train_precision]
Stack_Model = [stack_model_train_accuracy, stack_model_train_f1, stack_model_train_recall, stack_model_train_precision]
index = ['Accuracy','F1-score','Recall','Precision']
A = { 'Factors' : index, 'Loistic Regression' : Loistic_Regression, 'SVM' : SVM, 'Decision Tree' :Decision_Tree, 'Random Forest' : Random_Forest, 'Stack Model' : Stack_Model}
pd.DataFrame.from_dict(A, orient='index').transpose()


#Values of accuracy, F1-score, recall and precision for test set of all the models used

Loistic_Regression = [model_test_accuracy, model_test_f1, model_test_recall, model_test_precision]
SVM = [model1_test_accuracy, model1_test_f1, model1_test_recall, model1_test_precision]
Decision_Tree = [model2_test_accuracy, model2_test_f1, model2_test_recall, model2_test_precision]
Random_Forest = [model3_test_accuracy, model3_test_f1, model3_test_recall, model3_test_precision]
Stack_Model = [stack_model_test_accuracy, stack_model_test_f1, stack_model_test_recall, stack_model_test_precision]
index = ['Accuracy','F1-score','Recall','Precision']
A = { 'Factors' : index, 'Loistic Regression' : Loistic_Regression, 'SVM' : SVM, 'Decision Tree' :Decision_Tree, 'Random Forest' : Random_Forest, 'Stack Model' : Stack_Model}
pd.DataFrame.from_dict(A, orient='index').transpose()


#Values of AUC for all the models used

Loistic_Regression = [model_auc]
Decision_Tree = [model2_auc]
Random_Forest = [model3_auc]
Stack_Model = [stack_model_auc]
index = ['AUC']
pd.DataFrame(zip( index, Loistic_Regression, Decision_Tree, Random_Forest, Stack_Model))
A = { 'Factors' : index, 'Loistic Regression' : Loistic_Regression, 'Decision Tree' :Decision_Tree, 'Random Forest' : Random_Forest, 'Stack Model' : Stack_Model}
pd.DataFrame.from_dict(A, orient='index').transpose()


model_fpr, model_tpr, _ = roc_curve(Y_test, model_probs)
model2_fpr, model2_tpr, _ = roc_curve(Y_test, model2_probs)
model3_fpr, model3_tpr, _ = roc_curve(Y_test, model3_probs)
stack_model_fpr, stack_model_tpr, _ = roc_curve(Y_test, stack_model_probs)

plt.plot(model_fpr, model_tpr, marker='.', label='Logistic Regression (AUROC = %0.3f)' % model_auc)
plt.plot(model2_fpr, model2_tpr, marker='.', label='Decision Tree (AUROC = %0.3f)' % model2_auc)
plt.plot(model3_fpr, model3_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % model3_auc)
plt.plot(stack_model_fpr, stack_model_tpr, marker='.', label='Stack Model (AUROC = %0.3f)' % stack_model_auc)
# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()