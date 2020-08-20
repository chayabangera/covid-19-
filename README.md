# covid-19-
import pandas as pd
import numpy as np
import seaborn as sns
import quandl
import matplotlib.pyplot as plt
import sklearn
import scipy.stats
import datetime as dt

#1.importing dataset
data=pd.read_csv("owid-covid-data.csv")
#df=data.head(30)
#print(df)

#subseting the dataset (we need only 'india' in the location column)based on that we are doing the model
sub=data[data["location"]=="India"]
#print(sub)


#isnull concept
#data1=sub.isnull()
#data2=sub.isnull().sum()
#print(data1)
#print(data2)


#in this model having the null values so replacing continous column to that mean

sub['new_tests']=sub['new_tests'].fillna((sub["new_tests"].mean()),inplace=True)
print(sub["new_tests"])
sub['total_tests']=sub['total_tests'].fillna((sub['total_tests'].mean()),inplace=True)
print(sub['total_tests'])
sub['total_tests_per_thousand']=sub['total_tests_per_thousand'].fillna((sub['total_tests_per_thousand'].mean()),inplace=True)
print(sub['total_tests_per_thousand'])
sub['new_tests_per_thousand']=sub['new_tests_per_thousand'].fillna((sub['new_tests_per_thousand'].mean()),inplace=True)
print(sub['new_tests_per_thousand'])
sub['new_tests_smoothed']=sub['new_tests_smoothed'].fillna((sub['new_tests_smoothed'].mean()),inplace=True)
print(sub['new_tests_smoothed'])
sub['new_tests_smoothed_per_thousand']=sub['new_tests_smoothed_per_thousand'].fillna((sub['new_tests_smoothed_per_thousand'].mean()),inplace=True)
print(sub['new_tests_smoothed_per_thousand'])
sub['tests_per_case']=sub['tests_per_case'].fillna((sub['tests_per_case'].mean()),inplace=True)
print(sub['tests_per_case'])
sub['positive_rate']=sub['positive_rate'].fillna((sub['positive_rate'].mean()),inplace=True)
print(sub['positive_rate'])
sub['stringency_index']=sub['stringency_index'].fillna((sub['stringency_index'].mean()),inplace=True)
print(sub['stringency_index'])

#replacing the numerical column with its mode (null values are present in that column)

#sub['tests_units']=sub['tests_units'].fillna((sub['tests_units'].mode()),inplace=True)
#print(sub['tests_units'])

#histogram for 10 feature column

#sub.hist(column='total_cases' ,density='False',color='b',edgecolor='k',alpha=0.6)
#plt.title('Histogram of total_cases')
#plt.show()

#sub.hist(column='new_cases' ,density='False',color='b',edgecolor='k',alpha=0.6)
#plt.title('Histogram of new_cases')
#plt.show()

#sub.hist(column='total_deaths' ,density='False',color='b',edgecolor='k',alpha=0.6)
#plt.title('Histogram of total_deaths')
#plt.show()


#sub.hist(column='new_deaths' ,density='False',color='b',edgecolor='k',alpha=0.6)
#plt.title('Histogram of new_deaths')
#plt.show()

#sub.hist(column='total_cases_per_million' ,density='False',color='b',edgecolor='k',alpha=0.6)
#plt.title('Histogram of  total_cases_per_million')
#plt.show()

#sub.hist(column='new_cases_per_million' ,density='False',color='b',edgecolor='k',alpha=0.6)
#plt.title('Histogram of new_cases_per_million')
#plt.show()

#sub.hist(column='total_deaths_per_million' ,density='False',color='b',edgecolor='k',alpha=0.6)
#plt.title('Histogram of total_deaths_per_million')
#plt.show()

#sub.hist(column='new_deaths_per_million' ,density='False',color='b',edgecolor='k',alpha=0.6)
#plt.title('Histogram of  new_deaths_per_million ')
#plt.show()


#sub.hist(column='new_tests' ,density='False',color='b',edgecolor='k',alpha=0.6)
#plt.title('Histogram of  new_tests')
#plt.show()

#sub.hist(column='population' ,density='False',color='b',edgecolor='k',alpha=0.6)
#plt.title('Histogram of  population')
#plt.show()


#finding the mean ,median, mode for each column

#mean1=sub['total_cases'].mean()
#print(mean1)
#mode1=sub['total_cases'].mode()
#print(mode1)
#median1=sub['total_cases'].median()
#print(median1)

mean2=sub['new_cases'].mean()
print(mean2)
mode2=sub['new_cases'].mode()
print(mode2)
median2=sub['new_cases'].median()
print(median2)

mean3=sub['total_deaths'].mean()
print(mean3)
mode3=sub['total_deaths'].mode()
print(mode3)
median3=sub['total_deaths'].median()
print(median3)

mean4=sub['new_deaths'].mean()
print(mean4)
mode4=sub['new_deaths'].mode()
print(mode4)
median4=sub['new_deaths'].median()
print(median4)

mean5=sub['total_cases_per_million'].mean()
print(mean5)
mode5=sub['total_cases_per_million'].mode()
print(mode5)
median5=sub['total_cases_per_million'].median()
print(median5)

mean6=sub['new_cases_per_million'].mean()
print(mean6)
mode6=sub['new_cases_per_million'].mode()
print(mode6)
median6=sub['new_cases_per_million'].median()
print(median6)


mean7=sub['total_deaths_per_million'].mean()
print(mean7)
mode7=sub['total_deaths_per_million'].mode()
print(mode7)
median7=sub['total_deaths_per_million'].median()
print(median7)

mean8=sub['new_deaths_per_million'].mean()
print(mean8)
mode8=sub['new_deaths_per_million'].mode()
print(mode8)
median8=sub['new_deaths_per_million'].median()
print(median8)

mean9=sub['new_tests'].mean()
print(mean9)
mode9=sub['new_tests'].mode()
print(mode9)
median9=sub['new_tests'].median()
print(median9)

mean10=sub['total_tests'].mean()
print(mean10)
mode10=sub['total_tests'].mode()
print(mode10)
median10=sub['total_tests'].median()
print(median10)

mean11=sub['total_tests_per_thousand'].mean()
print(mean11)
mode11=sub['total_tests_per_thousand'].mode()
print(mode11)
median11=sub['total_tests_per_thousand'].median()
print(median11)

mean12=sub['new_tests_per_thousand'].mean()
print(mean12)
mode12=sub['new_tests_per_thousand'].mode()
print(mode12)
median12=sub['new_tests_per_thousand'].median()
print(median12)

mean14=sub['new_tests_smoothed_per_thousand'].mean()
print(mean14)
mode14=sub['new_tests_smoothed_per_thousand'].mode()
print(mode14)
median14=sub['new_tests_smoothed_per_thousand'].median()
print(median14)

mean15=sub['tests_per_case'].mean()
print(mean15)
mode15=sub['tests_per_case'].mode()
print(mode15)
median15=sub['tests_per_case'].median()
print(median15)

mean16=sub['positive_rate'].mean()
print(mean16)
mode16=sub['positive_rate'].mode()
print(mode16)
median16=sub['positive_rate'].median()
print(median16)

mean17=sub['stringency_index'].mean()
print(mean17)
mode17=sub['stringency_index'].mode()
print(mode17)
median17=sub['stringency_index'].median()
print(median17)

mean18=sub['population'].mean()
print(mean18)
mode18=sub['population'].mode()
print(mode18)
median18=sub['population'].median()
print(median18)

mean19=sub['population_density'].mean()
print(mean19)
mode19=sub['population_density'].mode()
print(mode19)
median19=sub['population_density'].median()
print(median19)

mean20=sub['median_age'].mean()
print(mean20)
mode20=sub['median_age'].mode()
print(mode20)
median20=sub['median_age'].median()
print(median20)

mean21=sub['aged_65_older'].mean()
print(mean21)
mode21=sub['aged_65_older'].mode()
print(mode21)
median21=sub['aged_65_older'].median()
print(median21)

mean22=sub['aged_70_older'].mean()
print(mean22)
mode22=sub['aged_70_older'].mode()
print(mode22)
median22=sub['aged_70_older'].median()
print(median22)

mean23=sub['gdp_per_capita'].mean()
print(mean23)
mode23=sub['gdp_per_capita'].mode()
print(mode23)
median23=sub['gdp_per_capita'].median()
print(median23)

mean24=sub['extreme_poverty'].mean()
print(mean24)
mode24=sub['extreme_poverty'].mode()
print(mode24)
median24=sub['extreme_poverty'].median()
print(median24)

mean25=sub['cardiovasc_death_rate'].mean()
print(mean25)
mode25=sub['cardiovasc_death_rate'].mode()
print(mode25)
median25=sub['cardiovasc_death_rate'].median()
print(median25)

mean26=sub['diabetes_prevalence'].mean()
print(mean26)
mode26=sub['diabetes_prevalence'].mode()
print(mode26)
median26=sub['diabetes_prevalence'].median()
print(median26)

mean27=sub['female_smokers'].mean()
print(mean27)
mode27=sub['female_smokers'].mode()
print(mode27)
median27=sub['female_smokers'].median()
print(median27)

mean28=sub['male_smokers'].mean()
print(mean28)
mode28=sub['male_smokers'].mode()
print(mode28)
median28=sub['male_smokers'].median()
print(median28)

mean29=sub['handwashing_facilities'].mean()
print(mean29)
mode29=sub['handwashing_facilities'].mode()
print(mode29)
median29=sub['handwashing_facilities'].median()
print(median29)

mean30=sub['hospital_beds_per_thousand'].mean()
print(mean30)
mode30=sub['hospital_beds_per_thousand'].mode()
print(mode30)
median30=sub['hospital_beds_per_thousand'].median()
print(median30)

mean31=sub['life_expectancy'].mean()
print(mean31)
mode31=sub['life_expectancy'].mode()
print(mode31)
median31=sub['life_expectancy'].median()
print(median31)


            #DRAW THE SCATTER PLOT OF TARGET VRESUS 10 FEATURE

#sub.plot.scatter(x='total_cases',y='new_cases')
# plt.title('scatterplot for total_cases and new_cases')
#plt.show()

#sub.plot.scatter(x='total_deaths',y='new_deaths')
#plt.title('scatterplot for total_deaths and new_deaths')
#plt.show()

#sub.plot.scatter(x='total_cases_per_million',y='new_cases_per_million')
#plt.title('scatterplot for total_cases_per_million and new_cases_per_million')
#plt.show()

#sub.plot.scatter(x='total_deaths_per_million',y='new_deaths_per_million')
#plt.title('scatterplot for total_deaths_per_million and new_deaths_per_million')
#plt.show()

#sub.plot.scatter(x='new_tests',y='total_tests')
#plt.title('scatterplot for new_tests and total_tests')
#plt.show()

#sub.plot.scatter(x='total_tests_per_thousand',y='new_tests_per_thousand')
#plt.title('scatterplot for total_tests_per_thousand and new_tests_per_thousand')
#plt.show()

#sub.plot.scatter(x='population',y='population_density')
#plt.title('scatterplot for population and population_density')
#plt.show()

#sub.plot.scatter(x='aged_65_older',y='aged_75_older')
#plt.title('scatterplot for aged_65_older and aged_75_older')
#plt.show()

#sub.plot.scatter(x='female_smokers',y='male_smokers')
#plt.title('scatterplot for female_smokers and male_smokers')
#plt.show()

#sub.plot.scatter(x='handwashing_facilities',y='hospital_beds_per_thousand')
#plt.title('scatterplot for handwashing_facilities and hospital_beds_per_thousand')
#plt.show()

         #line plot
#sub.plot('total_cases','new_cases')
#plt.title('total-cases Vs new-cases',fontsize=14)
#plt.xlabel('total_cases',fontsize=14)
#plt.ylabel('new_cases',fontsize=14)
#plt.grid()
#plt.show()

#sub.plot('total_deaths','new_deaths')
#plt.title('total_deaths Vs new_deaths',fontsize=14)
#plt.xlabel('total_deaths',fontsize=14)
#plt.ylabel('new_deaths',fontsize=14)
#plt.grid()
#plt.show()

#ub.plot('total_cases_per_million','new_cases_per_million')
#plt.title('total_cases_per_million Vs new_cases_per_million',fontsize=14)
#plt.xlabel('total_cases_per_million',fontsize=14)
#plt.ylabel('new_cases_per_million',fontsize=14)
#plt.grid()
#plt.show()

#sub.plot('total_deaths_per_million','new_deaths_per_million')
#plt.title('new_deaths_per_million Vs total_deaths_per_million',fontsize=14)
#plt.xlabel('total_deaths_per_million',fontsize=14)
#plt.ylabel('new_deaths_per_million',fontsize=14)
#plt.grid()
#plt.show()

#sub.plot('new_tests','total_tests')
#plt.title('new_tests Vs total_tests',fontsize=14)
#plt.xlabel('new_tests',fontsize=14)
#plt.ylabel('total_tests',fontsize=14)
#plt.grid()
#plt.show()

#sub.plot('total_tests_per_thousand','new_tests_per_thousand')
#plt.title('total_tests_per_thousand Vs new_tests_per_thousand',fontsize=14)
#plt.xlabel('total_cases',fontsize=14)
#plt.ylabel('new_tests_per_thousand',fontsize=14)
#plt.grid()
#plt.show()

#sub.plot('population','population_density')
#plt.title('population Vs population_density',fontsize=14)
#plt.xlabel('population',fontsize=14)
#plt.ylabel('population_density',fontsize=14)
#plt.grid()
#plt.show()

#sub.plot('aged_65_older','aged_75_older')
#plt.title('aged_65_older Vs aged_75_older',fontsize=14)
#plt.xlabel('aged_65_older',fontsize=14)
#plt.ylabel('aged_75_older',fontsize=14)
#plt.grid()
#plt.show()

#sub.plot('female_smokers','male_smokers')
#plt.title('female_smokers Vs male_smokers',fontsize=14)
#plt.xlabel('female_smokers',fontsize=14)
#plt.ylabel('male_smokers',fontsize=14)
#plt.grid()
#plt.show()

#sub.plot('handwashing_facilities','hospital_beds_per_thousand')
#plt.title('handwashing_facilities Vs hospital_beds_per_thousand',fontsize=14)
#plt.xlabel('handwashing_facilities',fontsize=14)
#plt.ylabel('hospital_beds_per_thousand',fontsize=14)
#plt.grid()
#plt.show()


#datatime
#converting the date column to ordinal

sub['date']=pd.to_datetime(sub['date'])
sub['date']=sub['date'].map(dt.datetime.toordinal)
print(sub['date'])
#converting ordinal date to datetime

from _datetime import datetime
ordinal_value=733828
dt=datetime.fromordinal(ordinal_value)
print(dt)

         #drop the useless categorical column and convert usefull categorical column to numerical form
sub=sub.drop(['continent','iso_code','location','tests_units','new_tests','total_tests','total_tests_per_thousand','new_tests_per_thousand','new_tests_smoothed','new_tests_smoothed_per_thousand','tests_per_case','positive_rate','stringency_index'],axis=1)
print(sub.info())
print(sub.head(100))

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
label.fit(sub['date'])
sub['date']=label.transform(sub['date'])


    #x is feature column
    #y is target column

X=sub[['date','total_cases','new_cases','total_deaths','new_deaths','total_cases_per_million','new_cases_per_million','total_deaths_per_million','new_deaths_per_million','population',
       'population_density','median_age','aged_65_older','aged_70_older','gdp_per_capita','extreme_poverty','cardiovasc_death_rate','diabetes_prevalence','female_smokers','male_smokers',
    'handwashing_facilities','hospital_beds_per_thousand','life_expectancy']]
y=sub['total_cases']

#spliting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

    #linearRegression
#from sklearn.linear_model import LinearRegression
#lr =LinearRegression()
#lr.fit(x_train, y_train)
#print(lr.intercept_)
#print(lr.coef_)
#coeff_df=pd.DataFrame(lr.coef_,X.columns,columns=['coefficient'])
#print(coeff_df)
#y_pred=lr.predict(x_test)
#print(y_pred)
#df=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
#print(df)

#from sklearn import metrics
#print('mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
#print('mean squared Error:',metrics.mean_absolute_error(y_test,y_pred))
#print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


#Random forest regression for regression
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#trainig the algorithm
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=20,random_state=0)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#evaluating
from sklearn import metrics
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean squarewd error:',metrics.mean_squared_error(y_test,y_pred))
print('root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))







