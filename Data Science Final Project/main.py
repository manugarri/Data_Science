
#===========================================================================================================================
#**********************STARTING MODEL
import numpy as np
import pandas as pd
train=pd.read_csv('train.csv')
crime=pd.DataFrame(train.Total_Crime_Risk,dtype=int)
train=train.drop(train.columns[[0,-1,-2]], axis=1)



#TRYING LINEAR
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import auc_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
#Linear regression
model = LinearRegression()#MAE of 38.0
model = linear_model.Ridge() #MAE 	36.2
model = linear_model.LassoLars() #MAE 48.6
model= linear_model.Lasso(alpha=50)#MAE of 35.9
#Trying ensenble techniques
from sklearn.ensemble import  GradientBoostingRegressor 
model = GradientBoostingRegressor() #MAE 40.4
from sklearn.ensemble import  RandomForestRegressor
model = RandomForestRegressor() #MAE 41.9
cross_val_score(model,train,crime.Total_Crime_Risk,score_func=mean_absolute_error,cv=5).mean() 

#=======================================================================================================================================
#Logistic + Linear
#Selecting the proper classifier
train=pd.read_csv('train.csv')
train=train.drop(train.columns[[0,-2,-1]], axis=1) #drop auxiliary columns as well as the Total Crime Risk Index

#auxiliar column to show if the zip is high or low crime
crime['highcrime']=0
crime.highcrime[crime.Total_Crime_Risk>crime.Total_Crime_Risk.median()]=1
crime['GEOGRAPHY_ID']=train.GEOGRAPHY_ID

from sklearn.ensemble import  GradientBoostingClassifier
model = GradientBoostingClassifier()
cross_val_score(model,train,crime.highcrime,score_func=auc_score).mean()
#GradientBoostingClassifier gave a AUC of 0.77 : 
#RandomForestClassifier() is faster and performs with a AUC of 0.75




#===========================================CROSS VALIDATION OF THE LOGIC LINEAR SPLIT====================================================================================================
import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_absolute_error
train=pd.read_csv('train.csv')
def cross_val(regressor_high,regressor_low,classifier,train):
	rows=random.sample(train.index, int(train.shape[0]*0.75))
	sample = train.ix[rows]

	crime=pd.DataFrame(sample.Total_Crime_Risk,dtype=int)
	crime['highcrime']=0
	crime.highcrime[crime.Total_Crime_Risk>crime.Total_Crime_Risk.median()]=1
	crime['GEOGRAPHY_ID']=sample.GEOGRAPHY_ID
	sample=sample.drop(train.columns[[0,-2,-1]], axis=1)
		
	model=classifier.fit(sample, crime.highcrime)
	Highcrime=model.predict(sample)
	Highcrime=np.array(Highcrime)
	sample['predicted_highcrime']=Highcrime
	
	high_areas=sample.ix[sample.predicted_highcrime==1]
	high_areas=pd.merge(high_areas, crime, on='GEOGRAPHY_ID', how= 'inner')
	high_areas_crime=high_areas.Total_Crime_Risk
	high_areas=high_areas.drop(high_areas.columns[[-1,-2,-3]],axis=1)

	low_areas=sample.ix[sample.predicted_highcrime==0]
	low_areas=pd.merge(low_areas, crime, on='GEOGRAPHY_ID', how= 'inner')
	low_areas_crime=low_areas.Total_Crime_Risk
	low_areas=low_areas.drop(low_areas.columns[[-1,-2,-3]],axis=1)

	model_high=regressor_high.fit(high_areas, high_areas_crime)
	high_crime=model_high.predict(high_areas)
	model_low=regressor_low.fit(low_areas, low_areas_crime)
	low_crime=model_low.predict(low_areas)

	high_error=mean_absolute_error(high_areas_crime,high_crime)
	low_error=mean_absolute_error(low_areas_crime,low_crime)
	print high_error,low_error, ((high_error+low_error)/2)


#Linear Benchmark = 35.9
regressor_high = RandomForestRegressor()
regressor_low = linear_model.Lasso(alpha=50)
classifier=GradientBoostingClassifier()
cross_val(regressor_high,regressor_low,classifier,train)	# AVG MAE 18 & 110 seconds

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import  GradientBoostingClassifier 

regressor_high = RandomForestRegressor()
regressor_low = RandomForestRegressor()
classifier=GradientBoostingClassifier()
cross_val(regressor_high,regressor_low,classifier,train)  #AVG MAE 12.1 & 143.3 seconds


from sklearn.linear_model import  LogisticRegression
regressor_high = RandomForestRegressor()
regressor_low = RandomForestRegressor()
classifier=RandomForestClassifier()
cross_val(regressor_high,regressor_low,classifier,train) #AVG MAE 9.72 & 77.1seconds The model we will use

from sklearn import linear_model
regressor_high = linear_model.Lasso(alpha=50)
regressor_low = RandomForestRegressor()
classifier=RandomForestClassifier()
cross_val(regressor_high,regressor_low,classifier,train) #MAE 23.2 & 45 seconds 



