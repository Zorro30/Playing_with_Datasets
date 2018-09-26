import pandas as pd
import numpy as np
import argparse
from tpot import TPOTRegressor
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument("--country", help="Enter the name of the country \n Ex : United States, India, France...")
parser.add_argument("--year", help="Year for which to predict the temperature \n Ex : 2100, 2000...")
args = parser.parse_args()
Country = args.country

# Import the Climate change dataset
GlobalTemp_df = pd.read_csv("GlobalLandTemperaturesByCountry.csv")

# Extract rows which contain temperature values from the country specified
IndianTemp_df = GlobalTemp_df.loc[GlobalTemp_df['Country'] == Country]
IndianTemp_df = IndianTemp_df.iloc[:,[0,1]]
IndianTemp_df = IndianTemp_df.dropna()

#list to store all the years present in the dataset.
int_dt = []
for i in range(len(IndianTemp_df['dt'])):
    int_dt.append(int(IndianTemp_df['dt'].iloc[i][:4]))

#to make a new dataframe onlywith years and avg. temp.
IndianTemp = pd.DataFrame({'dt':int_dt, 'AverageTemperature': list(IndianTemp_df.iloc[:,1])})


dt_unique = np.unique(np.array(IndianTemp['dt']))

#just a framework for new dataframe which will store the avg of a particular year. 
IndianTemp_avg = pd.DataFrame(columns=['dt','AverageTemperature'])

for dt in dt_unique:

    IndianTemp_dt = IndianTemp.loc[IndianTemp['dt'] == dt]

    mean_temp = IndianTemp_dt['AverageTemperature'].mean(axis=0)

    temp_df = pd.DataFrame([[dt,mean_temp]], columns= ['dt','AverageTemperature'])
    
    IndianTemp_avg = IndianTemp_avg.append(temp_df)

#converting it into proper format to feed the algorithm
year = np.array(IndianTemp_avg['dt'])
year = year.reshape(len(year),1)
Avg_Temp = np.array(IndianTemp_avg['AverageTemperature'])
Avg_Temp = Avg_Temp.reshape(len(Avg_Temp), 1)


clf = LinearRegression()
clf.fit(year,Avg_Temp)
years = np.array([int(args.year)])
years = years.reshape(len(years), 1)
temp_pred = clf.predict(years)
print ("The average temperature for the year {} is {} degree Celsius".format(args.year, temp_pred))
