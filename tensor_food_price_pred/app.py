import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
training_data = pd.read_excel('dataset/Data_Train.xlsx')
test_data = pd.read_excel('dataset/Data_Test.xlsx')
training_data = pd.DataFrame(training_data,columns=['TITLE','RESTAURANT_ID','CUISINES', 'TIME','CITY','LOCALITY','RATING', 'VOTES','COST'])
test_data = pd.DataFrame(test_data,columns=['TITLE','RESTAURANT_ID','CUISINES', 'TIME','CITY','LOCALITY','RATING', 'VOTES','COST'])


# --------------CHECK FEATURES---------------

#Training Set

print("\nEDA on Training Set\n")
print("#"*30)
print("\nFeatures/Columns : \n", training_data.columns)
print("\n\nNumber of Features/Columns : ", len(training_data.columns))
print("\nNumber of Rows : ",len(training_data))
print("\n\nData Types :\n", training_data.dtypes)
print("\nContains NaN/Empty cells : ", training_data.isnull().values.any())
print("\nTotal empty cells by column :\n", training_data.isnull().sum(), "\n\n")


# Test Set
print("#"*30)
print("\nEDA on Test Set\n")
print("#"*30)
print("\nFeatures/Columns : \n",test_data.columns)
print("\n\nNumber of Features/Columns : ",len(test_data.columns))
print("\nNumber of Rows : ",len(test_data))
print("\n\nData Types :\n", test_data.dtypes)
print("\nContains NaN/Empty cells : ", test_data.isnull().values.any())
print("\nTotal empty cells by column :\n", test_data.isnull().sum())




###############################################################################################################################################

# Data Analysisng

###############################################################################################################################################


#Combining trainig set and test sets for analysing data and finding patterns

data_temp = [training_data[['TITLE', 'RESTAURANT_ID', 'CUISINES', 'TIME', 'CITY', 'LOCALITY','RATING', 'VOTES']],
             test_data]
data_temp = pd.concat(data_temp)

# Analysing TITLE
new_df = data_temp['TITLE'].str.split(',',expand=True)
data_temp['TITLE'] = pd.concat([new_df[[0]], new_df[1]], axis=1)
data_temp['TITLE'] = data_temp['TITLE'].str.upper()

print("\n\nNumber of Unique Titles : ", len(pd.Series(data_temp['TITLE']).unique()))
print("\n\nUnique Titles:\n", pd.Series(data_temp['TITLE']).unique())
all_titles = list(pd.Series(data_temp['TITLE']).unique())
all_titles.append('NONE')


# Analysing CUISINES
new_df1 = data_temp['CUISINES'].str.split(',',expand=True)
data_temp['CUISINES'] = pd.concat([new_df1[[0]], new_df1[1], new_df1[3], new_df1[4], new_df1[5], new_df1[6],
                                   new_df1[7]],axis=1)
data_temp['CUISINES'] = data_temp['CUISINES'].str.upper()

print("\n\nNumber of Unique CUISINES : ", len(pd.Series(data_temp['CUISINES']).unique()))
print("\n\nUnique CUISINES:\n", pd.Series(data_temp['CUISINES']).unique())
all_cuisines = list(pd.Series(data_temp['CUISINES']).unique())

# Analysing CITY
data_temp['CITY'] = data_temp['CITY'].str.upper()

print("\n\nNumber of Unique CITY : ", len(pd.Series(data_temp['CITY']).unique()))
print("\n\nUnique CITY:\n", pd.Series(data_temp['CITY']).unique())
all_cities = list(pd.Series(data_temp['CITY']).unique())

# Analysing LOCALITY
data_temp['LOCALITY'] = data_temp['LOCALITY'].str.upper()

print("\n\nNumber of Unique LOCALITY : ", len(pd.Series(data_temp['LOCALITY']).unique()))
print("\n\nUnique LOCALITY:\n", pd.Series(data_temp['LOCALITY']).unique())

all_localities = pd.Series(data_temp['LOCALITY']).unique()



###############################################################################################################################################

# Data Cleaning

###############################################################################################################################################


# Cleaning Training Set
#______________________

# TITLE - 2 titels

new_df_1 = training_data['TITLE'].str.split(',',expand=True)
training_data['TITLE1'] = new_df_1[0].str.upper()
training_data['TITLE2'] = new_df_1[1].str.upper()
training_data['TITLE2'] = training_data['TITLE2'].str.replace('None','NONE')

#Cleaning CUISINES
new_df_2 = training_data['CUISINES'].str.split(',',expand=True)
training_data['CUISINE1'] = new_df_2[0].str.upper()
training_data['CUISINE2'] = new_df_2[1].str.upper()
training_data['CUISINE3'] = new_df_2[2].str.upper()
training_data['CUISINE4'] = new_df_2[3].str.upper()
training_data['CUISINE5'] = new_df_2[4].str.upper()
training_data['CUISINE6'] = new_df_2[5].str.upper()
training_data['CUISINE7'] = new_df_2[6].str.upper()
training_data['CUISINE8'] = new_df_2[7].str.upper()
all_cuisines.append('None')

# Cleaning CITY
training_data['CITY'] = training_data['CITY'].str.upper()
training_data['CITY'].fillna('NOT AVAILABLE',inplace=True)

# Cleaning LOCALITY
training_data['LOCALITY'] = training_data['LOCALITY'].str.upper()
training_data['LOCALITY'].fillna('NOT AVAILABLE',inplace=True)

#Cleaning Rating
# training_data.dtypes
training_data['RATING'] = training_data['RATING'].str.replace('-','')
training_data['RATING'] = training_data['RATING'].str.replace('NEW','')
training_data['RATING'].fillna(0,inplace=True)
training_data['RATING'] = pd.to_numeric(training_data['RATING'])

# Votes
training_data['VOTES'].fillna('0 votes',inplace = True)
training_data['VOTES'] = training_data['VOTES'].str.replace('votes','')
training_data['VOTES'] = pd.to_numeric(training_data['VOTES'])


new_data_train = {}

new_data_train['TITLE1'] = training_data['TITLE1']
new_data_train['TITLE2'] = training_data['TITLE2']
new_data_train['RESTAURANT_ID'] = training_data["RESTAURANT_ID"]
new_data_train['CUISINE1'] = training_data['CUISINE1']
new_data_train['CUISINE2'] = training_data['CUISINE2']
new_data_train['CUISINE3'] = training_data['CUISINE3']
new_data_train['CUISINE4'] = training_data['CUISINE4']
new_data_train['CUISINE5'] = training_data['CUISINE5']
new_data_train['CUISINE6'] = training_data['CUISINE6']
new_data_train['CUISINE7'] = training_data['CUISINE7']
new_data_train['CUISINE8'] = training_data['CUISINE8']
new_data_train['CITY'] = training_data['CITY']
new_data_train['LOCALITY'] = training_data['LOCALITY']
new_data_train['RATING'] = training_data['RATING']
new_data_train['VOTES'] = training_data['VOTES']
new_data_train['COST'] = training_data["COST"]

new_data_train = pd.DataFrame(new_data_train)
new_data_train.head()


# Cleaning Test Set
#______________________

# TITLE - 2 titels

new_df_3 = test_data['TITLE'].str.split(',',expand=True)
test_data['TITLE1'] = new_df_3[0].str.upper()
test_data['TITLE2'] = new_df_3[1].str.upper()
test_data['TITLE2'] = test_data['TITLE2'].str.replace('None','NONE')
#Cleaning CUISINES
new_df_4 = test_data['CUISINES'].str.split(',',expand=True)
test_data['CUISINE1'] = new_df_4[0].str.upper()
test_data['CUISINE2'] = new_df_4[1].str.upper()
test_data['CUISINE3'] = new_df_4[2].str.upper()
test_data['CUISINE4'] = new_df_4[3].str.upper()
test_data['CUISINE5'] = new_df_4[4].str.upper()
test_data['CUISINE6'] = new_df_4[5].str.upper()
test_data['CUISINE7'] = new_df_4[6].str.upper()
test_data['CUISINE8'] = new_df_4[7].str.upper()

# Cleaning CITY
test_data['CITY'] = test_data['CITY'].str.upper()
test_data['CITY'].fillna('NOT AVAILABLE',inplace=True)

# Cleaning LOCALITY
test_data['LOCALITY'] = test_data['LOCALITY'].str.upper()
test_data['LOCALITY'].fillna('NOT AVAILABLE',inplace=True)

#Cleaning Rating
# training_data.dtypes
test_data['RATING'] = test_data['RATING'].str.replace('-','')
test_data['RATING'] = test_data['RATING'].str.replace('NEW','')
test_data['RATING'].fillna(0,inplace=True)
test_data['RATING'] = pd.to_numeric(test_data['RATING'])

# Votes
test_data['VOTES'].fillna('0 votes',inplace = True)
test_data['VOTES'] = test_data['VOTES'].str.replace('votes','')
test_data['VOTES'] = pd.to_numeric(test_data['VOTES'])


new_data_test = {}

new_data_test['TITLE1'] = test_data['TITLE1']
new_data_test['TITLE2'] = test_data['TITLE2']
new_data_test['RESTAURANT_ID'] = test_data["RESTAURANT_ID"]
new_data_test['CUISINE1'] = test_data['CUISINE1']
new_data_test['CUISINE2'] = test_data['CUISINE2']
new_data_test['CUISINE3'] = test_data['CUISINE3']
new_data_test['CUISINE4'] = test_data['CUISINE4']
new_data_test['CUISINE5'] = test_data['CUISINE5']
new_data_test['CUISINE6'] = test_data['CUISINE6']
new_data_test['CUISINE7'] = test_data['CUISINE7']
new_data_test['CUISINE8'] = test_data['CUISINE8']
new_data_test['CITY'] = test_data['CITY']
new_data_test['LOCALITY'] = test_data['LOCALITY']
new_data_test['RATING'] = test_data['RATING']
new_data_test['VOTES'] = test_data['VOTES']
new_data_test['COST'] = test_data["COST"]

new_data_test = pd.DataFrame(new_data_test)


def morning_data_train():
    new_data_train.loc[:, ('MORNING')] = 0
    for index, val in enumerate(training_data.loc[:, ('TIME')]):
        if any(s in val for s in ['am', 'AM', 'Am', '24']):
            new_data_train.loc[index, 'MORNING'] = 1


def evening_data_train():
    new_data_train.loc[:, ('EVENING')] = 0
    for index, val in enumerate(training_data.loc[:, ('TIME')][:6]):
        if any(s in val for s in ['pm', 'PM', 'Pm', 'noon', '24']):
            new_data_train.loc[index, 'EVENING'] = 1


morning_data_train()
evening_data_train()


def morning_data_test():
    new_data_test.loc[:, ('MORNING')] = 0
    for index, val in enumerate(test_data.loc[:, ('TIME')]):
        if any(s in val for s in ['am', 'AM', 'Am', '24']):
            new_data_test.loc[index, 'MORNING'] = 1


def evening_data_test():
    new_data_test.loc[:, ('EVENING')] = 0
    for index, val in enumerate(test_data.loc[:, ('TIME')][:6]):
        if any(s in val for s in ['pm', 'PM', 'Pm', 'noon', '24']):
            new_data_test.loc[index, 'EVENING'] = 1


morning_data_test()
evening_data_test()


le_titles = LabelEncoder()
le_cuisines = LabelEncoder()

le_city = LabelEncoder()

le_locality = LabelEncoder()

le_titles.fit(all_titles)
le_cuisines.fit(all_cuisines)

le_city.fit(all_cities)
# le_locality.fit(all_localities)
# new_data_train.TITLE2.fillna(value=pd.np.nan, inplace=True)
print(new_data_train['TITLE2'][0])

# Training Set
new_data_train['TITLE1'] = le_titles.transform(new_data_train['TITLE1'])
# new_data_train['TITLE2'] = le_titles.transform(new_data_train['TITLE2'])


# new_data_train['CUISINE1'] = le_cuisines.transform(new_data_train['CUISINE1'])
# new_data_train['CUISINE2'] = le_cuisines.transform(new_data_train['CUISINE2'])
# new_data_train['CUISINE3'] = le_cuisines.transform(new_data_train['CUISINE3'])
# new_data_train['CUISINE4'] = le_cuisines.transform(new_data_train['CUISINE4'])
# new_data_train['CUISINE5'] = le_cuisines.transform(new_data_train['CUISINE5'])
# new_data_train['CUISINE6'] = le_cuisines.transform(new_data_train['CUISINE6'])
# new_data_train['CUISINE7'] = le_cuisines.transform(new_data_train['CUISINE7'])
# new_data_train['CUISINE8'] = le_cuisines.transform(new_data_train['CUISINE8'])


# new_data_train['CITY'] = le_city.transform(new_data_train['CITY'])
# new_data_train['LOCALITY'] = le_locality.transform(new_data_train['LOCALITY'])

# # Test Set

# new_data_test['TITLE1'] = le_titles.transform(new_data_test['TITLE1'])
# new_data_test['TITLE2'] = le_titles.transform(new_data_test['TITLE2'])


# new_data_test['CUISINE1'] = le_cuisines.transform(new_data_test['CUISINE1'])
# new_data_test['CUISINE2'] = le_cuisines.transform(new_data_test['CUISINE2'])
# new_data_test['CUISINE3'] = le_cuisines.transform(new_data_test['CUISINE3'])
# new_data_test['CUISINE4'] = le_cuisines.transform(new_data_test['CUISINE4'])
# new_data_test['CUISINE5'] = le_cuisines.transform(new_data_test['CUISINE5'])
# new_data_test['CUISINE6'] = le_cuisines.transform(new_data_test['CUISINE6'])
# new_data_test['CUISINE7'] = le_cuisines.transform(new_data_test['CUISINE7'])
# new_data_test['CUISINE8'] = le_cuisines.transform(new_data_test['CUISINE8'])


# new_data_test['CITY'] = le_city.transform(new_data_test['CITY'])
# new_data_test['LOCALITY'] = le_locality.transform(new_data_test['LOCALITY'])


# # Classifying Independent and Dependent Features
# #_______________________________________________

# # Dependent Variable
# Y_train = new_data_train.iloc[:, -1].values

# # Independent Variables
# X_train = new_data_train.iloc[:,0 : -1].values

# # Independent Variables for Test Set
# X_test = new_data_test.iloc[:,:].values


# # Feature Scaling
# #________________

# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)

# X_test = sc.transform(X_test)


# Y_train = Y_train.reshape((len(Y_train), 1))

# Y_train = sc.fit_transform(Y_train)

# Y_train = Y_train.ravel()