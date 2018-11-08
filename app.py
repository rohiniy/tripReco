import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# get places data set
placesUrl = "./places.csv"
placesColumns = ['City_id', 'City', 'Type']
placesDataset = pandas.read_csv(placesUrl, names=placesColumns)

# get user rating for place
userRatingsUrl = "./user_place_rating.csv"
userRatingsColumns = ['User_id', 'City_id', 'Rating']
userRatingsDataset = pandas.read_csv(userRatingsUrl, names=userRatingsColumns)

# merging places and user rating
userRatingsDataset = pandas.merge(userRatingsDataset, placesDataset, on='City_id')

# mean of the ratings per city
ratings = pandas.DataFrame(userRatingsDataset.groupby('City')['Rating'].mean())
#print(ratings)

# count of ratings per city
ratings['number_of_ratings'] = userRatingsDataset.groupby('City')['Rating'].count()
#print(ratings)

# creating matrix
place_matrix = userRatingsDataset.pivot_table(index='User_id', columns='City', values='Rating')
#print(place_matrix)

#finding places with similar ratings
new_york_user_rating = place_matrix['New York']
print(new_york_user_rating)
similar_to_new_york = place_matrix.corrwith(new_york_user_rating)
print(similar_to_new_york.head())
