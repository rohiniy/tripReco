import pandas
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

varToPass = 'City'
#ignoring warnings
numpy.seterr(divide='ignore', invalid='ignore')

# get places data set
placesUrl = "./placesWithSpace.csv"
placesColumns = ['City_id', 'City', 'Type']
placesDataset = pandas.read_csv(placesUrl, names=placesColumns)

# get user rating for place
userRatingsUrl = "./user_place_rating.csv"
userRatingsColumns = ['User_id', 'City_id', 'Rating']
userRatingsDataset = pandas.read_csv(userRatingsUrl, names=userRatingsColumns)
# merging places and user rating
userRatingsDataset = pandas.merge(userRatingsDataset, placesDataset, on='City_id')

# mean of the ratings per city
ratings = pandas.DataFrame(userRatingsDataset.groupby(varToPass)['Rating'].mean())

# count of ratings per city
ratings['number_of_ratings'] = userRatingsDataset.groupby(varToPass)['Rating'].count()
# creating matrix
place_matrix = userRatingsDataset.pivot_table(index='User_id', columns=varToPass, values='Rating')

#get matrix of places with CountVectorizer
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(placesDataset['Type'])

# get similarity between places based on the type
cosine_sim = linear_kernel(count_matrix, count_matrix)

# create index for each city
indices = pandas.Series(placesDataset.index, index=placesDataset['City'])

# # Function that get City recommendations based on the cosine similarity score of City's type
def get_city_recommendations(city):
  idx = indices[city]
  simScores = list(enumerate(cosine_sim[idx]))
  simScores = sorted(simScores, key=lambda x: x[1], reverse=True)
  simScores = simScores[1:34]

  simScoresDataFrame = pandas.DataFrame(simScores)
  simScoresDataFrame[0] = simScoresDataFrame[0].apply(lambda x: x + 1)
  simScoresWithCityDataFrame = pandas.merge(simScoresDataFrame, placesDataset, right_on='City_id', left_on=0)
  return simScoresWithCityDataFrame[['City', 'City_id']]
