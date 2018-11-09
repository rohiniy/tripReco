import pandas
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD, evaluate
from content_based_filetering import get_city_recommendations

varToPass = 'City'
#ignoring warnings
numpy.seterr(divide='ignore', invalid='ignore')

reader = Reader()

# get places data set
placesUrl = "./placesWithSpace.csv"
placesColumns = ['City_id', 'City', 'Type']
placesDataset = pandas.read_csv(placesUrl, names=placesColumns)

# get user rating for place
userRatingsUrl = "./user_place_rating.csv"
userRatingsColumns = ['User_id', 'City_id', 'Rating']
userRatingsDataset = pandas.read_csv(userRatingsUrl, names=userRatingsColumns)
# merging places and user rating
#userRatingsDataset = pandas.merge(userRatingsDataset, placesDataset, on='City_id')
user_rating_matrix = userRatingsDataset.pivot_table(index='User_id', columns=varToPass, values='Rating')


def hybridRecommendation(userId, city):
    #cityIndex = placesDataset.loc[placesDataset['City'] == city, 'City_id']
    contentBasedRecommendations = get_city_recommendations(city)