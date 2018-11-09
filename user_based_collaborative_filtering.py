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

data = Dataset.load_from_df(userRatingsDataset[['User_id', 'City_id', 'Rating']], reader)
data.split(n_folds=5)

svd = SVD()
trainset = data.build_full_trainset()
svd.fit(trainset)

def hybridRecommendation(userId, city):
    #cityIndex = placesDataset.loc[placesDataset['City'] == city, 'City_id']
    contentBasedRecommendations = get_city_recommendations(city)
    #print(contentBasedRecommendations)
    contentBasedRecommendations['est'] = \
        contentBasedRecommendations['City_id'].apply(
            lambda x: svd.predict(userId, placesDataset.loc[x]['City_id']).est)
    contentBasedRecommendations = contentBasedRecommendations.sort_values('est', ascending=False)
    return contentBasedRecommendations.head(33)

userId = 3
city = 'New York'
print('---- Hybrid recommendation of the userId = '+ str(userId) + ' for city ' + city + ' ----')
print(hybridRecommendation(userId, 'New York'))
print('----- Hybrid recommendation end -----')
