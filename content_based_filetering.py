import pandas
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#ignoring warnings
numpy.seterr(divide='ignore', invalid='ignore')

# get places data set
placesUrl = "./places.csv"
placesColumns = ['City_id', 'City', 'Type']
placesDataset = pandas.read_csv(placesUrl, names=placesColumns)

# get user rating for place
userRatingsUrl = "./user_place_rating.csv"
userRatingsColumns = ['User_id', 'City_id', 'Rating']
userRatingsDataset = pandas.read_csv(userRatingsUrl, names=userRatingsColumns)

# get matrix of places with tfId
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(placesDataset['Type'])
print(tfidf_matrix)

# get similarity between places based on the type
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#print(cosine_sim)

# build a 1-dimensional array with City names
citiesArray = placesDataset['City']
# create index for each city
indices = pandas.Series(placesDataset.index, index=placesDataset['City'])

# # Function that get City recommendations based on the cosine similarity score of City's type
def get_city_recommendations(city):
  idx = indices[city]
  sim_scores = list(enumerate(cosine_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  sim_scores = sim_scores[1:21]
  city_indices = [i[0] for i in sim_scores]
  return citiesArray.iloc[city_indices]

recommendationArray = get_city_recommendations('Detroit')
print('Recommendations::', recommendationArray)
