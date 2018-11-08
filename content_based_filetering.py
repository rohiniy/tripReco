import pandas
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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
ratings = pandas.DataFrame(userRatingsDataset.groupby('City')['Rating'].mean())
#print(ratings)

# count of ratings per city
ratings['number_of_ratings'] = userRatingsDataset.groupby('City')['Rating'].count()
# creating matrix
place_matrix = userRatingsDataset.pivot_table(index='User_id', columns='City', values='Rating')
#print(place_matrix)

#get matrix of places with CountVectorizer
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(placesDataset['Type'])
#print(count_matrix)


# get matrix of places with tfId
#tfidf_matrix = tf.fit_transform(placesDataset['Type'])
#print(tfidf_matrix)

# get similarity between places based on the type
cosine_sim = linear_kernel(count_matrix, count_matrix)
print(cosine_sim)

# build a 1-dimensional array with City names
cities = placesDataset['City']
# create index for each city
indices = pandas.Series(placesDataset.index, index=placesDataset['City'])

# # Function that get City recommendations based on the cosine similarity score of City's type
def get_city_recommendations(city):
  idx = indices[city]
  sim_scores = list(enumerate(cosine_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  sim_scores = sim_scores[1:33]
  #print(sim_scores)
  city_indices = [i[0] for i in sim_scores]

  # simple item based recommender system for New York
  city_index = placesDataset.loc[placesDataset['City'] == city, 'City_id']
  #print(city_index)
  city_user_rating = place_matrix[city]
  similar_to_city = place_matrix.corrwith(city_user_rating)
  corr_city = pandas.DataFrame(similar_to_city, columns=['Correlation'])
  #corr_city = corr_city.join(ratings['number_of_ratings'])
  #print(corr_city.sort_values(by='Correlation', ascending=False))

  simScoresDataFrame = pandas.DataFrame(sim_scores)
  #typeSimScoresRatingCorrDataset = pandas.merge(corr_city, simScoresDataFrame, on='City_id')
  #print(typeSimScoresRatingCorrDataset)
  return cities.iloc[city_indices]

recommendationArray = get_city_recommendations('Las Vegas')
#print('Recommendations::', recommendationArray)


