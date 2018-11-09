import pandas
import numpy

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

# simple item based recommender system for New York
def simple_recommender(city):
    city_user_rating = place_matrix[city]
    similar_to_city = place_matrix.corrwith(city_user_rating)
    corr_city = pandas.DataFrame(similar_to_city, columns=['Correlation'])
    corr_city = corr_city.join(ratings['number_of_ratings'])
    return (corr_city[corr_city['number_of_ratings'] > 1].sort_values(by='Correlation', ascending=False))


