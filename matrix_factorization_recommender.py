import pandas
import numpy
from scipy.sparse.linalg import svds

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
userRatingsPivot = userRatingsDataset.pivot(index = 'User_id', columns ='City_id', values = 'Rating').fillna(0)

userRatingsMatrix = userRatingsPivot.as_matrix()
userRatingsMean = numpy.mean(userRatingsMatrix, axis = 1)
userRatingsDemeaned = userRatingsMatrix - userRatingsMean.reshape(-1, 1)

U, sigma, Vt = svds(userRatingsDemeaned, k = 5)
sigma = numpy.diag(sigma)

all_user_predicted_ratings = numpy.dot(numpy.dot(U, sigma), Vt) + userRatingsMean.reshape(-1, 1)
preds_df = pandas.DataFrame(all_user_predicted_ratings, columns = userRatingsPivot.columns)

def recommendCities(predictions_df, userID, places_df, original_ratings_df, num_recommendations=5):
    # Get and sort the user's predictions
    user_row_number = userID - 1  # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the cities information.
    user_data = original_ratings_df[original_ratings_df.User_id == (userID)]
    user_full = (user_data.merge(places_df, how='left', left_on='City_id', right_on='City_id').
                 sort_values(['Rating'], ascending=False))

    # Recommend the highest predicted rating cities that the user hasn't seen yet.
    recommendations = (places_df[~places_df['City_id'].isin(user_full['City_id'])].
                           merge(pandas.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                 left_on='City_id',
                                 right_on='City_id').
                           rename(columns={user_row_number: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :-1]
                           )

    return recommendations[['City']]


def getRecommendedCities(userId):
    return recommendCities(preds_df, userId, placesDataset, userRatingsDataset, 10)
