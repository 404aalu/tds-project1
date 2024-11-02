from sklearn.linear_model import LinearRegression
import numpy as np

# Filter out rows where 'bio' is NaN (i.e., users without a bio)
users_with_bio = users_df[users_df['bio'].notna()]

# Calculate word count for each bio
users_with_bio['bio_word_count'] = users_with_bio['bio'].apply(lambda x: len(x.split()))

# Extract bio word counts and followers as separate arrays for regression
X = users_with_bio[['bio_word_count']]
y = users_with_bio['followers']

# Perform linear regression to get the slope
regression_model = LinearRegression().fit(X, y)
slope = regression_model.coef_[0]  # Get the slope of followers with respect to bio word count

print(slope)
