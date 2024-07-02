from flask import Flask, request, jsonify
import pandas as pd
import json
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the JSON file
with open('Food.json', 'r') as file:
    food_data = json.load(file)

food_df = pd.DataFrame(food_data)
food_df['Food_ID'] = food_df.index

# Load the CSV file
ratings_df = pd.read_csv('ratings.csv')

# Merge the food data with ratings
merged_df = pd.merge(ratings_df, food_df, left_on='Food_ID', right_on='Food_ID')

# Create a pivot table
user_item_matrix = merged_df.pivot_table(index='User_ID', columns='Food_ID', values='Rating').fillna(0)

# Transpose the matrix to have food items as rows
food_user_matrix = user_item_matrix.T

# Fit the KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(food_user_matrix)

# Function to get recommendations
def get_recommendations(food_id, n_neighbors=5):
    distances, indices = model_knn.kneighbors(food_user_matrix.loc[food_id].values.reshape(1, -1), n_neighbors=n_neighbors + 1)
    recommendations = [food_user_matrix.index[i] for i in indices.flatten() if food_user_matrix.index[i] != food_id]
    return recommendations[:n_neighbors]

@app.route('/recommend', methods=['GET'])
def recommend():
    food_id = int(request.args.get('food_id'))
    recommendations = get_recommendations(food_id, n_neighbors=5)
    recommendations = [int(item) for item in recommendations]  # Convert to Python int
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)