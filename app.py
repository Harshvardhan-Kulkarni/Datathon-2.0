# app.py

from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Read data from CSV file
file_path = 'Ai&DS.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Map categorical features to numerical values
difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}

# Function to run K-means clustering and display results
def run_kmeans(subject, platform, difficulty, duration, rating):
    selected_subject_df = df[(df['Subject'] == subject) & (df['Platform'] == platform)]

    selected_subject_df['Difficulty'] = selected_subject_df['Difficulty'].map(difficulty_mapping)

    features = selected_subject_df[['Difficulty', 'Duration', 'Rating']]

    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    kmeans = KMeans(n_clusters=4, random_state=50)
    selected_subject_df['Cluster'] = kmeans.fit_predict(features_scaled)

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    selected_subject_df['PCA1'] = features_pca[:, 0]
    selected_subject_df['PCA2'] = features_pca[:, 1]

    user_input = pd.DataFrame({
        'Difficulty': [difficulty],
        'Duration': [duration],
        'Rating': [rating]
    })

    user_input['Difficulty'] = user_input['Difficulty'].map(difficulty_mapping)

    user_input_imputed = imputer.transform(user_input)
    user_scaled = scaler.transform(user_input_imputed)

    user_cluster = kmeans.predict(user_scaled)

    plt.figure(figsize=(10, 6))
    for cluster in selected_subject_df['Cluster'].unique():
        cluster_data = selected_subject_df[selected_subject_df['Cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=100, c='red',
                label='Centroids')
    plt.scatter(user_scaled[:, 0], user_scaled[:, 1], marker='*', s=100, c='green', label='User Input')
    plt.title(f'K-means Clustering of {subject} Courses on {platform} with User Input')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()

    # Save the plot to a BytesIO object
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_data = base64.b64encode(img_stream.read()).decode('utf-8')
    plt.close()

    return img_data, selected_subject_df[selected_subject_df['Cluster'] == user_cluster[0]]


@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    recommended_courses_df = None

    if request.method == 'POST':
        subject = request.form['subject']
        platform = request.form['platform']
        difficulty = request.form['difficulty']
        duration = int(request.form['duration'])
        rating = float(request.form['rating'])

        img_data, recommended_courses_df = run_kmeans(subject, platform, difficulty, duration, rating)

    return render_template('index.html', img_data=img_data, recommended_courses_df=recommended_courses_df)


@app.route('/browse')
def browse():
    return render_template('browse.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
