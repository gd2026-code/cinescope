# cinescope
import pandas as pd

# Load the dataset
movies = pd.read_csv(r"C:\Users\lenovo\Downloads\Datasets\Datasets\movies.csv")

# Check the first few rows to understand the structure
movies.head()

# Check the columns to see what data we have
movies.columns


movies['summary'].head()  

movies.head()
from textblob import TextBlob

# Define a function to get the sentiment polarity
def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

# Apply the function to the 'description' column
movies['sentiment'] = movies['summary'].apply(get_sentiment)

# Show the updated dataset with sentiment values
movies[['title_x', 'sentiment']].head()

import matplotlib.pyplot as plt

# Plotting sentiment distribution
plt.figure(figsize=(10,6))
plt.hist(movies['sentiment'], bins=50, color='skyblue', edgecolor='black')
plt.title('Sentiment Distribution of Movie Descriptions', fontsize=16)
plt.xlabel('Sentiment Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Assuming you have a 'rating' column
plt.figure(figsize=(10,6))
plt.scatter(movies['imdb_rating'], movies['sentiment'], alpha=0.6, color='orange')
plt.title('Sentiment vs imdb_rating', fontsize=16)
plt.xlabel('Movie Rating', fontsize=12)
plt.ylabel('Sentiment Score', fontsize=12)
plt.show()

# Sorting movies by highest sentiment
top_positive_movies = movies.sort_values(by='sentiment', ascending=False).head(10)

# Plotting
plt.figure(figsize=(10,6))
plt.barh(top_positive_movies['title_x'], top_positive_movies['sentiment'], color='green')
plt.title('Top 10 Movies with Positive Sentiment', fontsize=16)
plt.xlabel('Sentiment Score', fontsize=12)
plt.ylabel('Movie Title', fontsize=12)
plt.show()

# Assuming there's a 'genre' column
genre_sentiment = movies.groupby('genres')['sentiment'].mean().head(20)

# Plotting average sentiment by genre
genre_sentiment.sort_values().plot(kind='barh', figsize=(10,6), color='purple')
plt.title('Average Sentiment by genres', fontsize=16)
plt.xlabel('Average Sentiment', fontsize=12)
plt.ylabel('genres', fontsize=12)
plt.show()

movies.head(2)
pip install plotly

import plotly.express as px

# Drop rows with missing required fields
movies= movies.dropna(subset=['imdb_rating', 'imdb_votes', 'wins_nominations']).head(40)

fig = px.scatter_3d(
    movies,
    x='imdb_rating',
    y='imdb_votes',
    z='wins_nominations',
    color='imdb_votes',
    hover_name='title_x',
    title='Interactive 3D Scatter: imdb_rating vs imdb_votes vs wins_nominations',
    color_continuous_scale='RdBu'
)

fig.show()

movies.head(1)
print(movies.columns)

import plotly.graph_objects as go
import pandas as pd

# Filter and clean data
movies_clean = movies.dropna(subset=['imdb_rating', 'wins_nominations', 'year_of_release'])

# Create 3D box plot simulation using scatter for box outlines
fig = go.Figure()

# Box 1: imdb_rating vs year_of_release
fig.add_trace(go.Box(
    y=movies_clean['imdb_rating'],
    x=movies_clean['year_of_release'],
    name='IMDb vs Year',
    boxpoints='outliers',
    marker_color='orange'
))

# Box 2: wins_nominations vs year_of_release
fig.add_trace(go.Box(
    y=movies_clean['wins_nominations'],
    x=movies_clean['year_of_release'],
    name='Wins vs Year',
    boxpoints='outliers',
    marker_color='skyblue'
))

# Box 3: imdb_rating vs wins_nominations
fig.add_trace(go.Box(
    y=movies_clean['imdb_rating'],
    x=movies_clean['wins_nominations'],
    name='IMDb vs Wins',
    boxpoints='outliers',
    marker_color='green'
))

fig.update_layout(
    title='3D-style Box Plot of IMDb, Awards, and Year',
    xaxis_title='X Axis',
    yaxis_title='Y Axis',
    showlegend=True,
    template='plotly_dark'
)

fig.show()

import plotly.express as px
import pandas as pd

# Clean the dataset: ensure all required fields are available
movies_clean = movies.dropna(subset=['title_x', 'imdb_rating', 'wins_nominations', 'year_of_release', 'story']).tail(10)

# Create the 3D scatter plot
fig = px.scatter_3d(
    movies_clean,
    x='imdb_rating',
    y='wins_nominations',
    z='year_of_release',
    color='imdb_rating',  # color intensity based on rating
    hover_data=['title_x', 'story'],  # show title and story on hover
    title='3D Movie Visualization: Rating, Awards, and Release Year',
    labels={
        'imdb_rating': 'IMDb Rating',
        'wins_nominations': 'Wins/Nominations',
        'year_of_release': 'Release Year',
        'title_x': 'Title',
        'story': 'Story'
    },
    color_continuous_scale='Viridis'
)

fig.update_traces(marker=dict(size=5, opacity=0.8))
fig.update_layout(
    scene=dict(
        xaxis_title='IMDb Rating',
        yaxis_title='Wins/Nominations',
        zaxis_title='Year of Release'
    ),
    template='plotly_dark'
)

fig.show()


