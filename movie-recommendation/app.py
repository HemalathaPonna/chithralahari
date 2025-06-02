from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

# Sample movie data - In production, you'd load this from a database or API
movies_data = [
    {
        'title': 'The Princess Diaries',
        'genre': 'Romance',
        'language': 'English',
        'region': 'US',
        'rating': 6.4,
        'description': 'A socially awkward teen discovers she is a princess',
        'year': 2001,
        'streaming': ['Disney+', 'Hulu']
    },
    {
        'title': 'Mean Girls',
        'genre': 'Comedy',
        'language': 'English',
        'region': 'US',
        'rating': 7.0,
        'description': 'New student navigates high school social hierarchy',
        'year': 2004,
        'streaming': ['Netflix', 'Amazon Prime']
    },
    {
        'title': 'La La Land',
        'genre': 'Romance',
        'language': 'English',
        'region': 'US',
        'rating': 8.0,
        'description': 'Jazz musician and actress fall in love in Los Angeles',
        'year': 2016,
        'streaming': ['Netflix', 'HBO Max']
    },
    {
        'title': 'Amélie',
        'genre': 'Romance',
        'language': 'French',
        'region': 'France',
        'rating': 8.3,
        'description': 'Shy waitress decides to help others find happiness',
        'year': 2001,
        'streaming': ['Amazon Prime', 'Criterion Channel']
    },
    {
        'title': 'Crazy Rich Asians',
        'genre': 'Romance',
        'language': 'English',
        'region': 'Singapore',
        'rating': 6.9,
        'description': 'Woman discovers her boyfriend is extremely wealthy',
        'year': 2018,
        'streaming': ['HBO Max', 'Hulu']
    },
    {
        'title': 'Parasite',
        'genre': 'Thriller',
        'language': 'Korean',
        'region': 'South Korea',
        'rating': 8.6,
        'description': 'Poor family infiltrates wealthy household',
        'year': 2019,
        'streaming': ['Hulu', 'Amazon Prime']
    },
    {
        'title': 'Your Name',
        'genre': 'Animation',
        'language': 'Japanese',
        'region': 'Japan',
        'rating': 8.2,
        'description': 'Two teenagers share a profound, magical connection',
        'year': 2016,
        'streaming': ['Funimation', 'Crunchyroll']
    },
    {
        'title': 'The Half of It',
        'genre': 'Romance',
        'language': 'English',
        'region': 'US',
        'rating': 6.9,
        'description': 'Shy student helps jock woo a girl they both like',
        'year': 2020,
        'streaming': ['Netflix']
    },
    {
        'title': 'To All the Boys I\'ve Loved Before',
        'genre': 'Romance',
        'language': 'English',
        'region': 'US',
        'rating': 7.0,
        'description': 'Teen\'s secret love letters get mailed out',
        'year': 2018,
        'streaming': ['Netflix']
    },
    {
        'title': 'Clueless',
        'genre': 'Comedy',
        'language': 'English',
        'region': 'US',
        'rating': 6.9,
        'description': 'Popular high schooler plays matchmaker',
        'year': 1995,
        'streaming': ['Paramount+', 'Amazon Prime']
    },
    {
        'title': 'Spirited Away',
        'genre': 'Animation',
        'language': 'Japanese',
        'region': 'Japan',
        'rating': 9.2,
        'description': 'Girl enters spirit world to save her parents',
        'year': 2001,
        'streaming': ['HBO Max', 'Netflix']
    },
    {
        'title': 'Pride and Prejudice',
        'genre': 'Romance',
        'language': 'English',
        'region': 'UK',
        'rating': 7.8,
        'description': 'Elizabeth Bennet and Mr. Darcy\'s complex romance',
        'year': 2005,
        'streaming': ['Netflix', 'Amazon Prime']
    },
    {
        'title': 'Mamma Mia!',
        'genre': 'Musical',
        'language': 'English',
        'region': 'UK',
        'rating': 6.4,
        'description': 'Wedding preparations with ABBA songs',
        'year': 2008,
        'streaming': ['Amazon Prime', 'Hulu']
    },
    {
        'title': 'The Proposal',
        'genre': 'Romance',
        'language': 'English',
        'region': 'US',
        'rating': 6.7,
        'description': 'Boss forces assistant into fake engagement',
        'year': 2009,
        'streaming': ['Disney+', 'Hulu']
    },
    {
        'title': 'Legally Blonde',
        'genre': 'Comedy',
        'language': 'English',
        'region': 'US',
        'rating': 6.4,
        'description': 'Sorority girl enrolls at Harvard Law School',
        'year': 2001,
        'streaming': ['Netflix', 'Amazon Prime']
    },
     {
        'title': 'The Princess Diaries',
        'genre': 'Romance',
        'language': 'English',
        'region': 'US',
        'rating': 6.4,
        'description': 'A socially awkward teen discovers she is a princess',
        'year': 2001,
        'streaming': ['Disney+', 'Hulu']
    },
    
    # Telugu Movies
    {
        'title': 'Baahubali: The Beginning',
        'genre': 'Action',
        'language': 'Telugu',
        'region': 'India',
        'rating': 8.0,
        'description': 'An epic tale of two brothers fighting for the throne of Mahishmati',
        'year': 2015,
        'streaming': ['Netflix', 'Amazon Prime']
    },
    {
        'title': 'Baahubali 2: The Conclusion',
        'genre': 'Action',
        'language': 'Telugu',
        'region': 'India',
        'rating': 8.2,
        'description': 'The conclusion of the epic Baahubali saga',
        'year': 2017,
        'streaming': ['Netflix', 'Amazon Prime']
    },
    {
        'title': 'Arjun Reddy',
        'genre': 'Romance',
        'language': 'Telugu',
        'region': 'India',
        'rating': 8.1,
        'description': 'A surgeon with anger management issues struggles with heartbreak',
        'year': 2017,
        'streaming': ['Netflix', 'Hotstar']
    },
    {
        'title': 'Pushpa: The Rise',
        'genre': 'Action',
        'language': 'Telugu',
        'region': 'India',
        'rating': 7.6,
        'description': 'A laborer rises through the ranks of a red sandalwood smuggling syndicate',
        'year': 2021,
        'streaming': ['Amazon Prime', 'Netflix']
    },
    {
        'title': 'Rangasthalam',
        'genre': 'Action',
        'language': 'Telugu',
        'region': 'India',
        'rating': 8.2,
        'description': 'A partially deaf man fights against corruption in his village',
        'year': 2018,
        'streaming': ['Amazon Prime', 'Hotstar']
    },
    {
        'title': 'Ala Vaikunthapurramuloo',
        'genre': 'Comedy',
        'language': 'Telugu',
        'region': 'India',
        'rating': 7.3,
        'description': 'A young man discovers his true parentage and fights for justice',
        'year': 2020,
        'streaming': ['Netflix', 'Amazon Prime']
    },
    {
        'title': 'Jersey',
        'genre': 'Sports',
        'language': 'Telugu',
        'region': 'India',
        'rating': 8.5,
        'description': 'A failed cricketer attempts a comeback to fulfill his son\'s dream',
        'year': 2019,
        'streaming': ['Netflix', 'Hotstar']
    },
    {
        'title': 'Fidaa',
        'genre': 'Romance',
        'language': 'Telugu',
        'region': 'India',
        'rating': 7.7,
        'description': 'A young man falls in love with a free-spirited village girl',
        'year': 2017,
        'streaming': ['Amazon Prime', 'Hotstar']
    },
    
    # Other Indian Movies (Hindi, Tamil, Malayalam)
    {
        'title': '3 Idiots',
        'genre': 'Comedy',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.4,
        'description': 'Three friends navigate the pressures of engineering college',
        'year': 2009,
        'streaming': ['Netflix', 'Amazon Prime']
    },
    {
        'title': 'Dangal',
        'genre': 'Sports',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.4,
        'description': 'A wrestler trains his daughters to become world-class wrestlers',
        'year': 2016,
        'streaming': ['Netflix', 'Hotstar']
    },
    {
        'title': 'Zindagi Na Milegi Dobara',
        'genre': 'Adventure',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.2,
        'description': 'Three friends go on a bachelor trip across Spain',
        'year': 2011,
        'streaming': ['Netflix', 'Amazon Prime']
    },
    {
        'title': 'KGF: Chapter 1',
        'genre': 'Action',
        'language': 'Kannada',
        'region': 'India',
        'rating': 8.2,
        'description': 'A young man rises to power in the Kolar Gold Fields',
        'year': 2018,
        'streaming': ['Amazon Prime', 'Netflix']
    },
    {
        'title': 'Vikram',
        'genre': 'Thriller',
        'language': 'Tamil',
        'region': 'India',
        'rating': 8.4,
        'description': 'A black-ops agent seeks revenge for his son\'s murder',
        'year': 2022,
        'streaming': ['Hotstar', 'Netflix']
    },
    {
        'title': 'Drishyam',
        'genre': 'Crime',
        'language': 'Malayalam',
        'region': 'India',
        'rating': 8.6,
        'description': 'A man protects his family from a murder investigation',
        'year': 2013,
        'streaming': ['Hotstar', 'Amazon Prime']
    },
    {
        'title': 'Tumhari Sulu',
        'genre': 'Drama',
        'language': 'Hindi',
        'region': 'India',
        'rating': 7.1,
        'description': 'A housewife becomes a radio jockey and transforms her life',
        'year': 2017,
        'streaming': ['Netflix', 'Amazon Prime']
    },
    
    # Netflix Originals and Popular Netflix Movies
    {
        'title': 'Stranger Things',
        'genre': 'Sci-Fi',
        'language': 'English',
        'region': 'US',
        'rating': 8.7,
        'description': 'Kids in a small town encounter supernatural forces',
        'year': 2016,
        'streaming': ['Netflix']
    },
    {
        'title': 'The Crown',
        'genre': 'Biography',
        'language': 'English',
        'region': 'UK',
        'rating': 8.7,
        'description': 'The reign of Queen Elizabeth II from the 1940s to modern times',
        'year': 2016,
        'streaming': ['Netflix']
    },
    {
        'title': 'Money Heist',
        'genre': 'Thriller',
        'language': 'Spanish',
        'region': 'Spain',
        'rating': 8.3,
        'description': 'A criminal mastermind leads a team in the biggest heist in history',
        'year': 2017,
        'streaming': ['Netflix']
    },
    {
        'title': 'Squid Game',
        'genre': 'Thriller',
        'language': 'Korean',
        'region': 'South Korea',
        'rating': 8.0,
        'description': 'Desperate people compete in deadly children\'s games for money',
        'year': 2021,
        'streaming': ['Netflix']
    },
    {
        'title': 'The Witcher',
        'genre': 'Fantasy',
        'language': 'English',
        'region': 'US',
        'rating': 8.2,
        'description': 'A monster hunter navigates a world of magic and political intrigue',
        'year': 2019,
        'streaming': ['Netflix']
    },
    {
        'title': 'Extraction',
        'genre': 'Action',
        'language': 'English',
        'region': 'US',
        'rating': 6.8,
        'description': 'A mercenary is hired to rescue a crime lord\'s kidnapped son',
        'year': 2020,
        'streaming': ['Netflix']
    },
    {
        'title': 'The Irishman',
        'genre': 'Crime',
        'language': 'English',
        'region': 'US',
        'rating': 7.8,
        'description': 'An aging hitman recalls his involvement with the mob',
        'year': 2019,
        'streaming': ['Netflix']
    },
    {
        'title': 'Roma',
        'genre': 'Drama',
        'language': 'Spanish',
        'region': 'Mexico',
        'rating': 7.7,
        'description': 'A year in the life of a middle-class family in Mexico City',
        'year': 2018,
        'streaming': ['Netflix']
    },
    {
        'title': 'Bird Box',
        'genre': 'Horror',
        'language': 'English',
        'region': 'US',
        'rating': 6.6,
        'description': 'A woman and her children navigate a post-apocalyptic world blindfolded',
        'year': 2018,
        'streaming': ['Netflix']
    },
    {
        'title': 'The Platform',
        'genre': 'Sci-Fi',
        'language': 'Spanish',
        'region': 'Spain',
        'rating': 7.0,
        'description': 'Prisoners in a vertical facility with limited food must survive',
        'year': 2019,
        'streaming': ['Netflix']
    },
    
    # Additional International Netflix Content
    {
        'title': 'Dark',
        'genre': 'Sci-Fi',
        'language': 'German',
        'region': 'Germany',
        'rating': 8.8,
        'description': 'Time travel and family secrets in a small German town',
        'year': 2017,
        'streaming': ['Netflix']
    },
    {
        'title': 'Lupin',
        'genre': 'Crime',
        'language': 'French',
        'region': 'France',
        'rating': 7.5,
        'description': 'A master thief seeks revenge for his father\'s death',
        'year': 2021,
        'streaming': ['Netflix']
    },
    {
        'title': 'Sacred Games',
        'genre': 'Crime',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.6,
        'description': 'A Mumbai police officer races to save the city from destruction',
        'year': 2018,
        'streaming': ['Netflix']
    },
    {
        'title': 'Delhi Crime',
        'genre': 'Crime',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.5,
        'description': 'Based on the investigation of the 2012 Delhi gang rape case',
        'year': 2019,
        'streaming': ['Netflix']
    }, {
        'title': 'Delhi Crime',
        'genre': 'Crime',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.5,
        'description': 'Based on the investigation of the 2012 Delhi gang rape case',
        'year': 2019,
        'streaming': ['Netflix']
    },
    
    {
        'title': 'Spirited Away',
        'genre': 'Fantasy',
        'language': 'Japanese',
        'region': 'Japan',
        'rating': 8.6,
        'description': 'A girl enters the world of spirits to rescue her parents',
        'year': 2001,
        'streaming': ['Netflix']
    },
    {
        'title': 'The Godfather',
        'genre': 'Crime',
        'language': 'English',
        'region': 'USA',
        'rating': 9.2,
        'description': 'The aging patriarch of an organized crime dynasty transfers control to his reluctant son',
        'year': 1972,
        'streaming': ['Amazon Prime']
    },
    {
        'title': 'Amélie',
        'genre': 'Romance',
        'language': 'French',
        'region': 'France',
        'rating': 8.3,
        'description': 'A whimsical Parisian girl improves the lives of those around her',
        'year': 2001,
        'streaming': ['Netflix']
    },
    {
        'title': '3 Idiots',
        'genre': 'Comedy',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.4,
        'description': 'Three engineering students learn life lessons during college',
        'year': 2009,
        'streaming': ['Amazon Prime']
    },
    {
        'title': 'Pan\'s Labyrinth',
        'genre': 'Fantasy',
        'language': 'Spanish',
        'region': 'Mexico',
        'rating': 8.2,
        'description': 'A young girl escapes to a fantasy world during the Spanish Civil War',
        'year': 2006,
        'streaming': ['HBO Max']
    },
    {
        'title': 'The Lives of Others',
        'genre': 'Drama',
        'language': 'German',
        'region': 'Germany',
        'rating': 8.4,
        'description': 'An East German officer becomes absorbed in the lives of those he spies on',
        'year': 2006,
        'streaming': ['Amazon Prime']
    },
    {
        'title': 'Coco',
        'genre': 'Animation',
        'language': 'English',
        'region': 'USA',
        'rating': 8.4,
        'description': 'A boy journeys into the Land of the Dead to meet his ancestors',
        'year': 2017,
        'streaming': ['Disney+']
    },
    {
        'title': 'Your Name',
        'genre': 'Romance',
        'language': 'Japanese',
        'region': 'Japan',
        'rating': 8.4,
        'description': 'Two teenagers mysteriously swap bodies and connect across time',
        'year': 2016,
        'streaming': ['Crunchyroll', 'Netflix']
    },
    {
        'title': 'Roma',
        'genre': 'Drama',
        'language': 'Spanish',
        'region': 'Mexico',
        'rating': 7.7,
        'description': 'A year in the life of a housekeeper in 1970s Mexico City',
        'year': 2018,
        'streaming': ['Netflix']
    },
    {
        'title': 'The Intouchables',
        'genre': 'Drama',
        'language': 'French',
        'region': 'France',
        'rating': 8.5,
        'description': 'An aristocrat hires a young man from the projects as his caregiver',
        'year': 2011,
        'streaming': ['Amazon Prime']
    },
    {
        'title': 'Lagaan',
        'genre': 'Sports',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.1,
        'description': 'Villagers challenge British officers to a cricket match to avoid taxes',
        'year': 2001,
        'streaming': ['Netflix']
    },
    {
        'title': 'City of God',
        'genre': 'Crime',
        'language': 'Portuguese',
        'region': 'Brazil',
        'rating': 8.6,
        'description': 'Two boys choose different paths amid gang wars in Rio de Janeiro',
        'year': 2002,
        'streaming': ['HBO Max']
    },
    {
        'title': 'Dangal',
        'genre': 'Biography',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.3,
        'description': 'A former wrestler trains his daughters to become champions',
        'year': 2016,
        'streaming': ['Netflix']
    },
    {
        'title': 'Oldboy',
        'genre': 'Thriller',
        'language': 'Korean',
        'region': 'South Korea',
        'rating': 8.4,
        'description': 'A man seeks revenge after being imprisoned for 15 years',
        'year': 2003,
        'streaming': ['Netflix']
    },
    {
        'title': 'Cinema Paradiso',
        'genre': 'Drama',
        'language': 'Italian',
        'region': 'Italy',
        'rating': 8.5,
        'description': 'A filmmaker recalls his childhood love for cinema',
        'year': 1988,
        'streaming': ['Amazon Prime']
    },
    {
        'title': 'PK',
        'genre': 'Sci-Fi',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.1,
        'description': 'An alien questions religious dogmas on Earth',
        'year': 2014,
        'streaming': ['Netflix']
    },
    {
        'title': 'Get Out',
        'genre': 'Horror',
        'language': 'English',
        'region': 'USA',
        'rating': 7.7,
        'description': 'A Black man uncovers disturbing secrets about his girlfriend’s family',
        'year': 2017,
        'streaming': ['Amazon Prime']
    },
    {
        'title': 'The Great Beauty',
        'genre': 'Drama',
        'language': 'Italian',
        'region': 'Italy',
        'rating': 7.7,
        'description': 'An aging writer reflects on his life in Rome’s high society',
        'year': 2013,
        'streaming': ['Criterion Channel']
    },
    {
        'title': 'Bajrangi Bhaijaan',
        'genre': 'Adventure',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.0,
        'description': 'A man helps a mute Pakistani girl reunite with her family',
        'year': 2015,
        'streaming': ['Disney+ Hotstar']
    },
    {
        'title': 'Jallikattu',
        'genre': 'Action',
        'language': 'Malayalam',
        'region': 'India',
        'rating': 7.5,
        'description': 'A bull escapes and chaos erupts in a remote village',
        'year': 2019,
        'streaming': ['Amazon Prime']
    },
    {
        'title': 'The Lunchbox',
        'genre': 'Romance',
        'language': 'Hindi',
        'region': 'India',
        'rating': 7.8,
        'description': 'A mistaken lunchbox delivery sparks an unexpected relationship',
        'year': 2013,
        'streaming': ['Netflix']
    },
    {
        'title': 'La Haine',
        'genre': 'Crime',
        'language': 'French',
        'region': 'France',
        'rating': 8.0,
        'description': 'Three young men navigate tensions in the Parisian suburbs',
        'year': 1995,
        'streaming': ['Criterion Channel']
    },
    {
        'title': 'Sardar Udham',
        'genre': 'Historical',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.4,
        'description': 'The life of Indian revolutionary Udham Singh who assassinated Michael O\'Dwyer',
        'year': 2021,
        'streaming': ['Amazon Prime']
    },
    {
        'title': 'Rang De Basanti',
        'genre': 'Drama',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.1,
        'description': 'Youth inspired by freedom fighters awaken to contemporary issues',
        'year': 2006,
        'streaming': ['Netflix']
    },
    {
        'title': 'Barfi!',
        'genre': 'Romantic Comedy',
        'language': 'Hindi',
        'region': 'India',
        'rating': 8.1,
        'description': 'A deaf-mute boy and an autistic girl experience an unconventional love story',
        'year': 2012,
        'streaming': ['Netflix']
    }
]

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(movies_data)

class MovieRecommendationSystem:
    def __init__(self, movies_df):
        self.movies_df = movies_df
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(movies_df['description'])
        
    def get_recommendations(self, genre=None, language=None, region=None, min_rating=0):
        # Filter movies based on criteria
        filtered_movies = self.movies_df.copy()
        
        if genre and genre != 'Any':
            filtered_movies = filtered_movies[filtered_movies['genre'] == genre]
        
        if language and language != 'Any':
            filtered_movies = filtered_movies[filtered_movies['language'] == language]
            
        if region and region != 'Any':
            filtered_movies = filtered_movies[filtered_movies['region'] == region]
            
        if min_rating:
            filtered_movies = filtered_movies[filtered_movies['rating'] >= float(min_rating)]
        
        if filtered_movies.empty:
            return []
        
        # Sort by rating and return top recommendations
        recommendations = filtered_movies.sort_values('rating', ascending=False).head(6)
        return recommendations.to_dict('records')

# Initialize recommendation system
recommender = MovieRecommendationSystem(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    genre = data.get('genre')
    language = data.get('language')
    region = data.get('region')
    min_rating = data.get('rating')
    
    recommendations = recommender.get_recommendations(genre, language, region, min_rating)
    
    return jsonify(recommendations)

@app.route('/filters')
def get_filters():
    genres = sorted(df['genre'].unique().tolist())
    languages = sorted(df['language'].unique().tolist())
    regions = sorted(df['region'].unique().tolist())
    
    return jsonify({
        'genres': genres,
        'languages': languages,
        'regions': regions
    })

if __name__ == '__main__':
    app.run(debug=True)