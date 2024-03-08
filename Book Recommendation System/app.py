import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    try:
        books_df = pd.read_csv('books.csv')
        return books_df
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV: {e}")
        return None

def preprocess_data(df):
    title_column = 'title' if 'title' in df.columns else 'your_actual_title_column_name'
    authors_column = 'authors' if 'authors' in df.columns else 'your_actual_authors_column_name'
    
    df['description'] = df[title_column] + ' ' + df[authors_column]
    df['description'] = df['description'].fillna('')

    return df

def compute_similarity(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim_matrix

def get_recommendations(book_title, cosine_sim_matrix, df, top_n=5):
    if df[df['title'] == book_title].empty:
        st.warning(f"No match found for '{book_title}'.")
        return pd.Series([])

    book_index = df[df['title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[book_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    book_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[book_indices]

def main():
    st.title('Book Recommendation System')
    ###
    # st.markdown('<link rel="stylesheet" type="text/css" href="static/styles.css">', unsafe_allow_html=True)

    with open('./static/styles.css') as f:
     css = f.read()

    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    ###


    books_df = load_data()
    if books_df is None:
        return

    books_df = preprocess_data(books_df)
    cosine_sim_matrix = compute_similarity(books_df)

    searched_book = st.text_input('Enter the book title:', 'The Hunger Games')

    if st.button('Get Recommendations'):
        recommendations = get_recommendations(searched_book, cosine_sim_matrix, books_df)
        st.write(f"Top 5 Recommendations for '{searched_book}':")
        st.write(recommendations)

if __name__ == '__main__':
    main()
