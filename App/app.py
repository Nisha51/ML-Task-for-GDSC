#Please Install The Dependencies - Pandas, Scikit-learn, flask
#Since we weren't provided any datasets, this is built on a randomly generated dataset
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

num_of_rec = 20
app = Flask(__name__)

pd.options.mode.chained_assignment = None
#Loading data into pandas
def load_data(data):
    df = pd.read_csv(data)
    return df
#Vectorizing our data and building cosine similarity matrix
def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat
#Main function
def recommend(career_goal, bg, subject):
    df = load_data('udemy_courses_random.csv')
    new_row = {'course_title': career_goal, 'level': bg, 'subject': subject}
    df.loc[len(df)] = new_row
    df.loc[:,'calc'] = df['course_title']+' '+df['subject']+' '+df['level']
    cosine_sim_mat = vectorize_text_to_cosine_mat(df['calc'])
    idx = len(df) - 1
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_scores = [i[1] for i in sim_scores[1:]]
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommended_courses = result_df[['course_title', 'similarity_score', 'price', 'num_subscribers']]
    return(final_recommended_courses.head(num_of_rec).to_html())

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    if request.method == 'POST':
        career_goal = request.form['Career_goal']
        subject = request.form['Subject']
        background = request.form['Background']
        recommendations = recommend(career_goal, background, subject)
    return render_template('index.html', recommendations=recommendations)
#This should provide an ip in the terminal to access the app
if __name__ == '__main__':
    app.run(port=8000, debug=False)
    #recommend()
