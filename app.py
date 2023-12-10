from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import numpy as np
import joblib

loaded_model = joblib.load('./pkl_objects/model.pkl')
loaded_stop = joblib.load('./pkl_objects/stopwords.pkl')
loaded_vec = joblib.load('./pkl_objects/vectorizer.pkl')

app = Flask(__name__)


def classify(document):
    label = {0: 'thrilled', 1: 'insightful feedback', 2: 'mixed emotions', 3: 'opportunity areas'}
    X = loaded_vec.transform([document])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return label[y], proba


class ReviewForm(Form):
    text_review = TextAreaField('', [validators.DataRequired(), validators.length(min=50)])


@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)


@app.route('/results/', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['text_review']
        y, proba = classify(review)
        return render_template('results.html', content=review, prediction=y, probability=round(proba * 100, 2))

    return render_template('reviewform.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
