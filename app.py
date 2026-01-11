from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

app = Flask(__name__)

# Sample training data
emails = [
    ("Win a free iPhone now!", 1),
    ("Limited offer, buy now!", 1),
    ("Meeting at 10am tomorrow", 0),
    ("Please find attached the report", 0),
]

texts, labels = zip(*emails)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        email_text = request.form["email"]
        X_input = vectorizer.transform([email_text])
        prediction = model.predict(X_input)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)


"""
    pip install flask scikit-learn
    python app.py
    
    
    ---  Spam Example  ---
    1)Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!
    2)Urgent! Your account has been compromised. Verify your password immediately.
    3)Earn money from home! No experience needed. Limited spots available. Sign up today!
    4)Get cheap meds online without prescription. Fast delivery, low prices!
    5)You've been selected for a FREE iPhone 14. Claim yours now!
    
    ---  Not Spam Example  ---
    1)Hi team, the meeting is scheduled for 2 PM tomorrow in the main conference room.

    2)Dear John, please find attached the sales report for Q2. Let me know if you have any questions.

    3)Your Amazon order has been shipped and will arrive by Thursday.

    4)Reminder: Your dentist appointment is on Monday at 10:00 AM.

    5)Reminder: Your dentist appointment is on Monday at 10:00 AM.


"""