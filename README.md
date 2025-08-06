**📱 SMS Spam & Not Spam Detection System**
Smart | Accurate | Secure
Detects spam messages with machine learning AND sender validation. Real-world, hybrid solution for digital communication safety.

**🚀 Project Overview**
Spam texts are a daily nuisance and a security risk. Our project builds a hybrid machine learning model (Naive Bayes + Header Validation) to automatically flag SMS as spam/not spam—combining message content intelligence and sender authenticity. Trained on real-world datasets, with an accuracy of 96%, integrated into a slick Flask web app.

**🌟 Features**
Hybrid Detection: Combines ML (Naive Bayes, TF-IDF) + rule-based sender header validation.
High Accuracy: 96% overall—with zero false positives!
Live Demo Web App: User can enter message & header and get instant result.
Easy Expand: Clean codebase—for new languages, better models, or SMS/email integration.
Beautiful Visualizations: WordClouds and Confusion Matrix for data explanation.
User Friendly: Simple interface (HTML/CSS + Flask API).

**🛠️ Project Structure**
├── app.py                  # Flask Web App
├── spam.csv                # SMS Spam Dataset
├── list_header.xlsx        # Trusted Headers
├── templates/
│   └── index.html          # Web UI
├── static/
│   └── style.css           # Styling
├── spam_model.pkl          # Trained Model
├── count_vectorizer.pkl    # Saved Vectorizer
├── README.md
└── ...

**⚡ Quickstart**
Clone this repo

1.text
git clone https://github.com/yourusername/sms-spam-detector-hybrid.git
cd sms-spam-detector-hybrid
Install requirements

2.text
pip install -r requirements.txt
(Or run the script — it will prompt to auto-install.)

3.Run the app

4.text
python app.py
Visit in your browser: http://127.0.0.1:5000

**💡 Usage**
Enter SMS sender header (from your message).
Enter the text content of the SMS.
Click Check Spam.
See instant result: "spam" or "not spam"!

**🤖 Tech Stack**
Python: Core programming
Pandas, NumPy: Data handling
scikit-learn: Naive Bayes, TF-IDF, Evaluation
Flask: Web API
HTML/CSS: Web UI
WordCloud, matplotlib, seaborn: Visualizations

**🗃️ Datasets**
SMS Spam Collection Dataset (5.5k labeled SMS)
Trusted Header List: Real sender IDs (from list_header.xlsx)

**🌍 Real Impact**
415M+ spam SMS sent every day
Hybrid filtering blocks advanced spoofing and phishing
Open-source code—easy to adapt for emails, multiple languages, increased datasets

**✨ Future Scope**
Smarter header matching & sender clustering
Multi-language SMS & email support
Deep learning for harder spam
Real-time feedback system

**🙏 Acknowledgements**
UCI SMS Spam Collection Dataset
All contributors & open-source spirit!

🚫 Say goodbye to SMS Spam! 🚫

