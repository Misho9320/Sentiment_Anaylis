# 🧠 Social Media Sentiment and Trend Analysis
*Python Machine Learning Project — PyCharm IDE*

## 📘 Overview
This project analyzes user sentiments, trending hashtags, and temporal patterns across multiple social media platforms using data science and machine learning techniques. The model processes text data from the **Social Media Sentiments Analysis Dataset** (Kaggle), cleans and transforms it, and classifies emotions using **Logistic Regression**. It also visualizes key insights such as the top 10 sentiments, hashtags, and trends over time.

The project demonstrates how natural language processing (NLP) can be applied to understand public opinion and engagement patterns across digital platforms.

---

## 🧾 Features
✅ Data Cleaning and Preprocessing (text normalization, punctuation removal, stopword filtering)  
✅ Sentiment Classification using Logistic Regression  
✅ TF-IDF Vectorization for text feature extraction  
✅ Visualization of top 10 Sentiments and Hashtags  
✅ Temporal Trend Analysis by Year  
✅ Word Cloud representation of frequent terms  
✅ Accuracy and Classification Report Metrics  

---

## 🧰 Technologies Used
| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python 3.x |
| **IDE** | PyCharm |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Machine Learning** | Scikit-learn |
| **Text Processing** | re (Regular Expressions), NLTK |
| **Dataset Source** | Kaggle — Social Media Sentiments Analysis Dataset |

---

## 📂 Project Structure
```
SocialMediaSentimentAnalysis/
│
├── main.py                      # Main Python script
├── README.md                    # Project documentation
├── sentimentsdataset.csv        # The Kaggle dataset
│   
├── output/
│   ├── sentiment_predictions.csv
│   ├── sentiment_distribution.png
│   ├── top_hashtags.png
│   ├── sentiment_trend.png
│   └── wordcloud.png
└── requirements.txt             # Required Python libraries
```

---

## ⚙️ Installation and Setup
1. **Clone or download** this project folder.  
2. Open the project in **PyCharm**.  
3. Make sure you have Python 3.x installed.  
4. Install all dependencies by running:  
   ```bash
   pip install -r requirements.txt
   ```
5. Place your dataset (`sentimentsdataset.csv`) in the `dataset` folder.  
6. Run the project:  
   ```bash
   python main.py
   ```

---

## 🧹 Data Preprocessing Steps
- Remove URLs, mentions (`@username`), hashtags (`#`), digits, and punctuation.  
- Convert text to lowercase.  
- Remove unnecessary whitespace.  
- Tokenize and prepare text for vectorization.  
- Convert categorical sentiment labels into machine-readable format.  

---

## 🤖 Model Development
1. **Feature Extraction:**  
   Used TF-IDF Vectorizer to convert cleaned text into numerical vectors.  

2. **Model Used:**  
   Logistic Regression (Scikit-learn implementation).  

3. **Training and Testing:**  
   Data split into 80% training and 20% testing sets.  

4. **Evaluation Metrics:**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - Confusion Matrix  

---

## 📊 Visualization
For clarity, only the **top 10 categories** are visualized in the following plots:  
- **Sentiment Distribution**: Frequency of each sentiment.  
- **Top Hashtags**: Most popular hashtags across posts.  
- **Sentiment Trend Over Time**: How sentiment frequency changes yearly.  
- **Word Cloud**: Common words showing the tone of discussion.  

---

## 🧩 Key Findings
- Positive emotions like *joy* and *admiration* dominated the dataset.  
- Negative sentiments such as *anger* and *sadness* were less frequent.  
- Trending hashtags aligned closely with global events and campaigns.  
- Temporal analysis revealed clear shifts in sentiment during significant periods.  
- The Logistic Regression model performed well in classifying text sentiment.  

---

## 👨‍💻 Author
**Name:** Mixo Chauke  
**Institution:** University of Mpumalanga  
**Project:** Data Science in Social Media — Sentiment and Trend Analysis  
**IDE Used:** PyCharm  

---
