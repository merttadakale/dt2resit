# **Climate Sentiment Analysis - README**

## **Project Overview**
This project analyzes climate-related discussions on Twitter using **Machine Learning (Random Forest)** and **Deep Learning (LSTM)** models to classify sentiment into **positive, neutral, and negative** categories. The goal is to extract meaningful insights from social media conversations about sustainability and climate change.

## **Dataset**
- **Name:** Sustainable Product Hashtags Dataset
- **Source:** Contains tweets related to climate change and sustainability.
- **Features:** Tweet text, retweets, likes, author, polarity, subjectivity, etc.
- **Target Variable:** Sentiment classification (positive, neutral, negative).

## **Installation & Dependencies**
Ensure you have the following installed before running the code:
```bash
pip install pandas numpy scikit-learn tensorflow nltk matplotlib seaborn wordcloud
```
Additionally, download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## **Project Structure**
```
📂 Climate_Sentiment_Analysis
 ├── 📂 data                  # Dataset files
 ├── 📂 notebooks             # Jupyter Notebooks
 ├── 📂 models                # Trained models
 ├── 📂 visuals               # Generated plots and figures
 ├── preprocess.py            # Data preprocessing functions
 ├── train_model.py           # Model training script
 ├── evaluate_model.py        # Model evaluation script
 ├── README.md                # This file
```

## **How to Run the Project**
1. **Load the Dataset**
   - Ensure the dataset is placed in the `data/` folder.
   - Load the dataset using Pandas:
   ```python
   import pandas as pd
   df = pd.read_csv('data/sustainable_tweets.csv')
   ```

2. **Data Preprocessing**
   - Run `preprocess.py` or execute the preprocessing functions in Jupyter Notebook.
   - This includes:
     - Removing punctuation and stopwords.
     - Tokenization and lemmatization.
     - Converting text to numerical format (TF-IDF for RF, embeddings for LSTM).

3. **Exploratory Data Analysis (EDA)**
   - Run the **EDA notebook** to visualize:
     - Sentiment distribution.
     - Word clouds.
     - Feature correlations.

4. **Train the Models**
   - Run `train_model.py` to train **Random Forest** and **LSTM** models.
   ```python
   python train_model.py
   ```

5. **Evaluate Model Performance**
   - Execute `evaluate_model.py` to compare model results:
   ```python
   python evaluate_model.py
   ```
   - Outputs include:
     - Accuracy, Precision, Recall, F1-score.
     - Confusion Matrices.
     - Model comparison visualizations.

## **Results**
| **Model** | **Accuracy** |
|-----------|-------------|
| Random Forest | 67.80% |
| LSTM | 64.41% |

## **Key Findings**
- **Random Forest performed better** in overall sentiment classification.
- **LSTM captured more complex text relationships** but had slightly lower accuracy.
- **Class imbalance issues** were observed, requiring future improvements.

## **Future Improvements**
- Increase dataset size for better generalization.
- Implement **BERT or Transformer models** for better sentiment analysis.
- Apply **SMOTE** to handle class imbalance issues.

## **Author & Contact**
- **Author:** Mert ADAKALE
- **Email:** m.adakale@student.fontys.nl




