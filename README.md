# Amazon Product Review Analysis

## Overview

This project is a Streamlit-based web application that performs sentiment analysis on Amazon product reviews. It extracts product details and reviews from an Amazon product page, predicts sentiment using a trained Logistic Regression model, and visualizes the results with bar charts and word clouds.

## Features

- **Web Scraping**: Fetches product title, image, and top reviews from a given Amazon product URL.
- **Sentiment Analysis**: Uses a Logistic Regression model trained on text data to classify reviews as positive or negative.
- **Rating Prediction**: Predicts an approximate rating for the product based on the sentiment analysis results.
- **Data Visualization**:
  - Sentiment distribution bar chart.
  - Word clouds for positive and negative reviews.

## Technologies Used

- **Python**
- **Streamlit** (for web interface)
- **BeautifulSoup** (for web scraping)
- **scikit-learn** (for model training and prediction)
- **matplotlib & WordCloud** (for visualization)

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd amazon-review-analysis
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the trained model files are available in the project directory:

   - `logistic_regression_model.pkl`
   - `tfidf_vectorizer.pkl`

4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter an **Amazon product URL** in the text input box.
2. The application will **scrape product details and reviews**.
3. Sentiment analysis will be performed on the extracted reviews.
4. The predicted **product rating, sentiment distribution, and word clouds** will be displayed.

## Notes

- This project relies on web scraping, which may be blocked by Amazon at times. Ensure you are using a valid product URL.
- The sentiment model should be trained on a relevant dataset for improved accuracy.

## License

This project is open-source and available under the MIT License.

