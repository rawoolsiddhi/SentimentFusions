# 📊 SentimentFusions - Product Review Sentiment Analyzer

A comprehensive web application built with Streamlit that analyzes product reviews using advanced AI sentiment analysis. The app provides detailed insights into customer sentiment with interactive visualizations and data export capabilities.

![SentimentFusions Demo](https://via.placeholder.com/800x400/667eea/ffffff?text=SentimentFusions+Demo)

## 🚀 Features

### Core Functionality
- **AI-Powered Sentiment Analysis**: Uses Hugging Face Transformers (cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Multi-Platform Support**: Designed for Amazon, Flipkart, BestBuy reviews (currently uses mock data)
- **Text Processing**: Advanced cleaning and preprocessing of review text
- **Confidence Scoring**: Each sentiment prediction includes a confidence score

### Visualizations
- **Interactive Charts**: Pie charts and bar charts for sentiment distribution
- **Word Clouds**: Visual representation of most common words in positive/negative reviews
- **Confidence Analysis**: Distribution of prediction confidence scores
- **Rating Correlation**: Analysis of rating vs sentiment relationships

### User Interface
- **Modern Design**: Clean, responsive interface with gradient styling
- **Real-time Progress**: Progress bars and status updates during analysis
- **Searchable Table**: Filter and search through analyzed reviews
- **Data Export**: Download results as CSV files
- **Theme Support**: Light/Dark mode toggle

### Technical Features
- **Batch Processing**: Efficient analysis of multiple reviews
- **Error Handling**: Robust error handling and user feedback
- **Caching**: Optimized performance with Streamlit caching
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sentimentfusions.git
   cd sentimentfusions
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

## 🌐 Deployment to Streamlit Cloud

### Step 1: Prepare Your Repository
1. Ensure all files are committed to your GitHub repository
2. Make sure `requirements.txt` is up to date
3. Verify `app.py` is in the root directory

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set main file path to `app.py`
6. Click "Deploy"

### Step 3: Configuration
- The app will automatically install dependencies from `requirements.txt`
- First deployment may take 5-10 minutes
- Subsequent updates deploy automatically on git push

### Environment Variables (if needed)
If you plan to add real API integrations:
```
HUGGINGFACE_API_KEY=your_api_key_here
AMAZON_API_KEY=your_amazon_key_here
```

## 📖 Usage Guide

### Basic Usage
1. **Enter Product Information**
   - Type a product name in the sidebar (e.g., "iPhone 14", "Samsung TV")
   - Adjust the number of reviews to analyze (20-100)

2. **Run Analysis**
   - Click "🔍 Analyze Reviews" button
   - Wait for the progress bar to complete

3. **Explore Results**
   - View summary metrics at the top
   - Explore interactive charts and visualizations
   - Browse the detailed reviews table
   - Download results as CSV

### Advanced Features
- **Filter Reviews**: Use sentiment filter and search functionality
- **Word Clouds**: Analyze common words in positive/negative reviews
- **Confidence Analysis**: Understand model prediction confidence
- **Export Data**: Download complete analysis results

## 🧪 Testing

The project includes comprehensive unit tests using pytest.

### Run Tests
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test_sentiment_analyzer.py

# Run with coverage
pytest --cov=sentiment_analyzer
```

### Test Coverage
- Sentiment analysis functionality
- Text cleaning and preprocessing
- Mock data generation
- Metrics calculation
- Error handling

## 📁 Project Structure

```
sentimentfusions/
├── app.py                      # Main Streamlit application
├── sentiment_analyzer.py       # Core sentiment analysis logic
├── mock_data.py               # Mock data generation
├── test_sentiment_analyzer.py # Unit tests
├── requirements.txt           # Python dependencies
├── Procfile                   # Deployment configuration
├── setup.sh                   # Streamlit setup script
└── README.md                  # Project documentation
```

## 🔧 Technical Details

### AI Model
- **Model**: cardiffnlp/twitter-roberta-base-sentiment-latest
- **Framework**: Hugging Face Transformers
- **Output**: Positive, Negative, Neutral with confidence scores
- **Performance**: Optimized for social media and review text

### Data Processing
- **Text Cleaning**: HTML tag removal, special character handling
- **Preprocessing**: Lowercase conversion, whitespace normalization
- **Batch Processing**: Efficient handling of multiple reviews
- **Error Handling**: Graceful degradation for problematic text

### Visualizations
- **Plotly**: Interactive charts and graphs
- **Matplotlib**: Word cloud generation
- **Streamlit**: Native components for tables and metrics

## 🚧 Future Enhancements

### Planned Features
- **Real Web Scraping**: Integration with actual e-commerce APIs
- **Multi-language Support**: Analysis of reviews in different languages
- **Aspect-based Analysis**: Sentiment analysis for specific product aspects
- **Trend Analysis**: Historical sentiment tracking
- **Comparison Tool**: Compare sentiment across different products

### Technical Improvements
- **Database Integration**: Store analysis results
- **API Endpoints**: RESTful API for programmatic access
- **Advanced Models**: Fine-tuned models for specific domains
- **Real-time Analysis**: Live sentiment monitoring

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Test thoroughly before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For providing excellent pre-trained models
- **Streamlit**: For the amazing web app framework
- **Cardiff NLP**: For the sentiment analysis model
- **Community**: For feedback and contributions

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/sentimentfusions/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

## 📊 Example Screenshots

### Main Dashboard
![Dashboard](https://via.placeholder.com/800x500/667eea/ffffff?text=Main+Dashboard)

### Sentiment Analysis Results
![Results](https://via.placeholder.com/800x500/764ba2/ffffff?text=Analysis+Results)

### Word Cloud Visualization
![Word Cloud](https://via.placeholder.com/800x500/2E8B57/ffffff?text=Word+Cloud)

---

**Built with ❤️ using Streamlit and Hugging Face Transformers**