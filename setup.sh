#!/bin/bash

echo "🚀 Setting up SentimentFusions Pro - Advanced AI Sentiment Analyzer"

# Create .streamlit directory
mkdir -p ~/.streamlit/

# Install Python dependencies with progress
echo "📦 Installing Python dependencies..."
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

# Download NLTK data
echo "📚 Downloading NLTK language models..."
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True) 
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print('✅ NLTK data downloaded successfully')
"

# Create Streamlit configuration files
echo "⚙️ Configuring Streamlit..."

echo "\
[general]\n\
email = \"admin@sentimentfusions.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = \$PORT\n\
address = 0.0.0.0\n\
maxUploadSize = 200\n\
maxMessageSize = 200\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = \"#667eea\"\n\
backgroundColor = \"#ffffff\"\n\
secondaryBackgroundColor = \"#f0f2f6\"\n\
textColor = \"#262730\"\n\
font = \"sans serif\"\n\
\n\
[client]\n\
caching = true\n\
displayEnabled = true\n\
\n\
[runner]\n\
magicEnabled = true\n\
installTracer = false\n\
fixMatplotlib = true\n\
" > ~/.streamlit/config.toml

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/huggingface_cache

echo "✅ SentimentFusions Pro setup completed successfully!"
echo "🧠 Advanced AI models ready for deployment"
echo "📊 Enhanced visualizations and analytics enabled"
echo "⚡ Performance optimizations applied"