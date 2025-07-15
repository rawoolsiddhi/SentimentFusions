#!/bin/bash

# Create .streamlit directory
mkdir -p ~/.streamlit/

# Install Python dependencies
echo "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Create Streamlit configuration files
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
\n\
[theme]\n\
primaryColor = \"#667eea\"\n\
backgroundColor = \"#ffffff\"\n\
secondaryBackgroundColor = \"#f0f2f6\"\n\
textColor = \"#262730\"\n\
font = \"sans serif\"\n\
" > ~/.streamlit/config.toml

echo "Setup completed successfully!"