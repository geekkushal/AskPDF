# PDF Chat Assistant with Mistral AI

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Mistral API Key
1. Go to [Mistral AI Console](https://console.mistral.ai/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the API key

### 3. Environment Configuration
1. Create a `.env` file in the project root directory
2. Add your Mistral API key:
```
MISTRAL_API_KEY=your_actual_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

## What Changed from ChatGroq to Mistral

### Code Changes:
- ✅ Replaced `langchain_groq.ChatGroq` with `langchain_mistralai.ChatMistralAI`
- ✅ Updated model from "LLaMA3-8b-8192" to "mistral-small-latest"
- ✅ Added Mistral API key validation
- ✅ Updated requirements.txt with Mistral dependencies
- ✅ Added error handling for API capacity issues

### Benefits of Mistral:
- 🚀 High-performance language model
- 💡 Excellent for RAG applications
- 🔧 Well-documented API
- 📚 Great for document Q&A tasks

### Model Options:
If you encounter capacity issues, you can try these models (edit `app.py` line 47):
1. `mistral-small-latest` (current - balanced performance/availability)
2. `open-mistral-7b` (completely free, good performance)
3. `open-mixtral-8x7b` (more powerful, may have capacity limits)
4. `mistral-large-latest` (most powerful, requires paid plan)

## Features
- 📄 Multi-PDF support
- 🧠 AI-powered question answering
- 💬 Conversation memory
- ⚡ Fast document processing
- 🎯 Accurate context retrieval

## Troubleshooting
- Make sure your `.env` file is in the same directory as `app.py`
- Verify your Mistral API key is valid and has sufficient credits
- Check that all dependencies are installed correctly
