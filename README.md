# 🏥 Medical AI Chatbot

An intelligent medical chatbot powered by OpenAI and LangChain that provides medical information based on a curated knowledge base. This RAG (Retrieval-Augmented Generation) application uses Pinecone for vector storage and HuggingFace embeddings for semantic search.

![Medical AI Assistant](https://img.shields.io/badge/AI-Medical%20Assistant-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![Flask](https://img.shields.io/badge/Flask-3.1.1-lightgrey)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-orange)

## 📸 Screenshots

### Welcome Screen
![Welcome Screen](screenshots/Screenshot%202025-12-01%20232455.png)
*Modern welcome screen with quick action buttons for common medical questions*

### Chat Interface
![Chat Interface](screenshots/Screenshot%202025-12-01%20232608.png)
*Clean and intuitive chat interface with real-time responses*

### Active Conversation
![Active Conversation](screenshots/Screenshot%202025-12-01%20232625.png)
*Real-time medical Q&A with AI assistant showing detailed responses*

### Conversation Features
![Conversation Features](screenshots/Screenshot%202025-12-01%20233014.png)
*Advanced features including save, view history, and conversation management*

## ✨ Features

- 🤖 **AI-Powered Responses**: Uses OpenAI's GPT-3.5-turbo for intelligent medical information
- 📚 **Knowledge Base**: Retrieves relevant information from medical documents stored in Pinecone
- 💬 **Interactive UI**: Modern, responsive chat interface with smooth animations
- 🔍 **Semantic Search**: Uses sentence transformers for accurate document retrieval
- ⚡ **Real-time Responses**: Fast and efficient response generation
- 🎨 **Beautiful Design**: Clean, professional medical-themed interface with gradient backgrounds
- 💾 **Save Conversations**: Save and manage your chat history locally
- 📱 **Fully Responsive**: Works seamlessly on desktop, tablet, and mobile devices
- ⚙️ **Quick Actions**: Pre-defined questions for common medical topics
- 🔔 **Smart Notifications**: Beautiful toast notifications for user feedback
- 📥 **Export Chats**: Download conversations as text files
- 🎯 **User-Friendly**: Intuitive interface with character counter and typing indicators

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Conda (recommended) or virtualenv
- OpenAI API key
- Pinecone API key

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Medical-chatbot
```

2. **Create and activate conda environment**
```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Prepare your knowledge base** (First time only)

Place your medical PDF documents in the `Data/` folder, then run:
```bash
python store_index.py
```

This will:
- Load PDF documents from the `Data/` folder
- Split them into chunks
- Create embeddings
- Store them in Pinecone

6. **Run the application**
```bash
python app.py
```

The app will be available at `http://localhost:8080`

## 📁 Project Structure

```
Medical-chatbot/
├── app.py                  # Main Flask application
├── store_index.py          # Script to index documents into Pinecone
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── Data/                   # Medical PDF documents
│   └── medical.pdf
├── src/
│   ├── helper.py          # Helper functions for document processing
│   ├── prompt.py          # System prompts for the AI
│   └── __init__.py
├── templates/
│   └── chat.html          # Chat interface HTML
├── static/
│   └── style.css          # Styling for the chat interface
├── screenshots/           # Application screenshots (for README)
│   ├── chat-interface.png
│   ├── conversation.png
│   ├── saved-conversations.png
│   ├── conversation-details.png
│   └── mobile-view.png
└── README.md              # This file
```



## 🔧 Configuration

### OpenAI Settings

In `app.py`, you can modify the LLM settings:
```python
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # or "gpt-4" for better responses
    temperature=0.4,         # Lower = more focused, Higher = more creative
    openai_api_key=OPENAI_API_KEY
)
```

### Retrieval Settings

Adjust the number of documents retrieved:
```python
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Number of documents to retrieve
)
```

## 🎯 Usage

1. Open your browser and navigate to `http://localhost:8080`
2. Type your medical question in the input field
3. Press "Send" or hit Enter
4. The AI will retrieve relevant information and provide a response

### Example Questions

- "What are the symptoms of diabetes?"
- "How can I prevent heart disease?"
- "What is the treatment for high blood pressure?"
- "Tell me about common cold remedies"

## 🛠️ Tech Stack

- **Backend**: Flask (Python web framework)
- **AI/ML**: 
  - OpenAI GPT-3.5-turbo (Language Model)
  - LangChain (RAG framework)
  - HuggingFace Sentence Transformers (Embeddings)
- **Vector Database**: Pinecone
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Gunicorn (production server)

## 📦 Dependencies

Key packages:
- `Flask` - Web framework
- `langchain` - RAG framework
- `langchain-openai` - OpenAI integration
- `langchain-pinecone` - Pinecone vector store
- `sentence-transformers` - Text embeddings
- `pypdf` - PDF processing
- `python-dotenv` - Environment variable management

## 🚢 Deployment

### Local Development
```bash
python app.py
```

### Production (using Gunicorn)
```bash
gunicorn app:app --bind 0.0.0.0:8080
```

### Docker
```bash
docker build -t medical-chatbot .
docker run -p 8080:8080 --env-file .env medical-chatbot
```

## ⚠️ Important Notes

- **Medical Disclaimer**: This chatbot is for informational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment.
- **API Costs**: Using OpenAI API incurs costs. Monitor your usage at https://platform.openai.com/usage
- **Data Privacy**: Do not share personal health information with the chatbot
- **Accuracy**: Always verify medical information with qualified healthcare professionals

## 🔐 Security

- Never commit your `.env` file to version control
- Keep your API keys secure
- Use environment variables for sensitive data
- Regularly update dependencies for security patches

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'langchain_community'`
```bash
pip install langchain-community
```

**Issue**: OpenAI API errors
- Check your API key is valid
- Ensure you have credits in your OpenAI account
- Verify the model name is correct

**Issue**: Pinecone connection errors
- Verify your Pinecone API key
- Check if the index "medicalbot" exists
- Ensure your Pinecone plan supports the index

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**⚕️ Remember**: This is an AI assistant for informational purposes only. Always consult with qualified healthcare professionals for medical advice.
