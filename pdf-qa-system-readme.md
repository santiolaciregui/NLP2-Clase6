# PDF Question Answering System

A Streamlit-based application that allows users to upload PDF documents, process them, and ask questions about their content using advanced language models.

## 📋 Features

- **PDF Processing**: Upload and extract content from PDF documents
- **Smart Chunking**: Divide documents into manageable pieces with customizable chunk size and overlap
- **Vector Database Integration**: Store and retrieve document chunks using Pinecone's vector database
- **Natural Language Querying**: Ask questions about your documents in plain English
- **Source Attribution**: View the exact sections of the document that were used to generate answers

## 🛠️ Technologies Used

- **Streamlit**: Web interface framework
- **LangChain**: Framework for LLM application development
- **Groq**: LLM provider (using llama3-70b-8192 model)
- **OpenAI**: Text embedding model (text-embedding-3-large)
- **Pinecone**: Vector database for semantic search
- **PyPDF**: PDF parsing library

## ⚙️ Prerequisites

- Python 3.8+
- Pinecone API key
- Groq API key
- OpenAI API key

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-qa-system.git
   cd pdf-qa-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   ```

## 🚀 Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the application:
   - Upload a PDF document using the file uploader
   - Adjust settings in the sidebar if needed
   - Click "Process Document" to extract and index the content
   - Ask questions about the document content and get AI-generated answers
   - Use sample questions to quickly test functionality

## ⚙️ Configuration Options

The sidebar offers several configuration options:

- **Chunk Size**: The size of text chunks for processing (larger chunks provide more context but may reduce precision)
- **Chunk Overlap**: The amount of overlap between chunks (helps maintain context between chunks)
- **Pinecone Index Name**: Name of the index to create in Pinecone
- **Pinecone Namespace**: Namespace within the index for organizing vectors

## 🧠 How It Works

1. **Document Loading**: The app loads and parses the uploaded PDF document
2. **Text Chunking**: The document is split into smaller, manageable chunks
3. **Embedding Generation**: Each chunk is converted into a vector embedding
4. **Vector Storage**: Embeddings are stored in a Pinecone vector database
5. **Question Processing**: User questions are converted to embeddings and used to search the vector database
6. **Answer Generation**: The LLM generates answers based on the most relevant document chunks

## 🔒 Security Notes

- API keys are stored in environment variables for security
- Temporary files are created and deleted during PDF processing
- The application regenerates the Pinecone index with each processing run

## 📝 Requirements

```
streamlit
python-dotenv
langchain-groq
langchain-openai
langchain-community
langchain-text-splitters
langchain-pinecone
pinecone-client
pypdf
tempfile
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
