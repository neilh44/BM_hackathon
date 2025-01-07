# Financial Report  Processing Application

## Overview
This project is a Flask-based web application designed to process Financial PDF files. It provides two main functionalities:
1. Upload a PDF file, convert it to markdown format, and store it in a vector database.
2. Query the processed documents to retrieve relevant information using an AI model.

## Features
- Analyze PDFs to determine if they are text-based or image-based.
- Convert image-based PDFs to markdown using OCR.
- Store processed PDFs in a vector database for efficient querying.
- Query stored documents using natural language to retrieve specific information.

## Prerequisites
Ensure the following are installed on your system:
- Python 3.8+
- Flask
- PyMuPDF (fitz)
- pytesseract
- PIL (Pillow)
- sentence-transformers
- langchain
- Groq API client
- tesseract-ocr (for OCR processing)

## Setup

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=<your_groq_api_key>
   ```

4. **Start the Application**
   ```bash
   python app.py
   ```

5. **Access the Application**
   Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

## File Structure
```
.
├── app.py                # Main Flask application
├── pdf_processor.py      # Handles PDF analysis and conversion
├── vector_store.py       # Manages vector storage and querying
├── templates/
│   └── index.html        # HTML template for the web interface
├── static/               # Static assets (CSS, JS, etc.)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## API Endpoints

### `GET /`
Renders the home page.

### `POST /upload`
Uploads and processes a PDF file.
- **Request:**
  - File upload (`file` parameter must be a `.pdf` file).
- **Response:**
  - `markdown_path`: Path to the converted markdown file.
  - `collection_name`: Name of the collection in the vector database.

### `POST /query`
Queries the processed document.
- **Request:**
  - `query`: The question to ask.
  - `collection_name`: The collection to query.
- **Response:**
  - `answer`: The response to the query.

## Logging
Logs are saved in `pdf_processor_<date>.log` and `vectorstore_<date>.log`.

## Cleanup
To clean up temporary files and collections, you can call the `cleanup` method in `VectorStore`.

## Acknowledgments
- [Flask](https://flask.palletsprojects.com/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [pytesseract](https://github.com/madmaze/pytesseract)
- [LangChain](https://langchain.readthedocs.io/)
- [Groq API](https://groq.io/)

## License
This project is licensed under the MIT License. See `LICENSE` for details.

