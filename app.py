from flask import Flask, request, jsonify, render_template
from pdf_processor import PDFProcessor
from vector_store import VectorStore
import os

app = Flask(__name__)
pdf_processor = PDFProcessor()
vector_store = VectorStore()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if not file or not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid file'}), 400
        
    try:
        temp_path = 'temp.pdf'
        file.save(temp_path)
        
        # Process the PDF
        markdown_path = pdf_processor.convert_to_markdown(temp_path)
        collection_name = vector_store.load_to_vectorstore(markdown_path)
        
        return jsonify({
            'markdown_path': markdown_path,
            'collection_name': collection_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(markdown_path):
            os.remove(markdown_path)
@app.route('/query', methods=['POST'])
def query_document():
    try:
        data = request.json
        query = data.get('query')
        collection_name = data.get('collection_name')
        
        if not query or not collection_name:
            return jsonify({'error': 'Missing parameters'}), 400
            
        answer = vector_store.query_document(query, collection_name)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)