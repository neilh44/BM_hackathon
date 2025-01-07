from flask import Flask, request, jsonify, render_template
from pdf_processor import PDFProcessor
from vector_store import VectorStore
import os
from financial_extractor import FinancialExtractor
import os
from datetime import datetime

app = Flask(__name__)
pdf_processor = PDFProcessor()
vector_store = VectorStore()
financial_extractor = FinancialExtractor(vector_store)

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


@app.route('/extract_financials', methods=['POST'])
def extract_financials():
    """
    Extract financial metrics from the document in the specified collection.
    Returns structured financial data or appropriate error messages.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Invalid request format - Content-Type must be application/json'
            }), 400

        data = request.json
        collection_name = data.get('collection_name')
        
        # Validate collection name
        if not collection_name:
            return jsonify({
                'error': 'Missing collection name',
                'details': 'The collection_name parameter is required'
            }), 400
            
        # Attempt to extract metrics
        results = financial_extractor.extract_metrics(collection_name)
        
        # Handle empty results
        if not results:
            return jsonify({
                'status': 'no_results',
                'message': 'No financial metrics could be extracted from the document',
                'metrics': {}
            }), 404
            
        # Format results
        formatted_results = financial_extractor.format_results(results)
        
        # Check if any metrics were successfully formatted
        if not formatted_results:
            return jsonify({
                'status': 'processing_error',
                'message': 'Failed to format extracted metrics',
                'metrics': {}
            }), 422
            
        # Prepare response with metadata
        response = {
            'status': 'success',
            'message': 'Financial metrics extracted successfully',
            'metrics': formatted_results,
            'metadata': {
                'metrics_found': len(formatted_results),
                'collection_name': collection_name,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        # Handle validation errors
        return jsonify({
            'error': 'Validation error',
            'details': str(e)
        }), 400
        
    except ConnectionError as e:
        # Handle vector store connection errors
        return jsonify({
            'error': 'Database connection error',
            'details': 'Failed to connect to vector store',
            'message': str(e)
        }), 503
        
    except Exception as e:
        # Log unexpected errors
        app.logger.error(f'Unexpected error in extract_financials: {str(e)}', exc_info=True)
        
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred while processing the request',
            'reference_id': datetime.now().strftime('%Y%m%d_%H%M%S')
        }), 500
    
if __name__ == '__main__':
    app.run(debug=True)