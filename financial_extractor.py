import json
import logging
from datetime import datetime
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

def setup_logger(name: str) -> logging.Logger:
    """Setup a logger with file and console handlers"""
    logger = logging.getLogger(name)
    
    # Only add handlers if the logger doesn't have any
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # File handler
        log_file = os.path.join('logs', f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Higher threshold for console
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger

@dataclass
class ExtractedValue:
    value: float
    coordinates: tuple[float, float, float, float]  # x0, y0, x1, y1
    snippet: str
    confidence: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedValue':
        # Handle potential string coordinates
        if isinstance(data.get('coordinates'), str):
            try:
                coords = eval(data['coordinates'])
                if not isinstance(coords, tuple) or len(coords) != 4:
                    coords = (0.0, 0.0, 0.0, 0.0)
            except:
                coords = (0.0, 0.0, 0.0, 0.0)
        else:
            coords = tuple(data.get('coordinates', (0.0, 0.0, 0.0, 0.0)))

        return cls(
            value=float(data['value']),
            coordinates=coords,
            snippet=str(data['snippet']),
            confidence=float(data.get('confidence', 0.0))
        )

@dataclass
class FinancialMetric:
    name: str
    values: List[ExtractedValue]
    final_value: Optional[float]
    reasoning: str
    confidence: float

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> 'FinancialMetric':
        return cls(
            name=name,
            values=[ExtractedValue.from_dict(v) for v in data.get('values', [])],
            final_value=float(data['final_value']) if data.get('final_value') is not None else None,
            reasoning=str(data.get('reasoning', '')),
            confidence=float(data.get('confidence', 0.0))
        )

class FinancialExtractor:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.metrics = [
            "EBIT", "EBITDA", "Net income", "Revenue", "currency", 
            "units", "depreciation", "amortization", "filing publish date",
            "filing type", "fiscal year end"
        ]
        
        # Initialize logger
        self.logger = setup_logger('financial_extractor')
        self.logger.info("FinancialExtractor initialized")
        
    def extract_metrics(self, collection_name: str) -> Dict[str, FinancialMetric]:
        """Extract financial metrics from document"""
        results = {}
        
        base_prompt = """You are a financial data extraction system. Your task is to analyze financial text and output ONLY valid JSON data according to the specified format, with no additional text or explanations.

IMPORTANT: Your entire response must be a single, valid JSON object. Do not include any other text, notes, or explanations.

Analyze this financial report text and extract information about {metric}.

Required JSON structure:
{{
    "values": [
        {{
            "value": <number>,
            "snippet": "<text>",
            "coordinates": [0, 0, 0, 0],
            "confidence": <number between 0 and 1>
        }}
    ],
    "final_value": <number or null>,
    "reasoning": "<text>",
    "confidence": <number between 0 and 1>
}}

Rules:
1. Response must be ONLY the JSON object, nothing else
2. All number values must be valid numbers (not strings)
3. For dates, use string format "YYYY-MM-DD"
4. Coordinates should always be [0, 0, 0, 0] if exact position unknown
5. Confidence scores must be between 0 and 1

Context from document:
{context}"""
        
        for metric in self.metrics:
            try:
                self.logger.info(f"Extracting metric: {metric}")
                
                # Get relevant context
                context_query = f"Find sections discussing or containing {metric}"
                context = self.vector_store.get_context(context_query, collection_name)
                
                if not context.strip():
                    self.logger.warning(f"No context found for metric: {metric}")
                    continue
                
                # Create prompt and get response
                prompt = base_prompt.format(metric=metric, context=context)
                response = self.vector_store.query_document_raw(prompt, collection_name)
                
                try:
                    # Clean response - remove any non-JSON text
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        cleaned_response = response[json_start:json_end]
                    else:
                        self.logger.error(f"No JSON object found in response for {metric}")
                        continue
                    
                    # Parse JSON
                    metric_data = json.loads(cleaned_response)
                    
                    # Validate required fields
                    required_fields = {'values', 'final_value', 'reasoning', 'confidence'}
                    if not all(field in metric_data for field in required_fields):
                        missing = required_fields - set(metric_data.keys())
                        self.logger.error(f"Missing required fields for {metric}: {missing}")
                        continue
                    
                    # Create FinancialMetric object
                    results[metric] = FinancialMetric.from_dict(metric, metric_data)
                    self.logger.info(f"Successfully extracted {metric}")
                    
                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"JSON parsing error for {metric}: {str(e)}\n" +
                        f"Response: {response}\n" +
                        f"Cleaned response: {cleaned_response if 'cleaned_response' in locals() else 'N/A'}"
                    )
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing {metric}: {str(e)}")
                    continue
                    
            except Exception as e:
                self.logger.error(f"Extraction error for {metric}: {str(e)}")
                continue
                
        return results

    def format_results(self, results: Dict[str, FinancialMetric]) -> dict:
        """Format results for API response"""
        formatted = {}
        
        for metric, data in results.items():
            try:
                formatted[metric] = {
                    "values": [
                        {
                            "value": str(v.value),
                            "coordinates": v.coordinates,
                            "snippet": v.snippet,
                            "confidence": v.confidence
                        } for v in data.values
                    ],
                    "final_value": str(data.final_value) if data.final_value is not None else None,
                    "reasoning": data.reasoning,
                    "confidence": data.confidence
                }
            except Exception as e:
                self.logger.error(f"Error formatting results for {metric}: {str(e)}")
                continue
                
        return formatted