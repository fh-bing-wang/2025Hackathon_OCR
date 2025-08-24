import os
import json
from datetime import datetime
from pathlib import Path
from pydoc import text
import tempfile
from typing import Dict, Any
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

from result_normalizer.tesseract_ocr_normanizer import TesseractOcrNormalizer
from processors.ocr_processor_interface import OCRProcessorInterface

# See doc https://github.com/madmaze/pytesseract
class TesseractOcrProcessor(OCRProcessorInterface):
    """
    Tesseract implementation of the OCR processor interface.

    This class provides methods to process image binary data using Tesseract
    and normalize the resulting JSON files into a standardized format.
    """

    def __init__(self, 
                 use_doc_orientation_classify: bool = False,
                 use_doc_unwarping: bool = False, 
                 use_textline_orientation: bool = False,
                 lang: str = 'en'):
        """
        Initialize the Tesseract processor.
        
        Args:
            use_doc_orientation_classify (bool): Whether to use document orientation classification
            use_doc_unwarping (bool): Whether to use document unwarping
            use_textline_orientation (bool): Whether to use text line orientation
            lang (str): Language for OCR recognition (default: 'en')
        """
        self.ocr = pytesseract
        self.normalizer = TesseractOcrNormalizer

    async def process_binary_data(self, 
                          binary_data: bytes, 
                          output_path: str = None, 
                          file_name: str = None,
                          file_type: str = None
                          ) -> Dict[str, Any]:
        """
        Process binary image data using TesseractOCR and save results locally.
        
        Args:
            binary_data (bytes): The binary data of the image file
            output_path (str, optional): Directory to save the results. Defaults to './results'
            filename (str, optional): Base filename for output files. If None, generates timestamp-based name.
            
        Returns:
            Dict[str, Any]: Processing results containing extracted text, confidence scores, 
                           and metadata about the processing operation.
                           
        Raises:
            ValueError: If binary_data is invalid or empty
            IOError: If unable to save results to specified path
            RuntimeError: If OCR processing fails
        """
        if not binary_data:
            raise ValueError("Binary data cannot be empty")

        # Set default output path
        if output_path is None:
            output_path = "../html_pages/results/"

        output_path = f"{output_path}/tesseract"

        # Create output directory if it doesn't exist
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Generate filename if not provided
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = timestamp

        try:
            # Convert binary data to image
            suffix = file_type == 'application/pdf' and '.pdf' or '.png'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(binary_data)
                temp_path = temp_file.name

            combined_text = ""
            all_results = []
            if file_type == 'application/pdf':
                pages = convert_from_path(temp_path, dpi=600, fmt='png')
                
                for i, page in enumerate(pages):
                    text, json_data = self._process_image_data(page, output_path, f"{file_name}_{i}")
                    combined_text += text

                    # Extract text and confidence data
                    page_result = {
                        "page_index": i,
                        "image_output_path":f"{output_path}_{i}",
                        "json_output_path": f"{output_path}_{i}",
                        "text_data": self._extract_text_data(json_data),
                        "full_text": text,
                        "average_confidence": self._calculate_average_confidence(json_data)
                    }

                    all_results.append(page_result)
            else:
                combined_text, json_data = self._process_image_data(Image.open(temp_path), output_path, file_name)
                page_result = {
                    "page_index": 0,
                    "image_output_path": output_path,
                    "json_output_path": output_path,
                    "text_data": self._extract_text_data(json_data),
                    "full_text": combined_text,
                    "average_confidence": self._calculate_average_confidence(json_data)
                }

                all_results.append(page_result)

            # Process results and save
            processing_metadata = {
                "timestamp": datetime.now().isoformat(),
                "model": "Tesseract",
                "image_size": 0,
                "total_pages": len(all_results)
            }

            # Create summary result
            summary_result = {
                "success": True,
                "metadata": processing_metadata,
                "pages": all_results,
                "combined_text": combined_text,
                "overall_rec_confidence": sum([page["average_confidence"] for page in all_results]) / len(all_results) if all_results else 0,
                "output_files": {
                    "base_path": output_path,
                    "filename_prefix": file_name
                }
            }

            # Save summary JSON
            summary_path = os.path.join(output_path, f"{file_name}_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_result, f, indent=2, ensure_ascii=False)

            summary_result["summary_file"] = summary_path
            return summary_result

        except Exception as e:
            raise RuntimeError(f"Failed to process binary data with TesseractOCR: {str(e)}")

    def _process_image_data(self, image_data: Image, output_path: str, file_name: str) -> None:
        # Process the image data with Tesseract OCR
        try:
            data = self.ocr.image_to_data(image_data, output_type=pytesseract.Output.DICT, lang='jpn+eng')
            print('tesseract ocr data: ')
            print(data)
            json_data = self._text_data_to_json(data, os.path.join(output_path, f"{file_name}.json"))
            
            hocr = pytesseract.image_to_pdf_or_hocr(image_data, extension="hocr", lang='jpn+eng')
            with open(os.path.join(output_path, f"{file_name}.html"), "w+b") as f:
                f.write(hocr)

            pdf = pytesseract.image_to_pdf_or_hocr(image_data, extension='pdf', lang='jpn+eng')
            with open(os.path.join(output_path, f"{file_name}.pdf"), 'w+b') as f:
                f.write(pdf)

            return pytesseract.image_to_string(image_data, lang="jpn"), json_data
        except Exception as e:
            raise RuntimeError(f"Failed to process image data with TesseractOCR: {str(e)}")

    def _text_data_to_json(self, data: list, path: str) -> list:
        # Process the data and structure it into a list of dictionaries for JSON
        text_data = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            if data['text'][i].strip():  # Only include if text is not empty
                box_info = {
                    'text': data['text'][i],
                    'confidence': float(data['conf'][i]),
                    'bounding_box': {
                        'left': int(data['left'][i]),
                        'top': int(data['top'][i]),
                        'width': int(data['width'][i]),
                        'height': int(data['height'][i])
                    },
                    'level': int(data['level'][i]),
                    'page_num': int(data['page_num'][i]),
                    'block_num': int(data['block_num'][i]),
                    'par_num': int(data['par_num'][i]),
                    'line_num': int(data['line_num'][i]),
                    'word_num': int(data['word_num'][i])
                }
                text_data.append(box_info)
        """Convert text data to JSON format."""
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(text_data, f, ensure_ascii=False, indent=2)
        
        return text_data

    def normalize_json_result(self, json_filename: str) -> Dict[str, Any]:
        """
        Read a JSON file containing TesseractOCR results and normalize the data structure.

        Args:
            json_filename (str): Path to the JSON file containing OCR results
            
        Returns:
            Dict[str, Any]: Normalized JSON object with standardized structure
                           containing text, confidence, coordinates, and metadata.
                           
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON file is malformed or has invalid structure
            IOError: If unable to read the file
        """
        if not os.path.exists(json_filename):
            raise FileNotFoundError(f"JSON file not found: {json_filename}")

        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                raw_result = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {json_filename}: {str(e)}")
        except IOError as e:
            raise IOError(f"Unable to read file {json_filename}: {str(e)}")

        try:
            # Use the existing normalizer
            normalized_data = self.normalizer.normalize(raw_result)
            
            # Add additional metadata
            # normalized_result = {
            #     "source_file": json_filename,
            #     "normalized_timestamp": datetime.now().isoformat(),
            #     "total_text_blocks": len(normalized_data),
            #     "data": normalized_data,
            #     "full_text": " ".join([item.get("text", "") for item in normalized_data]),
            #     "average_confidence": self._calculate_confidence_from_normalized(normalized_data)
            # }

            return normalized_data

        except Exception as e:
            raise ValueError(f"Failed to normalize JSON data from {json_filename}: {str(e)}")

    def _extract_text_data(self, result) -> list:
        """Extract text data from TesseractOCR result object."""
        text_data = []
        for text in result:
            text_data.append({
                "text": text,
                "confidence": text.get('confidence', 0),
                "bounding_box": text.get('bounding_box', None)
            })
        return text_data

    def _extract_full_text(self, result) -> str:
        """Extract all text as a single string from TesseractOCR result."""
        if hasattr(result, 'rec_texts'):
            return " ".join(result.rec_texts)
        return ""

    def _calculate_average_confidence(self, result) -> float:
        """Calculate average confidence score from TesseractOCR result."""
        sum = 0.0
        for res in result:
            sum += res.get('confidence', 0.0)
        return sum / len(result) / 100.00 if result else 0.0

    def _calculate_confidence_from_normalized(self, normalized_data: list) -> float:
        """Calculate average confidence from normalized data."""
        if not normalized_data:
            return 0.0
        
        confidences = [item.get("confidence", 0.0) for item in normalized_data]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def get_supported_formats(self) -> list:
        """Return list of supported image formats."""
        return ['PNG', 'JPG', 'JPEG', 'BMP', 'TIFF', 'GIF']

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the TesseractOCR model configuration."""
        return {
            "model_name": "TesseractOCR",
            "version": "2.x",
            "supported_languages": ["en", "ch", "fr", "german", "korean", "japan"],
            "features": {
                "text_detection": True,
                "text_recognition": True,
                "text_classification": True,
                "document_analysis": True
            }
        } 
