import asyncio
import os
import json
import io
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Dict, Any, Optional
from PIL import Image
from yomitoku import DocumentAnalyzer
import cv2

from yomitoku.data.functions import load_image, load_pdf


from result_normalizer.yomitoku_ocr_normalizer import YomitokuOcrNormalizer
from processors.ocr_processor_interface import OCRProcessorInterface

class YomitokuOcrProcessor(OCRProcessorInterface):
    """
    Yomitoku implementation of the OCR processor interface.

    This class provides methods to process image binary data using Yomitoku
    and normalize the resulting JSON files into a standardized format.
    """

    def __init__(self, 
                 use_doc_orientation_classify: bool = False,
                 use_doc_unwarping: bool = False, 
                 use_textline_orientation: bool = False,
                 lang: str = 'en'):
        """
        Initialize the YomitokuOCR processor.
        
        Args:
            use_doc_orientation_classify (bool): Whether to use document orientation classification
            use_doc_unwarping (bool): Whether to use document unwarping
            use_textline_orientation (bool): Whether to use text line orientation
            lang (str): Language for OCR recognition (default: 'en')
        """
        self.ocr = DocumentAnalyzer(visualize=True, device="cuda")
        self.normalizer = YomitokuOcrNormalizer

    async def process_binary_data(self, 
                          binary_data: bytes, 
                          output_path: str = None, 
                          file_name: str = None,
                          file_type: str = None
                          ) -> Dict[str, Any]:
        """
        Process binary image data using YomitokuOCR and save results locally.
        
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

        output_path = f"{output_path}/yomitoku"
        
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

            print(f"Processing image saved to temporary file: {temp_path}")

            # Process with Yomitoku
            images = file_type == 'application/pdf' and load_pdf(temp_path) or load_image(temp_path)
            if not images:
                raise RuntimeError("Yomitoku returned no images")

            # Process results and save
            all_results = []
            processing_metadata = {
                "timestamp": datetime.now().isoformat(),
                "model": "Yomitoku",
                "image_size": 0,
                "total_pages": len(images)
            }

            for i, img in enumerate(images):
                # Save individual result files
                result_filename = f"{file_name}_{i}"
                
                print(f"Processing page {i+1}/{len(images)}: {result_filename}")
                # Save as image with annotations
                image_output_path = os.path.join(output_path, f"{result_filename}")

                # Use executor to run the synchronous DocumentAnalyzer in a thread pool
                # This avoids the "asyncio.run() cannot be called from a running event loop" error
                loop = asyncio.get_running_loop()
                result, ocr_vis, layout_vis = await loop.run_in_executor(None, self.ocr, img)
                print(f"Result page {i+1}/{len(images)}: {result_filename}")

                # HTML形式で解析結果をエクスポート
                result.to_html(f"{image_output_path}_{i}.html", img=img)
                result.to_json(f"{image_output_path}_{i}.json", img=img)

                # 可視化画像を保存
                cv2.imwrite(f"{image_output_path}_{i}.jpg", ocr_vis)
                cv2.imwrite(f"{image_output_path}_layout_{i}.jpg", layout_vis)

                # Extract text and confidence data
                text_data, full_text, average_rec_confidence, average_det_confidence = self._extract_text_data(result)

                page_result = {
                    "page_index": i,
                    "image_output_path": image_output_path,
                    "json_output_path": image_output_path + f"_{i}.json",
                    "text_data": text_data,
                    "full_text": full_text,
                    "average_rec_confidence": average_rec_confidence,
                    "average_det_confidence": average_det_confidence
                }

                all_results.append(page_result)

            # Create summary result
            summary_result = {
                "success": True,
                "metadata": processing_metadata,
                "pages": all_results,
                "combined_text": " ",
                "combined_text": " ".join([page["full_text"] for page in all_results]),
                "overall_det_confidence": sum([page["average_det_confidence"] for page in all_results]) / len(all_results) if all_results else 0,
                "overall_rec_confidence": sum([page["average_rec_confidence"] for page in all_results]) / len(all_results) if all_results else 0,
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
            raise RuntimeError(f"Failed to process binary data with YomitokuOCR: {str(e)}")

    def normalize_json_result(self, json_filename: str) -> Dict[str, Any]:
        """
        Read a JSON file containing YomitokuOCR results and normalize the data structure.
        
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
            normalized_result = {
                "source_file": json_filename,
                "normalized_timestamp": datetime.now().isoformat(),
                "total_text_blocks": len(normalized_data),
                "data": normalized_data,
                "full_text": " ".join([item.get("text", "") for item in normalized_data]),
                "average_confidence": self._calculate_confidence_from_normalized(normalized_data)
            }

            return normalized_result

        except Exception as e:
            raise ValueError(f"Failed to normalize JSON data from {json_filename}: {str(e)}")

    def _extract_text_data(self, result) -> list:
        """Extract text data from YomitokuOCR result object."""
        text_data = []
        full_text = ""
        rec_confidences = 0
        det_confidences = 0

        words = result.words
        for word in words:
            text_data.append({
                "text": word.content,
                "det_confidence": word.det_score,
                "rec_confidence": word.rec_score,
                "bounding_box": word.points
            })
            full_text += word.content
            rec_confidences += word.rec_score
            det_confidences += word.det_score

        average_rec_confidence = rec_confidences / len(words) if words else 0
        average_det_confidence = det_confidences / len(words) if words else 0

        return text_data, full_text, average_rec_confidence, average_det_confidence


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
        """Return information about the YomitokuOCR model configuration."""
        return {
            "model_name": "YomitokuOCR",
            "version": "2.x",
            "supported_languages": ["en", "ch", "fr", "german", "korean", "japan"],
            "features": {
                "text_detection": True,
                "text_recognition": True,
                "text_classification": True,
                "document_analysis": True
            }
        } 
