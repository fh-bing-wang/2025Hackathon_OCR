from abc import ABC, abstractmethod
from typing import Union, Dict, Any


class OCRProcessorInterface(ABC):
    """
    Abstract interface for OCR processors.
    
    This interface defines the contract that all OCR processor implementations
    must follow, ensuring consistent behavior across different OCR engines.
    """

    @abstractmethod
    async def process_binary_data(self, binary_data: bytes, output_path: str = None, file_name: str = None, file_type: str = None) -> Dict[str, Any]:
        """
        Process binary image data using the OCR engine and save results locally.
        
        Args:
            binary_data (bytes): The binary data of the image file
            output_path (str, optional): Directory to save the results. Defaults to current directory.
            filename (str, optional): Base filename for output files. If None, generates timestamp-based name.
            
        Returns:
            Dict[str, Any]: Processing results containing extracted text, confidence scores, 
                           and metadata about the processing operation.
                           
        Raises:
            ValueError: If binary_data is invalid or empty
            IOError: If unable to save results to specified path
            RuntimeError: If OCR processing fails
        """
        pass

    @abstractmethod
    def normalize_json_result(self, json_filename: str) -> Dict[str, Any]:
        """
        Read a JSON file containing OCR results and normalize the data structure.
        
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
        pass 
