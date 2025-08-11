#!/usr/bin/env python3
"""
FastAPI OCR Processing Endpoint

This module provides a REST API for processing images with multiple OCR engines.
Supports binary data upload and multiple model selection.
"""

from abc import ABC, abstractmethod
import asyncio
import os
import base64
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from processors.paddle_ocr_processor import PaddleOcrProcessor
from processors.yomitoku_ocr_processor import YomitokuOcrProcessor
from processors.ocr_processor_interface import OCRProcessorInterface

import cv2
from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_image
    
# Pydantic models for request/response
class OCRRequest(BaseModel):
    """Request model for OCR processing."""
    binary_data: str = Field(..., description="Base64 encoded binary image data")
    filename: str = Field(..., description="Original filename of the image")
    filetype: Optional[str] = Field(None, description="Type of the file (image/pdf)")
    models: List[str] = Field(..., description="List of OCR model names to use")
    output_path: Optional[str] = Field(None, description="Custom output path for results")


class OCRResult(BaseModel):
    """Result model for individual OCR processing."""
    model: str = Field(..., description="OCR model used")
    success: bool = Field(..., description="Whether processing was successful")
    text: str = Field("", description="Extracted text")
    confidence: float = Field(0.0, description="Average confidence score")
    processing_time: str = Field("", description="Time taken to process")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional processing metadata")


class OCRResponse(BaseModel):
    """Response model for OCR processing endpoint."""
    request_id: str = Field(..., description="Unique identifier for this request")
    filename: str = Field(..., description="Original filename")
    timestamp: str = Field(..., description="Processing timestamp")
    total_models: int = Field(..., description="Total number of models requested")
    successful_models: int = Field(..., description="Number of models that processed successfully")
    results: List[OCRResult] = Field(..., description="Results from each OCR model")


class OCRProcessorFactory:
    """Factory class for creating OCR processor instances."""
    
    _processors = {}
    
    @classmethod
    def register_processor(cls, model_name: str, processor_class):
        """Register a new OCR processor."""
        cls._processors[model_name.lower()] = processor_class
    
    @classmethod
    def create_processor(cls, model_name: str) -> OCRProcessorInterface:
        """Create an OCR processor instance for the given model."""
        model_name_lower = model_name.lower()
        
        if model_name_lower not in cls._processors:
            raise ValueError(f"Unsupported OCR model: {model_name}")
        
        processor_class = cls._processors[model_name_lower]
        
        # Configure processor based on model type
        if model_name_lower == 'paddle':
            return processor_class(
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_textline_orientation=True,
                lang='en'
            )
        else:
            # For future processors, add specific configurations here
            return processor_class()
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported OCR models."""
        return list(cls._processors.keys())


# Placeholder processors for models not yet implemented
class EasyOCRProcessor(OCRProcessorInterface):
    """Placeholder for EasyOCR processor."""
    
    async def process_binary_data(self, binary_data: bytes, output_path: str = None, filename: str = None) -> Dict[str, Any]:
        raise NotImplementedError("EasyOCR processor not yet implemented")
    
    def normalize_json_result(self, json_filename: str) -> Dict[str, Any]:
        raise NotImplementedError("EasyOCR processor not yet implemented")


class TesseractProcessor(OCRProcessorInterface):
    """Placeholder for Tesseract processor."""
    
    async def process_binary_data(self, binary_data: bytes, output_path: str = None, filename: str = None) -> Dict[str, Any]:
        raise NotImplementedError("Tesseract processor not yet implemented")
    
    def normalize_json_result(self, json_filename: str) -> Dict[str, Any]:
        raise NotImplementedError("Tesseract processor not yet implemented")

# Register available processors
OCRProcessorFactory.register_processor('paddle', PaddleOcrProcessor)
OCRProcessorFactory.register_processor('easy', EasyOCRProcessor)
OCRProcessorFactory.register_processor('tesseract', TesseractProcessor)
OCRProcessorFactory.register_processor('yomitoku', YomitokuOcrProcessor)


# FastAPI application
app = FastAPI(
    title="OCR Processing API",
    description="REST API for processing images with multiple OCR engines",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "OCR Processing API",
        "version": "1.0.0",
        "supported_models": OCRProcessorFactory.get_supported_models(),
        "endpoints": {
            "process": "/process - POST endpoint for OCR processing",
            "health": "/health - GET endpoint for health check",
            "models": "/models - GET endpoint for supported models"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "supported_models": OCRProcessorFactory.get_supported_models()
    }


@app.get("/models")
async def get_supported_models():
    """Get list of supported OCR models."""
    return {
        "supported_models": OCRProcessorFactory.get_supported_models(),
        "descriptions": {
            "paddle": "PaddleOCR - Multilingual OCR engine with high accuracy",
            "easy": "EasyOCR - Fast and easy-to-use OCR with 80+ language support",
            "tesseract": "Tesseract - Google's open-source OCR engine",
            "yomitoku": "YomiToku - Specialized OCR for Japanese text and documents"
        }
    }

@app.post("/process", response_model=OCRResponse)
async def process_images(request: OCRRequest):
    """
    Process images with selected OCR models.
    
    This endpoint accepts binary image data and processes it with the specified OCR models.
    Returns results from all requested models, including any errors encountered.
    """
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    start_time = datetime.now()
    
    try:
        # Validate request
        if not request.binary_data:
            raise HTTPException(status_code=400, detail="Binary data is required")
        
        if not request.models:
            raise HTTPException(status_code=400, detail="At least one OCR model must be specified")
        
        # Decode binary data
        try:
            binary_data = base64.b64decode(request.binary_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 binary data: {str(e)}")
        
        # Validate models
        supported_models = OCRProcessorFactory.get_supported_models()
        invalid_models = [model for model in request.models if model.lower() not in supported_models]
        if invalid_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported models: {invalid_models}. Supported models: {supported_models}"
            )
        
        # Set output path
        output_path = request.output_path or f"./html_pages/ocr_results/{request_id}"
        os.makedirs(output_path, exist_ok=True)
        
        # Process with each requested model
        results = []
        successful_count = 0
        
        for model_name in request.models:
            model_start_time = datetime.now()
            
            try:
                # Create processor for this model
                processor = OCRProcessorFactory.create_processor(model_name)
                
                # Generate unique filename for this model
                base_filename = os.path.splitext(request.filename)[0]
                model_filename = f"{base_filename}_{model_name.lower()}"
                
                # Process binary data
                processing_result = await processor.process_binary_data(
                    binary_data=binary_data,
                    output_path=output_path,
                    file_name=model_filename,
                    file_type=request.filetype
                )
                print(f"Processing result for {model_name}: {processing_result}")

                # Calculate processing time
                processing_time = str(datetime.now() - model_start_time)
                
                # Create result object
                result = OCRResult(
                    model=model_name,
                    success=True,
                    text=processing_result.get('combined_text', ''),
                    confidence=processing_result.get('overall_confidence', 0.0),
                    processing_time=processing_time,
                    metadata=processing_result.get('metadata', {})
                )
                
                successful_count += 1
                
            except NotImplementedError:
                result = OCRResult(
                    model=model_name,
                    success=False,
                    error_message=f"{model_name} processor is not yet implemented",
                    processing_time=str(datetime.now() - model_start_time)
                )
                
            except Exception as e:
                result = OCRResult(
                    model=model_name,
                    success=False,
                    error_message=str(e),
                    processing_time=str(datetime.now() - model_start_time)
                )
                
                # Log the error for debugging
                print(f"Error processing with {model_name}: {e}")
                print(traceback.format_exc())
            
            results.append(result)
        
        # Create response
        response = OCRResponse(
            request_id=request_id,
            filename=request.filename,
            timestamp=start_time.isoformat(),
            total_models=len(request.models),
            successful_models=successful_count,
            results=results
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in /process endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/process-form")
async def process_images_form(
    file: UploadFile = File(...),
    models: str = Form(..., description="Comma-separated list of OCR models")
):
    """
    Alternative endpoint that accepts file upload via form data.
    
    This is useful for testing with tools like curl or Postman.
    """
    try:
        # Read file data
        file_content = await file.read()
        
        # Parse models from comma-separated string
        model_list = [model.strip() for model in models.split(',')]
        
        # Create request object
        request = OCRRequest(
            binary_data=base64.b64encode(file_content).decode('utf-8'),
            filename=file.filename or "uploaded_file",
            models=model_list
        )
        
        # Process using the main endpoint logic
        return await process_images(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file upload: {str(e)}")


@app.get("/compare-results")
async def compare_results():
    """
    Compare the results of different OCR models.
    """
    # Implement comparison logic here
    res1 = "./html_pages/ocr_results/1754916020965/paddle/3_病理検査報告書_paddle_0_result_json/tmp8z77o5s5_0_res.json"
    res2 = "./html_pages/ocr_results/1754916331901/yomitoku/3_病理検査報告書_yomitoku_0_0.json"

    paddleProcessor = OCRProcessorFactory.create_processor('paddle')
    yomitokuProcessor = OCRProcessorFactory.create_processor('yomitoku')

    # Compare results using the processors
    paddle_result = paddleProcessor.normalize_json_result(res1)
    yomitoku_result = yomitokuProcessor.normalize_json_result(res2)

    return {
        "paddle": paddle_result,
        "yomitoku": yomitoku_result
    }
