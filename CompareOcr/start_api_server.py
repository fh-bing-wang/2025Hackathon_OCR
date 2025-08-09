#!/usr/bin/env python3
"""
Startup script for the OCR Processing API server.

This script starts the FastAPI server with proper configuration
and provides helpful information for testing.
"""

import os
import sys
import time
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import PaddleOCR
        print("✅ All required dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False


def check_test_files():
    """Check if test files exist."""
    test_files_dir = Path("TestFiles")
    if test_files_dir.exists():
        image_files = list(test_files_dir.glob("*.png")) + list(test_files_dir.glob("*.jpg"))
        if image_files:
            print(f"✅ Found {len(image_files)} test images in {test_files_dir}")
            return True
    
    print(f"⚠️  No test images found in {test_files_dir}")
    print("   Consider adding some test images for testing the API")
    return False


def print_startup_info(host="0.0.0.0", port=8000):
    """Print startup information and usage examples."""
    print("\n" + "=" * 60)
    print("🚀 OCR Processing API Server")
    print("=" * 60)
    print(f"Server running on: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"ReDoc Documentation: http://{host}:{port}/redoc")
    print(f"Health Check: http://{host}:{port}/health")
    print()
    print("📋 Available Endpoints:")
    print(f"  • GET  /models  - List supported OCR models")
    print(f"  • POST /process - Process images with OCR")
    print(f"  • POST /process-form - Process via form upload")
    print()
    print("🧪 Testing Options:")
    print("  1. Open the web interface: ocr_processor.html")
    print("  2. Use the API documentation: /docs")
    print("  3. Run the test client: python test_api_client.py")
    print("  4. Use curl commands (see examples below)")
    print()
    print("💡 Example curl commands:")
    print("  # Health check")
    print(f"  curl http://{host}:{port}/health")
    print()
    print("  # Get supported models")
    print(f"  curl http://{host}:{port}/models")
    print()
    print("  # Upload file for processing")
    print(f"  curl -X POST http://{host}:{port}/process-form \\")
    print(f"       -F 'file=@TestFiles/00_breast_examine.png' \\")
    print(f"       -F 'models=paddle'")
    print()
    print("🔧 Configuration:")
    print("  • Supported Models: paddle, easy, tesseract, yomitoku")
    print("  • Currently Implemented: paddle (PaddleOCR)")
    print("  • CORS enabled for web frontends")
    print("  • Results saved to: ./html_pages/ocr_results/")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)


def main():
    """Main startup function."""
    print("Starting OCR Processing API Server...")
    
    # Check dependencies
    # if not check_dependencies():
    #     sys.exit(1)
    
    # Check test files
    # check_test_files()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    # debug = os.getenv("DEBUG", "false").lower() == "true"
    debug = True
    
    # Print startup information
    print_startup_info(host, port)
    
    # Create results directory
    os.makedirs("./html_pages/ocr_results", exist_ok=True)
    
    try:
        # Import and run the FastAPI app
        import uvicorn
        print("change Import and run the FastAPI app")
        uvicorn.run(
            "ocr_api:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info" if not debug else "debug",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
