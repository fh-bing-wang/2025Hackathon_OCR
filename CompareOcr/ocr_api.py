#!/usr/bin/env python3
"""
FastAPI OCR Processing Endpoint

This module provides a REST API for processing images with multiple OCR engines.
Supports binary data upload and multiple model selection.
"""

from abc import ABC, abstractmethod
import asyncio
import difflib
import os
import base64
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from comparator.comparator import Comparator
from processors.tesseract_ocr_processor import TesseractOcrProcessor
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
    rec_confidence: float = Field(0.0, description="Average recognition confidence score")
    det_confidence: float = Field(0.0, description="Average detection confidence score")
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

# Register available processors
OCRProcessorFactory.register_processor('paddle', PaddleOcrProcessor)
OCRProcessorFactory.register_processor('easy', EasyOCRProcessor)
OCRProcessorFactory.register_processor('tesseract', TesseractOcrProcessor)
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

                # Calculate processing time
                processing_time = str(datetime.now() - model_start_time)
                
                # Create result object
                result = OCRResult(
                    model=model_name,
                    success=True,
                    text=processing_result.get('combined_text', ''),
                    rec_confidence=processing_result.get('overall_rec_confidence', 0.0),
                    det_confidence=processing_result.get('overall_det_confidence', 0.0),
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
    file_num = 5
    # Implement comparison logic here
    paddleRes = f"./html_pages/ocr_results/paddle/{file_num:02d}.json"
    yomitokuRes = f"./html_pages/ocr_results/yomitoku/{file_num:02d}.json"
    tesseractRes = f"./html_pages/ocr_results/tesseract/{file_num:02d}.json"

    paddleProcessor = OCRProcessorFactory.create_processor('paddle')
    yomitokuProcessor = OCRProcessorFactory.create_processor('yomitoku')
    tesseractProcessor = OCRProcessorFactory.create_processor('tesseract')

    # Compare results using the processors
    paddle_result = paddleProcessor.normalize_json_result(paddleRes)
    yomitoku_result = yomitokuProcessor.normalize_json_result(yomitokuRes)
    tesseract_result = tesseractProcessor.normalize_json_result(tesseractRes)

    paddle_combined_result = Comparator.combine_texts(paddle_result)
    yomitoku_combined_result = Comparator.combine_texts(yomitoku_result)
    tesseract_combined_result = Comparator.combine_texts(tesseract_result)

    answer_00 = "D DWIBS: search\n検診有無: マンモ 3-4年前・エコー 7-8年前\n自覚症状: 左胸と脇が痛むときがある\n無痛MRI乳がん検診報告書\n拡散強調画像(DWIBS 法) の全体像 (図 1,2)にお\nいて、 がんを示唆する異常な結節は認めません。\n脂肪抑制 T2強調画像 (図3)においても、 特に異常を\n認めません。\n左胸と脇が痛むときがあるとのことですが、 特に炎症\nなどの異常は認めません。 このためこの点については\n安心してよいと思われます。\n判定 A: 異常なし\nがんを示唆する所見は認めません。\n今後も定期的な検診を受けて、健康にお過ごしくださ\nい。\n(図1 前から見た画像)\n(図2下から見た画像)\n(図3下から見た画像)\n"
    answer_01 = "月\n和4年2月1日\n精度管理調査評価結果通知書\n【胸部エックス線検査分野】\n施設コード: 38007\n(医)順風会 健診センター 殿\n公益社団法人 全国労働衛生団体会\n総合精度管理委员会\n胸部エックス線検 委員会\n令和3年度 全衛連ニックス線写真精度管理調査を評価した結果\n貴施設は、【胸部エックス線画像の\n【評価区分】\n審査結果は 評価 A】 に\nなりましたので通知します。\n評価 評価合計点の平均が、 85点以上\n評価 評価\n:\n85点未満。\n点の均が、70点以上\n評価C: 評価合計点の平均が、60点以上 70点未満。\n評価 評価合計点の平均が、60点未満。\n"
    
    answer_02_rarranged_for_yomitoku = "病理検査/細胞診/遺伝子検査《525》NORTHLaB®DVMsどうぶつ医療センター横浜御中〒003-0027札幌市白石区本通2丁目北8-35TEL:(011)827-7407/FAX(011)827-7406診断内容k-okada@northlab.netその他info@northlab.net受付番号BP21-11194病理檢查報告書受付日2021/06/18報告日:2021/06/21受付番号BP21-11194患者名:アオキファラオ(GM)(SP)ちゃんカルテNO:64132動物種:イヌ種類:ブリュッセルグリフォン年齢:11Y性別:♂C送付組織:甲状腺担当医:渡邊·樋口先生病理組織診断左甲状腺:甲状腺濾胞腺癌thyroidfollicularadenocarcinoma肝内側左葉:肝細胞癌hepatocellularcarcinoma左甲状腺では、境界やや不明瞭な腫瘍性病変が形成されています。腫瘍は、大小不整な濾胞構造を形成すある異型な上皮性細胞の腫瘍性増殖から成り立っています。増殖する個々の細胞は、立方状で弱好酸性細胞質と軽度から中程度の大小不同を示す類円形の異型核を有し、核分裂像は2個/10高倍率視野です。腫瘍は一部で被膜外に浸潤しています。腫瘤周囲には1ヶ所、上皮小体が含まれています。肝内側左葉では、境界明瞭な腫瘍性病変が形成されています。腫瘍は、索状の配列を示す異型な肝細胞の腫瘍性増殖からなり、腫瘤内に正常な三つ組構造は認められません。肝細胞索の太さ、走行は不整となっています。腫瘍細胞は軽度から中程度の大小不同と核異型を示し、核分裂像はほとんど認められません。左甲狀腺肝内侧左葉左甲状腺の腫瘤は、甲状腺の濾胞上皮由来の悪性腫瘍と判断されます。明らかな血管内への浸潤やマージン部に腫瘍性の病変は認められませんが、腫瘍は一部で被膜外に浸潤しています。腫瘍の摘出状態は良好と判断されますが、引き続き、経過観察をお勧めします。肝内側左葉の腫瘤は、肝細胞癌と判断されます。腫瘍細胞の分化は高く、高分化な肝細胞癌と判断されます。腫瘍の境界は明瞭ですが、最小限のマージンでの切除となっており、断端マージンに腫瘍細胞が認められます。引き続き、局所の状態について経過観察をお勧めします。診断医岡田一喜、DVMandJCVP*診断書を学会などで使用される場合は、事前にご連絡下さい。*ホルマリン組織は、受付後30日間保存しています。返却が必要な場合は、連絡して下さい。*組織ブロックの返却は行っておりませんので、ご了承お願いいたします。*飼い主様からの直接のお問い合わせはご遠慮いただいております。かかりつけの病院を通じて、お問い合わせください。"
    answer_02=answer_02_rarranged_for_yomitoku
    
    answer_03_rearranged_for_yomitoku="SAMPLE病理検査報告書H0000000SAMPLE病理検査報告書H0000000氏名:79才女患者番号:採取日:氏名:79才女患者番号:採取日:依頼医療機関名:依頼医:検体:食道依頼医療機関名:クリニック依頼医:検体:食道患者ID:検査日時:検査種別:上部内視鏡患者名:検査室:2F内視鏡室2生年月日:79才女施行者:田中孝検査部位:食道、胃、十二指腸ソル・コーテフ(mg):止血处理:検査方法:経口その他薬剤:色素:スポラミン(A):生検:生検検査の様子:Aグルカゴン(V):ポリペク:カメラNo:31ロヒプノール(mg):臟器数:1臓器その他:アネキセート(ml):クリップ:■診断バレット食道の疑い多発性胃ポリープGERDgradeM2■所見内容食道:EC部に色調変化あり下部にSSBEと思われる胃粘膜の変則的なせり上がりあり中心部に白苔の付着を伴う毛羽立ちあり生検13435標本1-1標本1-2胃:C-0の粘膜萎縮胃底部から胃体部にかけてポリープ散在十二指腸:球部から下行脚にかけて嚢胞多発[病理検査結果]High-gradeintraetpithelialneoplasmoftheesophagus,Bx:Seeanadditionalreport![病理所見]1squamousepithelium:分化傾向はありますが、基底側1/2には、全長に亘って、幼若な細胞(basal-parabasalcell相当)の増殖がみられます(標本1-1)。核は小型で類円形で、比較的に揃っていますが、よくみると、若干ですが、大小不同・多形性が認められます(標本1-2,3)。そして表層まで腫大核がみられます。少なくともhigh-gradedysplasia*/high-gradeIENにしたいです。標本1-3標本1p53コメント:*欧米の基準では、CISを含めておりますが、我が国の基準では、CISはhigh-gradeの上に別に設ける形になってます。本例に関しては、私は、上記のように判断しましたが、ほとんどCISに近いものとみたいです。是非EMR/ESDを行っていただきたいです。なお、できればp53,Ki-67免疫染色で評価したいです。<追加報告(免疫染色の結果)>上皮の深部1/2には、p53が基底層から(とくに標本1p53の右端のdown-growthの部分)diffuseに染色されております。一方、Ki-67の染色野は、p53染色野より狭く、ややまばらであります(標本1Ki-67)。これは、これまでの知見から、Carcinomaにconsistentな所見とされてます。検査結果は、既報と変わりません(CISとしても構いません)。標本1Ki-67病理医1:株式会社パソネット病理医1:株式会社パソネット〒420-0834静岡県静岡市葵区音羽町8番18号〒420-0834静岡県静岡市葵区音羽町8番18号報告日:病理医2:報告日:病理医2:電話:054-295-5100検査責任者:森貞晴電話:054-295-5100検査責任者:森貞晴本報告記事を公表される際は予め弊社までご連絡ください。本報告記事を公表される際は予め弊社までご連絡ください。"
    answer_03_rearranged_for_paddle="SAMPLE病理検査報告書H0000000SAMPLE病理検査報告書H0000000氏名:79才女患者番号:氏名:採取日:79才女患者番号:採取日:依頼医療機関名:依頼医:検体:食道依頼医療機関名:クリニック検体:食道依頼医:患者ID:検査日時:検査種別:上部内視鏡患者名:検査室:2F内視鏡室2生年月日:79才女施行者:田中孝検査部位:食道、胃、十二指腸ソル・コーテフ(mg):止血处理:検査方法:経口その他薬剤:色素:スポラミン(A):生検:生検検査の様子:Aグルカゴン(V):ポリペク:カメラNo:31ロヒプノール(mg):臟器数:1臓器その他:アネキセート(ml):クリップ:■診断バレット食道の疑い多発性胃ポリープGERDgradeM2■所見内容食道:EC部に色調変化あり下部にSSBEと思われる胃粘膜の変則的なせり上がりあり中心部に白苔の付着を伴う毛羽立ちあり生検13435標本1-1標本1-2胃:C-0の粘膜萎縮胃底部から胃体部にかけてポリープ散在十二指腸:球部から下行脚にかけて嚢胞多発[病理検査結果]High-gradeintraetpithelialneoplasmoftheesophagus,Bx:Seeanadditionalreport![病理所見]1squamousepithelium:分化傾向はありますが、基底側1/2には、全長に亘って、幼若な細胞(basal-parabasalcell相当)の増殖がみられます(標本1-1)。核は小型で類円形で、比較的に揃っていますが、よくみると、若干ですが、大小不同・多形性が認められます(標本1-2,3)。そして表層まで腫大核がみられます。少なくともhigh-gradedysplasia*/high-gradeIENにしたいです。標本1-3標本1p53コメント:*欧米の基準では、CISを含めておりますが、我が国の基準では、CISはhigh-gradeの上に別に設ける形になってます。本例に関しては、私は、上記のように判断しましたが、ほとんどCISに近いものとみたいです。是非EMR/ESDを行っていただきたいです。なお、できればp53,Ki-67免疫染色で評価したいです。<追加報告(免疫染色の結果)>上皮の深部1/2には、p53が基底層から(とくに標本1p53の右端のdown-growthの部分)diffuseに染色されております。一方、Ki-67の染色野は、p53染色野より狭く、ややまばらであります(標本1Ki-67)。これは、これまでの知見から、Carcinomaにconsistentな所見とされてます。検査結果は、既報と変わりません(CISとしても構いません)。標本1Ki-67病理医1:株式会社パソネット病理医1:株式会社パソネット〒420-0834静岡県静岡市葵区音羽町8番18号〒420-0834静岡県静岡市葵区音羽町8番18号報告日:病理医2:電話:054-295-5100検査責任者:森貞晴報告日:病理医2:電話:054-295-5100検査責任者:森貞晴本報告記事を公表される際は予め弊社までご連絡ください。本報告記事を公表される際は予め弊社までご連絡ください。"
    
    answer_03="SAMPLE病理検査報告書H0000000SAMPLE病理検査報告書H0000000氏名:79才女患者番号:氏名:採取日:79才女患者番号:採取日:依頼医療機関名:依頼医:検体:食道依頼医療機関名:クリニック検体:食道依頼医:患者ID:検査日時:検査種別:上部内視鏡患者名:検査室:2F内視鏡室2生年月日:79才女施行者:田中孝検査部位:食道、胃、十二指腸ソル・コーテフ(mg):止血处理:検査方法:経口その他薬剤:色素:スポラミン(A):生検:生検検査の様子:Aグルカゴン(V):ポリペク:カメラNo:31ロヒプノール(mg):臟器数:1臓器その他:アネキセート(ml):クリップ:■診断バレット食道の疑い多発性胃ポリープGERDgradeM2■所見内容食道:EC部に色調変化あり下部にSSBEと思われる胃粘膜の変則的なせり上がりあり中心部に白苔の付着を伴う毛羽立ちあり生検13435標本1-1標本1-2胃:C-0の粘膜萎縮胃底部から胃体部にかけてポリープ散在十二指腸:球部から下行脚にかけて嚢胞多発[病理検査結果]High-gradeintraetpithelialneoplasmoftheesophagus,Bx:Seeanadditionalreport![病理所見]1squamousepithelium:分化傾向はありますが、基底側1/2には、全長に亘って、幼若な細胞(basal-parabasalcell相当)の増殖がみられます(標本1-1)。核は小型で類円形で、比較的に揃っていますが、よくみると、若干ですが、大小不同・多形性が認められます(標本1-2,3)。そして表層まで腫大核がみられます。少なくともhigh-gradedysplasia*/high-gradeIENにしたいです。標本1-3標本1p53コメント:*欧米の基準では、CISを含めておりますが、我が国の基準では、CISはhigh-gradeの上に別に設ける形になってます。本例に関しては、私は、上記のように判断しましたが、ほとんどCISに近いものとみたいです。是非EMR/ESDを行っていただきたいです。なお、できればp53,Ki-67免疫染色で評価したいです。<追加報告(免疫染色の結果)>上皮の深部1/2には、p53が基底層から(とくに標本1p53の右端のdown-growthの部分)diffuseに染色されております。一方、Ki-67の染色野は、p53染色野より狭く、ややまばらであります(標本1Ki-67)。これは、これまでの知見から、Carcinomaにconsistentな所見とされてます。検査結果は、既報と変わりません(CISとしても構いません)。標本1Ki-67病理医1:株式会社パソネット病理医1:株式会社パソネット〒420-0834静岡県静岡市葵区音羽町8番18号〒420-0834静岡県静岡市葵区音羽町8番18号報告日:病理医2:電話:054-295-5100検査責任者:森貞晴報告日:病理医2:電話:054-295-5100検査責任者:森貞晴本報告記事を公表される際は予め弊社までご連絡ください。本報告記事を公表される際は予め弊社までご連絡ください。"
    # answer_03 = "N\nPATHONET\nSAMPLE\n病理検査報告書\nH0000000\nN\nPATHONET\nSAMPLE\n病理検査報告書\n氏 名:\n79 才女 患者番号:\n採取日 :\n氏\n名:\n依頼医療機関名:\n依頼医 :\n検体: 食道\n依頼医療機関名: クリニック\n79 才女 患者番号:\n依頼医:\n採取日:\nH0000000\n検体: 食道\n患者ID:\n患者名:\n生年月日:\n検査部位:\n検査方法:\n79才 女\n検査日時:\n検査種別:\n検査室:\n施行者:\n上部内視鏡\n2F内視鏡室2\n田中 孝\n食道、胃、十二指腸\n経口\nスポラン(A):\nグルカゴン(V):\nソル・コーテフ (mg):\nその他薬剤:\n生検:\n止血处理:\n色素:\n生検\n検査の様子:A\nポリペク:\nカメラNo :\n31\nロヒプノール (mg):\nアネキセートml):\n臟器数:\nクリップ:\n臓器\nその他:\n■診断\nバレット食道の疑い 多発性胃ポリープ GERD grade M2\n■所見内容\n食道: EC部に色調変化あり\n下部にSSBEと思われる胃粘膜の変則的なせり上がりあり\n中心部に白苔の付着を伴う毛羽立ちあり 生検 3435\n胃:\nC-0の粘膜萎縮\n胃底部から胃体部にかけてポリープ散在\n十二指腸球部から下行脚にかけて嚢胞多発\n【 病理検査結果】\nHigh-grade intraetpithelial neoplasm of the esophagus, Bx: See an additional report!\n【 病理所見】\n① squamous epithelium: 分化傾向はありますが、 基底側1/2には、 全長に亘って、\n幼若な細胞(basal-parabasal cell 相当)の増殖がみられます (標本 ①-1)。\n核は小型で類円形で、比較的に揃っていますが、 よくみると、 若干ですが、 大小不同・多形性が\n認められます(標本①-2,3)。そして表層まで腫大核がみられます。\n少なくとも high-grade dysplasia*/ high-grade IEN にしたいです。\nコメント:*欧米の基準では、CIS を含めておりますが、 我が国の基準では、CIS は high-grade の上に\n別に設ける形になってます。 本例に関しては、 私は、上記のように判断しましたが、ほとんど CIS に\n近いものとみたいです。 是非 EMR/ESD を行っていただきたいです。\nなお、できれば p53, Ki-67 免疫染色で評価したいです。\n\u003c追加報告(免疫染色の結果) \u003e\n上皮の深部1/2には、p53 が基底層から(とくに標本 ①p53の右端の down-growth の部分)\ndiffuse に染色されております。\n一方、Ki-67 の染色野は、 p53 染色野より狭く、 ややまばらであります (標本 ① Ki-67)\nこれは、これまでの知見から、 Carcinoma に consistent な所見とされてます。\n標本①-1\n標本①-2\n標本 ①-3\n標本①p53\n標本 ① Ki-67\n病理医1:\n病理医2:\n株式会社パソネット\n〒420-0834 静岡県静岡市葵区音羽町8番18号\n電話 : 054-295-5100 検査責任者: 森貞晴\n本報告記事を公表される際は予め弊社までご連絡ください。\n検査結果は、既報と変わりません (CISとしても構いません)\n。\n病理医1:\n株式会社パソネット\n報告日:\n病理医2:\n〒420-0834 静岡県静岡市葵区音羽町8番18号\n電話 : 054-295-5100 検査責任者: 森貞晴\n報告日:\n本報告記事を公表される際は予め弊社までご連絡ください。\n"
    
    answer_05="病理組織学的診断:腺管癌(Tubularadenocarcinoma)概要:乳管上皮由来の悪性腫瘍性病変(乳癌)の浸潤性増生が認められました。自壊部を中心として皮下に及ぶ、内部が壊死脱落して不規則な境界を有する腫瘤状の乳癌増生巣が形成されています(標本-1~2)。核小体明瞭な型円形異型核を有する癌細胞は、腺腔様配列傾向がみられる大小の胞巣状配列で密に増生し、周囲組織へ向けて浸潤性に拡大しています。腫瘍境界はやや不規則ですが、切除縁には及ばず取り切れています。検索した範囲内では脈管侵襲は見い出されず、同時に検索した左鼠径部リンパ節(標本-2)に腫瘍性病変は認められません。2022824獣医師高橋秀俊特:セアラ切出5カセット:2個(-1:2片,-2:1片)株式会社アマネヤル检查責任者"
    normalized_answer = Comparator.normalize_text(answer_05)
    paddle_diff = difflib.unified_diff(
        paddle_combined_result, normalized_answer, fromfile="paddle", lineterm="expected"
    )
    with open(f"./pycon_res/paddle_{file_num:02d}_text.txt", "w", encoding="utf-8") as f:
        f.write("Paddle: \n")
        f.write(paddle_combined_result)
        f.write("\n")
        f.write("answer: \n")
        f.write(normalized_answer)

    with open(f"./pycon_res/paddle_{file_num:02d}.txt", "w", encoding="utf-8") as f:
        for line in paddle_diff:
            f.write(line)
            f.write("\n")
        f.write("\ndiff ratio: ")
        f.write(str(difflib.SequenceMatcher(None, paddle_combined_result, normalized_answer).ratio()))

    yomitoku_diff = difflib.unified_diff(
        yomitoku_combined_result, normalized_answer, fromfile="yomitoku", lineterm="expected"
    )

    with open(f"./pycon_res/yomitoku_{file_num:02d}_text.txt", "w", encoding="utf-8") as f:
        f.write("Yomitoku: \n")
        f.write(yomitoku_combined_result)
        f.write("\n")
        f.write("Google: \n")
        f.write(normalized_answer)

    with open(f"./pycon_res/yomitoku_{file_num:02d}.txt", "w", encoding="utf-8") as f:
        for line in yomitoku_diff:
            f.write(line)
            f.write("\n")
        f.write("\ndiff ratio: ")
        f.write(str(difflib.SequenceMatcher(None, yomitoku_combined_result, normalized_answer).ratio()))

    tesseract_diff = difflib.unified_diff(
        tesseract_combined_result, normalized_answer, fromfile="tesseract", lineterm="expected"
    )
    with open(f"./pycon_res/tesseract_{file_num:02d}_text.txt", "w", encoding="utf-8") as f:
        f.write("Tesseract: \n")
        f.write(tesseract_combined_result)
        f.write("\n")
        f.write("answer: \n")
        f.write(normalized_answer)

    with open(f"./pycon_res/tesseract_{file_num:02d}.txt", "w", encoding="utf-8") as f:
        for line in tesseract_diff:
            f.write(line)
            f.write("\n")
        f.write("\ndiff ratio: ")
        f.write(str(difflib.SequenceMatcher(None, tesseract_combined_result, normalized_answer).ratio()))
    # compare_res = Comparator.compare("paddle", paddle_result, "yomitoku", yomitoku_result, iou_threshold=0.1)


    return {
        # "paddle": paddle_result,
        # "yomitoku": yomitoku_result,
        # "tesseract": tesseract_result
    }
