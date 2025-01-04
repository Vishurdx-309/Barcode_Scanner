import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import tempfile
import json
from datetime import datetime

# Configure the Gemini API key
api_key = 'YOUR_GEMINI_API_KEY'  # Replace with your actual API key
genai.configure(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def upload_image(image_bytes):
    """
    Save uploaded image bytes and return file for Gemini processing
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name
    
    sample_file = genai.upload_file(temp_file_path)
    return sample_file

def detect_and_decode_barcode(sample_file):
    """
    Use Gemini to detect and decode barcodes in the image
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    
    # Prompt engineering for barcode detection and decoding
    prompt = """Analyze this image for barcodes and provide the following details in JSON format:
    1. Detect all barcodes (1D barcodes, QR codes, etc.)
    2. For each detected barcode provide:
       - Type of barcode (1D, QR, EAN-13, etc.)
       - Decoded content/number
       - Location description in the image
       - Estimated scan quality (clear/blurry)
       
    Return the results in this JSON format:
    {
        "barcodes": [
            {
                "type": "EAN-13",
                "content": "1234567890123",
                "location": "center of image",
                "quality": "clear"
            }
        ]
    }
    
    If no barcodes are found, return an empty barcodes array. Focus only on actual barcodes, not text or numbers that aren't barcodes."""

    response = model.generate_content([sample_file, prompt])
    response_text = response.text.strip()
    
    try:
        # Clean up response if it contains markdown code blocks
        if response_text.startswith("json") and response_text.endswith(""):
            response_text = response_text[7:-3].strip()
            
        result = json.loads(response_text)
        
        # Add timestamps to each barcode detection
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " +5:30"  # IST timezone
        for barcode in result.get("barcodes", []):
            barcode["timestamp"] = timestamp
            
        return result
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return {"barcodes": []}

def verify_barcode_format(sample_file, barcode_content):
    """
    Use Gemini to verify if the detected barcode content follows proper format
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    
    verify_prompt = f"""For the barcode content "{barcode_content}", verify:
    1. If it follows standard barcode format
    2. Check digit validity if applicable
    3. Identify the barcode standard (EAN-13, UPC-A, etc.)
    
    Return result in JSON format:
    {{
        "is_valid": true/false,
        "format": "barcode standard name",
        "validation_details": "explanation of validity"
    }}

    response = model.generate_content([verify_prompt])
    response_text = response.text.strip()
    
    try:
        if response_text.startswith("json") and response_text.endswith(""):
            response_text = response_text[7:-3].strip()
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "is_valid": False,
            "format": "unknown",
            "validation_details": "Failed to verify format"
        }

@app.get('/')
def index():
    return {'message': 'Gemini Barcode Scanner API - Send an image to /scan'}

@app.post('/scan')
async def scan_barcode(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        sample_file = upload_image(contents)
        
        if not sample_file:
            return {"error": "Failed to upload the image"}
        
        # Detect and decode barcodes
        result = detect_and_decode_barcode(sample_file)
        
        # Verify each detected barcode
        for barcode in result.get("barcodes", []):
            validation = verify_barcode_format(sample_file, barcode["content"])
            barcode["validation"] = validation
        
        return {
            "filename": file.filename,
            "message": "Barcode scanning completed",
            "results": result
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post('/scan-batch')
async def scan_multiple(files: list[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            contents = await file.read()
            sample_file = upload_image(contents)
            
            if sample_file:
                # Detect and decode barcodes
                result = detect_and_decode_barcode(sample_file)
                
                # Verify each detected barcode
                for barcode in result.get("barcodes", []):
                    validation = verify_barcode_format(sample_file, barcode["content"])
                    barcode["validation"] = validation
                
                results.append({
                    "filename": file.filename,
                    "results": result
                })
            
        return {
            "message": "Batch processing completed",
            "batch_results": results
        }
        
    except Exception as e:
        return {"error": str(e)}

# Run the API with Uvicorn
if _name_ == '_main_':
    uvicorn.run(app, host='127.0.0.1', port=8000)
