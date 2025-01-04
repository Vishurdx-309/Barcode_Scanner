from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import tempfile
import json

# Initialize FastAPI and Gemini
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key='AIzaSyBXC_o3DTYBLbVBOLoQHCOQtXVA_DCqp-o')

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    try:
        # Read and save the uploaded image
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        # Upload to Gemini
        image_file = genai.upload_file(temp_file_path)
        
        # Create Gemini model and prompt
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = """Analyze this image for barcodes and provide only the following details in valid JSON format:
        {
            "barcodes": [
                {
                    "type": "barcode type (e.g., QR, EAN-13, etc.)",
                    "content": "decoded content",
                    "location": "location in image",
                    "quality": "scan quality"
                }
            ]
        }
        If no barcodes are found, return: {"barcodes": []}"""
        
        # Get response from Gemini
        response = model.generate_content([image_file, prompt])
        response_text = response.text.strip()
        
        # Clean up response if it contains markdown code blocks
        if response_text.startswith("json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith(""):
            response_text = response_text[3:-3].strip()
            
        # Parse JSON response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a default structure
            result = {"barcodes": []}
        
        return {
            "filename": file.filename,
            "results": result
        }
        
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
