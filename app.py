from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Barcode Detection API")

# Initialize the OpenCV barcode detector
bd = cv2.barcode.BarcodeDetector()

# Function to process and detect barcodes from the image
def process_barcode(img: BytesIO):
    # Convert the uploaded image into a format suitable for OpenCV
    image_array = np.frombuffer(img.getvalue(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Detect and decode barcodes
    ret_bc, decoded_info, _, points = bd.detectAndDecode(image)
    
    result = {}
    
    if ret_bc:
        # Draw polygons around detected barcodes
        image = cv2.polylines(image, points.astype(int), True, (0, 255, 0), 3)
        
        # Process each detected barcode
        for idx, (info, point) in enumerate(zip(decoded_info, points)):
            if info:
                # Add detected barcode to results
                result[f"barcode_{idx+1}"] = {
                    "data": info,
                    "coordinates": point.tolist()
                }
                
                # Add text annotation to image
                image = cv2.putText(
                    image,
                    info,
                    point[1].astype(int),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
    
    return result

# API endpoint to handle barcode detection from an uploaded image
@app.post("/decode-barcode")
async def decode_barcode(image: UploadFile = File(...)):
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": "Uploaded file must be an image"},
                status_code=400
            )
        
        # Read image content
        contents = await image.read()
        img = BytesIO(contents)
        
        # Process barcode detection
        barcode_data = process_barcode(img)
        
        if barcode_data:
            return JSONResponse(content={
                "status": "success",
                "barcodes": barcode_data
            })
        else:
            return JSONResponse(
                content={"status": "no_barcode", "message": "No barcodes detected"},
                status_code=404
            )
            
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

# Add CORS middleware if needed
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# For deployment to render.com
# Create a new file named render.yaml in your project root with:
"""
services:
  - type: web
    name: barcode-detection-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.9
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
