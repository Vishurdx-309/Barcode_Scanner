from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
import uvicorn
import imghdr

# Initialize FastAPI app
app = FastAPI(title="Barcode Detection API")

# Initialize the OpenCV barcode detector
bd = cv2.barcode.BarcodeDetector()

def is_valid_image(file_content: bytes) -> bool:
    """Check if the uploaded file is a valid image."""
    image_type = imghdr.what(None, file_content)
    return image_type is not None

# Function to process and detect barcodes from the image
def process_barcode(img: BytesIO):
    # Convert the uploaded image into a format suitable for OpenCV
    image_array = np.frombuffer(img.getvalue(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image")
    
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
        if not image.filename:
            return JSONResponse(
                content={"status": "error", "message": "No file provided"},
                status_code=400
            )

        # Read file content
        contents = await image.read()
        
        # Validate if it's an actual image
        if not is_valid_image(contents):
            return JSONResponse(
                content={"status": "error", "message": "Invalid image file"},
                status_code=400
            )
        
        img = BytesIO(contents)
        
        try:
            # Process barcode detection
            barcode_data = process_barcode(img)
        except ValueError as e:
            return JSONResponse(
                content={"status": "error", "message": str(e)},
                status_code=400
            )
        
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

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
