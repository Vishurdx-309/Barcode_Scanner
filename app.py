import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
from pyzbar.pyzbar import decode
from ultralytics import YOLO
import numpy as np
import tempfile

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLOv8n model
model = YOLO('barcode.pt')

# Function to rotate the image by a given angle
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Generate a rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated

# Function to decode barcodes from a rotated image
def decode_rotated_barcode(region):
    for angle in range(0, 360, 30):  # Rotate in 30-degree intervals
        rotated_region = rotate_image(region, angle)

        # Convert to grayscale and preprocess
        gray = cv2.cvtColor(rotated_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Decode barcodes
        detectedBarcodes = decode(thresh)

        if detectedBarcodes:
            for barcode in detectedBarcodes:
                if barcode.data:
                    return {
                        "data": barcode.data.decode('utf-8'),
                        "type": barcode.type,
                        "image": rotated_region
                    }
    return None

@app.get('/')
def index():
    return {"message": "Barcode Detection API - Upload an image to /decode"}

@app.post('/decode')
async def decode_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Load the image using OpenCV
        frame = cv2.imread(temp_file_path)

        # Run YOLO inference
        results = model(frame)

        barcode_detected = False
        barcodes_info = []

        # Access bounding boxes for the detected objects
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # Extract the barcode region
                barcode_region = frame[y1:y2, x1:x2]

                # Attempt to decode barcode with rotation
                decoded = decode_rotated_barcode(barcode_region)
                if decoded:
                    barcodes_info.append({
                        "data": decoded["data"],
                        "type": decoded["type"]
                    })
                    barcode_detected = True

        # Response
        if barcode_detected:
            return {
                "message": "Barcodes decoded successfully.",
                "barcodes": barcodes_info
            }
        else:
            return {"message": "No barcodes were successfully decoded."}

    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
     uvicorn.run(app, host='0.0.0.0', port=10000)
