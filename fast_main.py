from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from for_core import Pix2TexModel
import cv2
import numpy as np

app = FastAPI()

model = Pix2TexModel()  # 모델 초기화

@app.post("/predict")
async def predict(file: UploadFile = File(...), is_full_image: bool = False):
    try:
        # 파일 스트림에서 numpy 배열로 이미지를 읽습니다.
        nparr = np.frombuffer(await file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        prediction = model.predict(img, is_full_image)
        return JSONResponse(content={"prediction": prediction}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
