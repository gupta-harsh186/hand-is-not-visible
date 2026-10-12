import requests
import numpy as np
import cv2

img = np.zeros((224, 224, 3), dtype=np.uint8)
success, encoded_image = cv2.imencode('.jpg', img)

files = {'file': ('test.jpg', encoded_image.tobytes(), 'image/jpeg')}
try:
    response = requests.post('http://127.0.0.1:5000/analyze', files=files)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.text)
except Exception as e:
    print("Error:", e)
