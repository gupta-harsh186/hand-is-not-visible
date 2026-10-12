from app import app
import cv2
import numpy as np

# Create a blank image to test
img = np.zeros((224, 224, 3), dtype=np.uint8)
success, encoded_image = cv2.imencode('.jpg', img)

client = app.test_client()
data = {'file': (encoded_image.tobytes(), 'test.jpg')}

response = client.post('/analyze', data=data, content_type='multipart/form-data')
print("Status:", response.status_code)
print("Data:", response.get_data(as_text=True))
