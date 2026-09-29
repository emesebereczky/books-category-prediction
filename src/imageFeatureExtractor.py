from io import BytesIO
import cv2
import numpy as np
import requests
from PIL import Image


class ImageFeatureExtractor:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
                    "./haarcascade_frontalface_default.xml"
                )
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.cache = {}

    def _calculate_colorfulness(self, image_rgb: np.ndarray) -> float:

        R = image_rgb[:, :, 0].astype("float")
        G = image_rgb[:, :, 1].astype("float")
        B = image_rgb[:, :, 2].astype("float")

        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)

        std_root = np.sqrt(
            np.std(rg) ** 2 + np.std(yb) ** 2
        )

        mean_root = np.sqrt(
            np.mean(rg) ** 2 + np.mean(yb) ** 2
        )

        return float(std_root + 0.3 * mean_root)

    def extract(self, image_url: str) -> dict:

        if image_url in self.cache:
            return self.cache[image_url]

        default_res = {
            "brightness": np.nan,
            "colorfulness": np.nan,
            "face_count": np.nan,
            "person_count": np.nan,
        }

        try:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()

            pil_image = Image.open(
                BytesIO(resp.content)
            ).convert("RGB")

            image_rgb = np.array(pil_image)

            image_bgr = cv2.cvtColor(
                image_rgb, cv2.COLOR_RGB2BGR
            )

            gray = cv2.cvtColor(
                image_bgr, cv2.COLOR_BGR2GRAY
            )

            brightness = float(np.mean(gray))
            colorfulness = self._calculate_colorfulness(image_rgb)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(20, 20)
            )

            boxes, _ = self.hog.detectMultiScale(
                image_bgr,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05
            )

            result = {
                "brightness": round(brightness, 2),
                "colorfulness": round(colorfulness, 2),
                "face_count": len(faces),
                "person_count": len(boxes),
            }

        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
            result = default_res

        self.cache[image_url] = result

        return result