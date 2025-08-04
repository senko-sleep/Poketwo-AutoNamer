import onnxruntime as ort
import numpy as np
import requests
from PIL import Image
import io
import os
import json  # for reading labels from JSON

SUBMODULE_PATH = os.path.dirname(os.path.realpath(__file__))

ONNX_PATH = os.path.join(SUBMODULE_PATH, "model/pokemon_cnn_v2.onnx")
LABELS_PATH = os.path.join(SUBMODULE_PATH, "model/labels_v2.json")
SAVE_PATH = os.path.join(SUBMODULE_PATH, "data/commands/pokemon/images")

class Prediction:
    def __init__(self, onnx_path=ONNX_PATH, labels_path=LABELS_PATH, save_path=SAVE_PATH):
        self.onnx_path = onnx_path
        self.labels_path = labels_path
        self.save_path = save_path
        self.class_names = self.load_class_names()
        self.ort_session = ort.InferenceSession(self.onnx_path)

    def generate_labels_file_from_save_path(self):
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"SAVE_PATH does not exist: {self.save_path}")

        class_names = sorted([
            d for d in os.listdir(self.save_path)
            if os.path.isdir(os.path.join(self.save_path, d))
        ])

        os.makedirs(os.path.dirname(self.labels_path), exist_ok=True)
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2)

        return class_names

    def load_class_names(self):
        if not os.path.exists(self.labels_path):
            return self.generate_labels_file_from_save_path()

        with open(self.labels_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    return [data[k] for k in sorted(data, key=lambda x: int(x))]
                elif isinstance(data, list):
                    return data
                else:
                    raise ValueError("labels_v2.json must be a list or dict")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in {self.labels_path}")

    def preprocess_image_from_url(self, url):
        try:
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")

        image = image.resize((224, 224))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = np.transpose(image, (2, 0, 1))  # CHW
        image = np.expand_dims(image, axis=0).astype(np.float32)  # NCHW
        return image

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, url):
        image = self.preprocess_image_from_url(url)
        inputs = {self.ort_session.get_inputs()[0].name: image}
        outputs = self.ort_session.run(None, inputs)
        logits = outputs[0][0]
        pred_idx = int(np.argmax(logits))
        prob = float(np.max(self.softmax(logits)))
        name = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"unknown_{pred_idx}"
        return name, f"{prob * 100:.2f}%"

def main():
    try:
        predictor = Prediction()
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    while True:
        url = input("Enter Pokémon image URL (or 'q' to quit): ").strip()
        if url.lower() == 'q':
            break
        try:
            name, confidence = predictor.predict(url)
            print(f"Predicted Pokémon: {name} (confidence: {confidence})")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
