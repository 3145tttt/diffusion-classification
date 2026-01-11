# handler.py
import torch
import base64
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io
import os

class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        
        model_dir = properties.get("model_dir")
        self.device = 'cpu'
        serialized_file = self.manifest['model']['serializedFile']
        model_path = os.path.join(model_dir, serialized_file)
        
        if not os.path.isfile(model_path):
            raise RuntimeError("Модель не найдена")
        
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        self.initialized = True
    
    def preprocess(self, data):
        images = []
        
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                image = base64.b64decode(image)
            
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            
            image = self._transform_image(image)
            images.append(image)
        
        return torch.stack(images).to(self.device)
    
    def _transform_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image)
    
    @torch.inference_mode()
    def inference(self, data):
        return self.model(data)
    
    def postprocess(self, data):
        probabilities = torch.nn.functional.softmax(data, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        results = []
        for i in range(data.shape[0]):
            result = {
                "predictions": [
                    {
                        "class_id": int(top_indices[i][j]),
                        "probability": float(top_probs[i][j])
                    }
                    for j in range(top_probs.shape[1])
                ]
            }
            results.append(result)
        
        return results