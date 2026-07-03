from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        device_str = str(device)
        if "cuda" in device_str:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model.to(device=device, dtype=dtype)
        else:
            self.model.to(device=device, dtype=torch.float32)

    def cuda(self):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model.to(device="cuda", dtype=dtype)

    def cpu(self):
        self.model.to(device="cpu", dtype=torch.float32)
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        input_images = self.transform_image(image).unsqueeze(0).to(device=device, dtype=dtype)
        # Prediction
        try:
            if device.type == "cuda":
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
                    preds = self.model(input_images)[-1].sigmoid().cpu()
            else:
                with torch.no_grad():
                    preds = self.model(input_images)[-1].sigmoid().cpu()
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
                print("[BiRefNet] CUDA OOM encountered during preprocessing. Falling back to CPU...")
                self.cpu()
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                input_images = self.transform_image(image).unsqueeze(0).to(device="cpu", dtype=torch.float32)
                with torch.no_grad():
                    preds = self.model(input_images)[-1].sigmoid().cpu()
            else:
                raise e
        pred = preds[0].squeeze().float()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    