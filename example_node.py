import os
import sys

import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import torch
import logging
import glob
try:
    from .util import image_processing as impro
    from .models import loadmodel, runmodel
except Exception as e:
    print(e)
    input('Please press any key to exit.\n')
    sys.exit(0)

deep_mosaic_model_dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")

mosaic_model_path = os.path.join(deep_mosaic_model_dir_name, "mosaic_position.pth")
remove_image_mosaic_model_path = os.path.join(deep_mosaic_model_dir_name, "clean_youknow_resnet_9blocks.pth")
remove_video_mosaic_model_path = os.path.join(deep_mosaic_model_dir_name, "clean_youknow_video.pth")

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2cv(image):
    # Convert PIL Image to numpy array and then change the color channels from RGB to BGR
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2pil(image):
    # Change the color channels from BGR to RGB and then convert to PIL Image
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def tensor2cv(image):
    # Convert Tensor to numpy array and then change the color channels from RGB to BGR
    return cv2.cvtColor(image.cpu().numpy().squeeze(), cv2.COLOR_RGB2BGR)

def cv2tensor(image):
    # Change the color channels from BGR to RGB and then convert to Tensor
    return torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)

class POSITIONS:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

class DeepMosaicGetImageMosaicMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
            },
            "optional": {
                "mask_threshold": ("INT", {
                    "default": 48, 
                    "min": 0, #Minimum value
                    "max": 128, #Maximum value
                    "step": 4, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                })
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "POSITIONS")
    RETURN_NAMES = ("origin_image", "mosaic_mask", "mosaic_position")
    FUNCTION = "main"
    CATEGORY = "deepmosic"

    def main(self, image, mask_threshold=48):
        mosaic_model = loadmodel.load_mosaic_bisenet(mosaic_model_path)
        remove_image_mosaic_model = loadmodel.load_pix2pix_model(remove_image_mosaic_model_path, "resnet_9blocks")
        print("DeepMosaicGetImageMosaicMask load model ok")
        cv2_image = tensor2cv(image)
        x, y, size, mask = runmodel.get_mosaic_position(cv2_image, remove_image_mosaic_model, mosaic_model, mask_threshold)
        return image, cv2tensor(mask), POSITIONS(x, y, size)

class DeepMosaicGetVideoMosaicMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
            },
            "optional": {
                "mask_threshold": ("INT", {
                    "default": 48, 
                    "min": 0, #Minimum value
                    "max": 128, #Maximum value
                    "step": 4, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                })
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "POSITIONS")
    RETURN_NAMES = ("origin_images", "mosaic_masks", "mosaic_positions")
    FUNCTION = "main"
    CATEGORY = "deepmosic"

    def main(self, images, mask_threshold=48):
        mosaic_model = loadmodel.load_mosaic_bisenet(mosaic_model_path)
        remove_video_mosaic_model = loadmodel.load_video_model(remove_video_mosaic_model_path)
        print("DeepMosaicGetVideoMosaicMask load model ok")
        masks = []
        positions = []
        for img in images:
            cv2_image = tensor2cv(img)
            x, y, size, mask = runmodel.get_mosaic_position(cv2_image, remove_video_mosaic_model, mosaic_model, mask_threshold)
            masks.append(cv2tensor(mask))
            positions.append(POSITIONS(x, y, size))
        return images, masks, positions

class DeepMosaicRemoveImageMosaic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mask": ("IMAGE", {}),
                "position": ("POSITIONS", {}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "deepmosic"

    def main(self, image, mask, position:POSITIONS):
        remove_image_mosaic_model = loadmodel.load_pix2pix_model(remove_image_mosaic_model_path, "resnet_9blocks")
        print("DeepMosaicRemoveImageMosaic load model ok")
        cv2_image = tensor2cv(image)
        cv2_mask = tensor2cv(mask)
        image_fake = runmodel.run_pix2pixHD(cv2_image, remove_image_mosaic_model)
        image_result = impro.replace_mosaic(cv2_image, image_fake, cv2_mask, position.x, position.y, position.size)
        return cv2tensor(image_result)

class DeepMosaicRemoveVideoMosaic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "masks": ("IMAGE", {}),
                "positions": ("POSITIONS", {}),
            }
        }
    RETURN_TYPES = ("IMAGE",) # 返回值必须是元组
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "deepmosic"

    def main(self, images, masks, positions):
        remove_video_mosaic_model = loadmodel.load_video_model(remove_video_mosaic_model_path)
        print("DeepMosaicRemoveVideoMosaic load model ok")
        image_results = []
        for i, img in enumerate(images):
            cv2_image = tensor2cv(img)
            cv2_mask = tensor2cv(masks[i])
            position = positions[i]
            image_fake = runmodel.run_pix2pixHD(cv2_image, remove_video_mosaic_model)
            image_result = impro.replace_mosaic(cv2_image, image_fake, cv2_mask, position.x, position.y, position.size)
            image_results.append(cv2tensor(image_result))
        return image_results

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DeepMosaicGetImageMosaicMask": DeepMosaicGetImageMosaicMask,
    "DeepMosaicGetVideoMosaicMask": DeepMosaicGetVideoMosaicMask,
    "DeepMosaicRemoveImageMosaic": DeepMosaicRemoveImageMosaic,
    "DeepMosaicRemoveVideoMosaic": DeepMosaicRemoveVideoMosaic
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepMosaicGetImageMosaicMask": "Deep Mosaic Get Image Mosaic Mask",
    "DeepMosaicGetVideoMosaicMask": "Deep Mosaic Get Video Mosaic Mask",
    "DeepMosaicRemoveImageMosaic": "Deep Mosaic Remove Image Mosaic",
    "DeepMosaicRemoveVideoMosaic": "Deep Mosaic Remove Video Mosaic"
}
