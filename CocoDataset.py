import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import numpy as np
import cv2

class CocoSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None, target_size=(256, 256)):
        """
        Args:
            root (str): Path to the directory containing COCO images.
            annFile (str): Path to the COCO annotations JSON file.
            transforms (callable, optional): Transformations to apply to images and masks.
            target_size (tuple): The target size to resize images and masks (height, width).
        """
        self.coco = CocoDetection(root, annFile)
        self.transforms = transforms
        self.target_size = target_size

    def __getitem__(self, index):
        image, annotations = self.coco[index]

        # Create a blank mask for the entire image
        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        # Combine all annotation masks into a single binary mask
        for annotation in annotations:
            if "segmentation" in annotation:
                rle_or_poly = annotation["segmentation"]
                if isinstance(rle_or_poly, list):
                    # Polygon format
                    for segment in rle_or_poly:
                        poly = np.array(segment).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 1)
                else:
                    # RLE format
                    from pycocotools import mask as mask_utils
                    rle = mask_utils.frPyObjects(rle_or_poly, image.height, image.width)
                    rle_mask = mask_utils.decode(rle)
                    mask = np.maximum(mask, rle_mask)

        # Convert PIL image to tensor and resize
        image = F.to_tensor(image)
        image = F.resize(image, self.target_size)

        # Resize and convert mask to a tensor
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask = F.resize(mask, self.target_size)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask

    def __len__(self):
        return len(self.coco)
