import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw

from . import common
from . import db

class ObjectDetectionBatch:
    def __init__(self):
        self.X: np.array
        self.ids: np.array
        self.xyxy: np.array
        self.xywh: np.array
        self.indexes: np.array

    def get_batch_size(self):
        return self.X.shape[0]


class DeckImageData:
    def __init__(self, X, ids, xyxy):
        self.X = X
        self.ids = ids
        self.xyxy = xyxy
        self.xywh = common.xyxy2xywh(self.xyxy)

class DetResult:
    def __init__(self, image, bboxes, embeds):
        self.image = image
        self.bboxes = bboxes
        self.embeds = embeds
        self.names = [""] * len(self.embeds)
        self.ids = [""] * len(self.embeds)

    def save(self, path):
        out_image = self.image.copy()
        font = ImageFont.truetype("fonts/MALGUN.TTF")
        for bbox, name in zip(self.bboxes, self.names):
            xyxy = bbox[:4]
            p1 = (int(xyxy[0]), int(xyxy[1]))
            p2 = (int(xyxy[2]), int(xyxy[3]))
            out_image = cv2.rectangle(out_image, p1, p2, (255, 0, 0), 1)

            out_image = Image.fromarray(out_image)
            draw = ImageDraw.Draw(out_image)
            draw.text(p1, name, font=font, fill=(255, 0, 0))

            out_image = np.array(out_image)
        cv2.imwrite(path, out_image[:, :, ::-1])
            
