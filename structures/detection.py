import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw


class DetectionResult:
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

            out_image = Image.fromarray(out_image.astype(np.uint8))
            draw = ImageDraw.Draw(out_image)
            if isinstance(name, list):
                for i, n in enumerate(name):
                    pos = (p1[0], p1[1] + i * 16)
                    draw.text(pos, n, font=font, fill=(255, 0, 0))
            else:
                draw.text(p1, name, font=font, fill=(255, 0, 0))

            out_image = np.array(out_image)
        cv2.imwrite(path, out_image[:, :, ::-1])
