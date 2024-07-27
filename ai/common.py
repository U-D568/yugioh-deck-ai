import os

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

def extract_images(path):
    images = []
    for curdir, subdir, files in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if not ext in IMAGE_EXTENSIONS:
                continue
            path = os.path.join(curdir, file)
            images.append(path)
    
    return images

