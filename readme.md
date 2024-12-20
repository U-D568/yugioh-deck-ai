# Yu-Gi-Oh card finder


Using yolov8 and EfficientNet, recognize Yu-Gi-Oh cards from the deck list image.


### Requirements
- numpy                        1.26.4
- opencv-python                4.10.0.84
- pandas                       2.2.2
- tensorflow                   2.15.0
- ultralytics                  8.2.82


### Motivation
Many people share their deck list by capturing game screens or another web page. However, putting all cards in one image, the card size will be reduced and it will be hard to recognize. If you want to know some cards in the shared deck list image, you have to ask the author what he used or manually compare 10000+ Yugioh cards. To solve this problem yolov8 is used to crop card images from the deck list, then embed each card image using EfficientNetB0, and find the card with the closest distance of the input image and the precomputed card vectors.


### Overall process
Deck list image → Yolov8 → card illustrations → Embedding model → calculate distance with card vectors → determine the card id


### Data preparation
All the card images for the dataset is downloaded from (ygoprodeck)[https://db.ygoprodeck.com]
To train embedding model, frist crop all illustrations from card images. The card itself contains lots of information about card types, text, level and atk/def points. However, the model couldn't focus on the illustration in the card, and thoes information distracts the model's training. The trained model without cropping determines the cards just by the color of the card border, rather than recognizing the illustration of the card.


### Data argumentation
Implemented zoom-in and zoom-out augmentation to make low pixel resolution of train image. The image size can be reduced up to -40% and the reduction ratio is randomly selected in every train steps.


### Used Model
EfficientNetB0 has been used for the backbone model and the Dense layers are removed. To train the model, the contrastive loss is used for the loss function. In each step, the model gets 3 inputs, the original image, positive image, and negative image. The original image is an input image with augmentation. The positive image is also same as the input image but different augmentation value. The negative image is a different image from the input image. In the first 5 epochs, the negative image is selected randomly from the dataset. After 5 epochs, choose the most difficult negative-positive image pairs in every step. To select a difficult pair, every epoch the model makes vectors of cards and picks a minimum distance from negative samples. Euclidean distance is used for measuring distance.


### Demo
```
python test.py path/to/image.png
```