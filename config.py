class Config():
    IMAGE_SHAPE = (224, 224, 3) # small size
    IMAGE_PATH = "card_images_cropped"
    EPOCHS = 300
    OFFLINE_SELECT = 5
    BATCH_SIZE = 8
    INITIAL_LEARNING_RATE = 0.00001
    HARD_SELECT = 30
    WARM_UP = False
    CHECKPOINT = 71