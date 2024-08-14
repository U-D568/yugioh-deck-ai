class Config():
    IMAGE_SHAPE = (224, 224, 3) # small size
    IMAGE_PATH = "dataset/card_images_cropped"
    EPOCHS = 100
    OFFLINE_SELECT = 5
    BATCH_SIZE = 16
    INITIAL_LEARNING_RATE = 0.0001
    HARD_SELECT = 30
    WARM_UP = False
    CHECKPOINT = 71
    SAVE_PATH = "effnet_checkpoints"