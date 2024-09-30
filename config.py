class Config():
    IMAGE_SHAPE = (224, 224, 3) # small size
    EPOCHS = 300
    OFFLINE_SELECT = 5
    BATCH_SIZE = 16
    INITIAL_LEARNING_RATE = 0.0001
    HARD_SELECT = 30
    WARM_UP = False
    CHECKPOINT = 86
    SAVE_PATH = "effnet_checkpoints"