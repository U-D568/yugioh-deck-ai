import logging
import datetime

class TrainLogger:
    def __init__(self):
        filename = datetime.datetime.now().strftime("train_%Y-%m-%d_%H%M%S.log")
        # configuration
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
        )
        self._logger = logging.getLogger()
    
    def info(self, msg):
        self._logger.info(msg)
    
    def error(self, err_msg):
        self._logger.error(err_msg)