import logging


class CustomLogger:
    def __init__(self):
        self.logger = logging.getLogger('lightgbm_custom')
        self.logger.setLevel(logging.ERROR)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        # Suppress warnings by not doing anything
        pass

    def error(self, message):
        self.logger.error(message)
