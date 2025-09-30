# quiz_model_package.py
class QuizModelPackage:
    def __init__(self, model, scaler=None, metadata=None):
        self.model = model
        self.scaler = scaler
        self.metadata = metadata
