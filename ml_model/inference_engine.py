from ml_model.voice_size_weight_model import VoiceSizeWeightModel
from ml_model.data_preprocessor import DataPreprocessor


class InferenceEngine:
    def __init__(self, model_filename):
        self.model = VoiceSizeWeightModel.load(model_filename)

    def predict_one(self, audio):
        preprocessor = DataPreprocessor(self.model.meta_data['hyperparameters'])
        inputs = preprocessor.preprocess_for_inference(audio)

        predictions = self.model.predict(inputs)

        return predictions[0]
