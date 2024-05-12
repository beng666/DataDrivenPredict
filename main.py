from water_quality_predictor import WaterQualityPredictor

# Example usage
if __name__ == "__main__":
    predictor = WaterQualityPredictor('waterquality.csv')
    predictor.load_data()
    predictor.preprocess_data()
    predictor.define_models()
    predictor.train_models()
    predictor.evaluate_models()
