# Water Quality Predictor

The Water Quality Predictor is a Python class designed to facilitate the prediction of dissolved oxygen levels in water bodies based on various environmental parameters. This predictor utilizes machine learning models to analyze historical data and make predictions.

## Getting Started

- Python 3.x
- Required Python libraries: 'pandas', 'scikit-learn', 'xgboost'
## Installation

1. Clone the repository.

2. Install the required Python libraries if not already installed:
```bash
pip install pandas scikit-learn xgboost
```
## Usage

```python
from water_quality_predictor import WaterQualityPredictor

predictor = WaterQualityPredictor('waterquality.csv')

predictor.load_data()
predictor.preprocess_data()

predictor.define_models()

predictor.train_models()

predictor.evaluate_models()
```
## Dataset
The dataset used for training and testing the predictor should be provided in CSV format. Ensure that the dataset contains relevant features (independent variables) and the target variable (dissolved oxygen levels).

## Acknowledgments

- Inspired by the need for accurate water quality prediction models.
- Built using scikit-learn, pandas, and xgboost libraries.
