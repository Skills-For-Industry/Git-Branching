
# Matrix HomeRepair Inc. Churn Prediction
Matrix HomeRepair Inc. is facing a growing churn rate among its subscribers. To tackle this challenge, we've developed a machine learning solution to predict potential churn, enabling proactive retention measures.

## Overview

This project:
- Trains a Logistic Regression model to predict subscriber churn.
- Provides an API endpoint to make churn predictions for new data.

## Features
.  
- **Machine Learning Model**: A Logistic Regression model trained on the dataset, achieving 87% accuracy in predicting churn.

- **API Deployment**: A FastAPI application offers an endpoint for predicting churn based on the input subscriber features.

## Getting Started

### Requirements

- Python 3.x
- Libraries: pandas, sklearn, Faker, FastAPI, joblib, uvicorn

### Running the API Locally

1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Train model using:
`cd scripts`
`python train_model.py`
3. Run the FastAPI app using 
`cd ../api`
`uvicorn app:app --reload`.
4. Access the API documentation at `http://127.0.0.1:8000/docs`.

## Usage

Send a POST request to `http://127.0.0.1:8000/predict/` with subscriber features to receive a churn prediction.

Example Request Body:

```
{
    "age": 28,
    "subscription_duration": 12,
    "last_purchase": 15,
    "average_monthly_usage": 50,
    "customer_support_calls": 2
}
```

## Future Work

- **Feedback Score Integration**: We're planning to introduce a "feedback_score" based on customer feedback to improve the model's prediction capability.
- **Model Improvements**: Experiment with more complex models like Random Forest and Gradient Boosting for potentially better performance.

## Contributing

Feel free to fork this repository, make changes, and submit pull requests. For major changes, please open an issue first to discuss the proposed change.
