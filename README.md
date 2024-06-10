### Stock Price Predictor
![image](https://github.com/GeetGhai/Stock-Market-Predictor/assets/154088899/14e6e895-6b10-42b3-bbe9-5c004decfb17)
### Overview

This project is a Stock Price Predictor that leverages machine learning techniques to forecast stock prices. The predictor is built using Python and various data science libraries to analyze historical stock data and make predictions.

### Project Structure

The project is organized in a Jupyter notebook with the following main sections:

1. **Data Collection**: Gathering historical stock price data.
2. **Data Preprocessing**: Cleaning and preparing the data for analysis.
3. **Feature Engineering**: Creating features that will help in predicting stock prices.
4. **Model Building**: Implementing and training machine learning models.
5. **Model Evaluation**: Assessing the performance of the models.
6. **Prediction**: Making future stock price predictions.
7. **Saving the Model**: Persisting the trained model for future use.

### Installation

To run the project, you need the following dependencies:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- pickle
- Jupyter Notebook

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib pickle-mixin jupyter
```

### Usage

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Stock_Market_Prediction_Model_Creation.ipynb
   ```

3. **Run the Cells**: Execute the cells in the notebook to load data, preprocess it, build models, and make predictions.

### Data

The dataset used in this project includes historical stock price data, which typically consists of the following features:
- `Date`: The date of the stock prices.
- `Open`: The opening price of the stock on that date.
- `High`: The highest price of the stock on that date.
- `Low`: The lowest price of the stock on that date.
- `Close`: The closing price of the stock on that date.
- `Volume`: The volume of stocks traded on that date.

### Data Preprocessing

The preprocessing steps include:
- Handling missing values
- Converting date columns to datetime objects
- Scaling numerical features

### Feature Engineering

The following features are created to enhance the predictive power of the model:
- Moving averages
- Lagged returns
- Technical indicators (e.g., RSI, MACD)

### Model Building

Several machine learning models are implemented and trained on the processed data:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- LSTM (Long Short-Term Memory) Neural Network for time series forecasting

### Model Evaluation

The models are evaluated using metrics such as:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) score

### Prediction

The trained model is used to make future stock price predictions. The prediction results are visualized using matplotlib to compare with actual stock prices.

### Saving the Model

The trained model and other important objects are saved using the pickle module for future use. The following files are generated:
- `model.pkl`: Pickled machine learning model.
- `scaler.pkl`: Pickled scaler object.

### Example

```python
# Load the processed data and model
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Making a prediction
new_data = [...]  # New input data
scaled_data = scaler.transform(new_data)
prediction = model.predict(scaled_data)

print(prediction)
```

### Conclusion

This Stock Price Predictor project demonstrates how to use historical stock data and machine learning techniques to predict future stock prices. By following the steps outlined in this README, you can set up and run the predictor on your own machine.

![image](https://github.com/GeetGhai/Stock-Market-Predictor/assets/154088899/e4a90991-f43c-4e90-826e-c6fddc6e8881)

