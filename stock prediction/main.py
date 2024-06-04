from keras.models import load_model
import yfinance as yf
model = load_model(r'C:\Users\MY PC\stock prediction\Stock Predictions Model.keras')
data = yf.download(stock, start ,end)
print(data)

