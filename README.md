# Solar Irradiance Forecasting with Deep Learning (RNN, GRU, LSTM, TCN)

This project presents a benchmark comparison of four deep learning architectures (RNN, GRU, LSTM, TCN) for **solar irradiance prediction** using meteorological variables in Egypt. The study leverages data from multiple latitude-longitude sheets and evaluates models using MAE, RMSE, R², and inference time.

## 🚀 Models Implemented
- Recurrent Neural Network (RNN)
- Gated Recurrent Unit (GRU)
- Long Short-Term Memory (LSTM)
- Temporal Convolutional Network (TCN)

## 📊 Input Features
- Total Column Water Vapor (mm)
- Cloud Cover (%)
- 2-meter Temperature (°C)
- Wind Speed (m/s)
- Wind Direction (deg)
- Clear-Sky GHI (W/m²)
- Actual GHI (W/m²) → **Target**

## 🗃 Data Preprocessing
- Data from 48 spatially distributed Excel sheets (`All_Data.xlsx`)
- Aggregated, normalized using `MinMaxScaler`
- Train set: First 10 years
- Test set: Last 3 years

## 📦 Dependencies
```bash
pip install pandas numpy scikit-learn tensorflow keras-tcn matplotlib openpyxl tqdm
