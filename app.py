import torch
import logging
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
from flask import Flask, Response, json
from typing import Dict, Any, List

app = Flask(__name__)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Define the updated model with the correct architecture
class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int, output_size: int, num_layers: int, dropout: float) -> None:
        super(EnhancedBiLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_layer_size, num_layers=num_layers, 
            dropout=dropout, batch_first=True, bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_layer_size * 2)
        self.batch_norm = nn.BatchNorm1d(hidden_layer_size * 2)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size * 3)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size)
        c_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size)
        
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        lstm_out = self.layer_norm(lstm_out)
        lstm_out_last = lstm_out[:, -1]
        lstm_out_last = self.batch_norm(lstm_out_last)
        predictions = self.linear(lstm_out_last) 
        return predictions

# Load and initialize the model once
model = EnhancedBiLSTMModel(input_size=1, hidden_layer_size=115, output_size=1, num_layers=2, dropout=0.3)
model.load_state_dict(torch.load("enhanced_bilstm_model_validation.pth", map_location=torch.device('cpu')))
model.eval()

# Fetch historical data from Binance
def fetch_binance_data(symbol: str = "ETHUSDT", interval: str = "1m", limit: int = 1000) -> List[Any]:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Failed to fetch data from Binance: {response.status_code}")
        raise ValueError("Binance API error")
    return response.json()

@app.route("/inference/<int:timeframe>/<string:token>")
def get_price_inference(timeframe: int, token: str) -> Response:
    try:
        token = token.upper()
        symbol_map: Dict[str, str] = {'ETH': 'ETHUSDT', 'BTC': 'BTCUSDT', 'BNB': 'BNBUSDT', 'SOL': 'SOLUSDT', 'ARB': 'ARBUSDT'}
        if token not in symbol_map:
            return Response(json.dumps({"error": "Unsupported token"}), status=400, mimetype='application/json')

        symbol = symbol_map[token]
        data = fetch_binance_data(symbol)
        df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", 
                                         "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
                                         "taker_buy_quote_asset_volume", "ignore"])
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["close_time", "close"]]
        df.columns = ["date", "price"]
        df["price"] = df["price"].astype(float)
        timeframe_map: Dict[int, int] = {5: 5, 10: 10, 20: 20, 24: 1440}
        df = df.tail(timeframe_map.get(timeframe, 10))

        logger.info(f"Current Price: {df.iloc[-1]['price']} at {df.iloc[-1]['date']}")
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))
        seq = torch.FloatTensor(scaled_data).view(1, -1, 1)

        with torch.no_grad():
            y_pred = model(seq).detach()
        
        predicted_prices = scaler.inverse_transform(y_pred.numpy().reshape(-1, 1))
        prediction_map: Dict[int, float] = {10: predicted_prices[0][0], 20: predicted_prices[1][0], 24: predicted_prices[2][0]}
        predicted_price = round(float(prediction_map.get(timeframe, predicted_prices[0][0])), 2)

        logger.info(f"Prediction: {predicted_price}")
        return Response(json.dumps(predicted_price), mimetype='application/json')
    except Exception as e:
        logger.exception("Price inference failed")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9191)
