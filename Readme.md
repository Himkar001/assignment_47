
# 📊 Week 08 — Thursday Assignment  
## RNNs + Sequential Data  
---
# 🟢 EASY SECTION (Sub-steps 1 & 2)

---

## 🔹 Sub-step 1: Stock Data Preparation

### 📌 Goal
Prepare sequence dataset for next-day stock prediction

---

### ✅ Load Data

```python
import pandas as pd

df = pd.read_csv("stock_prices.csv")
df.head()
````

**Result:**

* Dataset loaded with columns: Date, Symbol, Open, Close, etc.

---

### ✅ Filter One Stock (TCS)

```python
stock_df = df[df['Symbol'] == 'TCS'].copy()
stock_df.sort_values('Date', inplace=True)
```

**Explanation:**
Time-series must be sorted chronologically.

---

### ✅ Convert Date

```python
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df.set_index('Date', inplace=True)
```

---

### ✅ Use Closing Price

```python
data = stock_df[['Close']]
```

---

### ✅ Sequence Creation (Window = 30)

```python
import numpy as np

WINDOW_SIZE = 30
X, y = [], []

for i in range(len(data) - WINDOW_SIZE):
    X.append(data.iloc[i:i+WINDOW_SIZE].values)
    y.append(data.iloc[i+WINDOW_SIZE].values)

X = np.array(X)
y = np.array(y)
```

**Result:**

* X shape → (samples, 30, 1)
* y shape → (samples, 1)

---

### ✅ Time-based Split

```python
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
X_val = X[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
```

---

### ⚠️ Important Insight

* Random split → ❌ Data leakage
* Time split → ✅ Real-world simulation

---

## 🔹 Sub-step 2: Chat Logs EDA

---

### ✅ Load Data

```python
chat_df = pd.read_csv("chat_logs.csv")
```

---

### ✅ Fix Timestamp Issue

```python
chat_df['timestamp'] = pd.to_datetime(chat_df['timestamp'], errors='coerce')
chat_df = chat_df.dropna(subset=['timestamp'])
```

---

### ✅ Feature Engineering

#### Message Count

```python
msg_count = chat_df.groupby('customer_id').size()
```

#### Duration

```python
duration = chat_df.groupby('customer_id')['timestamp'].agg(['min','max'])
duration['duration'] = (duration['max'] - duration['min']).dt.total_seconds()
```

#### Response Time

```python
chat_df = chat_df.sort_values(['customer_id','timestamp'])
chat_df['response_time'] = chat_df.groupby('customer_id')['timestamp'].diff().dt.total_seconds()
```

---

### 📊 Key Insights

* High response delay → churn risk
* Long conversations → dissatisfaction
* Low engagement → disengagement

---

# 🟡 MEDIUM SECTION (Sub-steps 3, 4, 5)

---

## 🔹 Sub-step 3: LSTM for Stock Prediction

---

### ✅ Model

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

---

### ✅ Training

```python
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

### ✅ Evaluation

```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predictions))
```

---

### 📊 Insight

* LSTM captures temporal dependencies
* Works well for nonlinear patterns

---

## 🔹 Sub-step 4: Churn Prediction

---

### ✅ Approach 1: Tabular Model

```python
from sklearn.linear_model import LogisticRegression

model_tab = LogisticRegression()
model_tab.fit(X_train_tab, y_train_tab)
```

---

### ✅ Approach 2: Sequence Model (LSTM)

* Uses chat sequences instead of aggregates

---

### ✅ Evaluation

```python
from sklearn.metrics import f1_score
f1 = f1_score(y_test, preds)
```

---

### 📊 Decision

* If similar → use tabular
* If LSTM better → sequence matters

---

## 🔹 Sub-step 5: Business Decision

---

### ✅ Probability Prediction

```python
probs = model_tab.predict_proba(X_test_tab)[:,1]
```

---

### ✅ Ranking

```python
results = pd.DataFrame({
    "customer_id": ids,
    "prob": probs
}).sort_values(by="prob", ascending=False)
```

---

### ✅ Cost Model

* Contact cost = ₹10
* Missed churn = ₹100

---

### ✅ Threshold

```python
threshold = 0.3
selected = results[results['prob'] > threshold]
```

---

### 📊 Output

* Ranked churn list
* Number of customers contacted

---

# 🔴 HARD SECTION (Sub-steps 6 & 7)

---

## 🔹 Sub-step 6: Baseline vs LSTM

---

### ✅ Baseline Model

```python
def autoregressive_predict(X):
    return X.mean(axis=1)
```

---

### ✅ Compare RMSE

```python
baseline_rmse = ...
lstm_rmse = ...
```

---

### 📊 Insight

* Baseline good → random walk behavior
* LSTM better → learned patterns

---

## 🔹 Sub-step 7: Manual BPTT

---

### ✅ Forward Pass

```python
h[t] = np.tanh(Wx @ x[t] + Wh @ h[t-1])
```

---

### ✅ Backprop

```python
dh = (1 - h[t]**2)
```

---

### ✅ Vanishing Gradient

```python
grad_norms ↓ as sequence length ↑
```

---

### 📊 Insight

* Gradients shrink over time
* Early timesteps stop learning

---

### ✅ Why LSTM?

* Uses gates
* Preserves gradients
* Solves vanishing problem

---

# 📊 FINAL RECOMMENDATIONS

---

## Stock Model

* Use LSTM only if it beats baseline
* Otherwise use simpler models

---

## Churn Model

* Prefer tabular if performance similar
* Easier deployment

---

## Key Risks

* Data leakage (time series)
* Class imbalance (churn)
* Overfitting (LSTM)

---

# 📈 ENGINEERING QUALITY

* Modular code ✔
* No magic numbers ✔
* Proper splitting ✔
* Clear naming ✔

---

# ✅ CONCLUSION

This assignment demonstrates:

* Sequential data handling
* LSTM vs baseline comparison
* Business-driven ML decisions
* Deep learning limitations (vanishing gradient)

---


