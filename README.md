# PocketOptionML-Bot í´–

An automated trading bot for PocketOption using Machine Learning and Technical Analysis.

## Features
- Machine Learning predictions using LogisticRegression
- Technical indicators (SMA, MACD, RSI)
- Automated trade execution
- Trade logging and model persistence
- Connection resilience with auto-reconnect
- Configurable parameters

## Requirements
```
pocketoptionapi
pandas
scikit-learn
ta-lib
joblib
```

## Setup
1. Install dependencies:
```bash
pip install pandas scikit-learn joblib ta-lib
```

2. Configure your credentials in `config.py`
3. Run the bot:
```bash
python ai_po.py
```

## Disclaimer
This is for educational purposes only. Trade at your own risk.
