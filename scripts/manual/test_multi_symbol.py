from daily_kronos_pipeline import DailyKronosPipeline
import pandas as pd
import joblib

symbols = ['TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'RELIANCE.NS']

for sym in symbols:
    print(f'\n=== {sym} ===')
    try:
        pipe = DailyKronosPipeline(sym, horizon=10)
        # Try to load existing model
        try:
            pipe.ml = joblib.load(f'models/ml_{sym}.pkl')
            pipe.trained = True
            print(f'Loaded existing model for {sym}')
        except:
            print(f'Training new model for {sym}...')
            pipe.train(period='3y')
        
        pred = pipe.predict()
        kronos = pred['kronos_dir']
        kronos_ret = pred['kronos_ret']
        ml_prob = pred['ml_prob']
        ml_dir = pred['ml_dir']
        mood = pred['mood']
        mood_score = pred['mood_score']
        direction = pred['direction']
        confidence = pred['confidence']
        tradeable = 'YES' if pred['tradeable'] else 'NO'
        
        print(f'Kronos: {kronos} ({kronos_ret:+.2f}%)')
        print(f'ML: {ml_prob:.1%} ({ml_dir})')
        print(f'Mood: {mood} ({mood_score:+.3f})')
        print(f'Direction: {direction} | Conf: {confidence:.1f}% | Trade: {tradeable}')
    except Exception as e:
        print(f'Error: {e}')