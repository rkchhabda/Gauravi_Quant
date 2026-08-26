from daily_kronos_pipeline import DailyKronosPipeline
import joblib

sym = 'INFY.NS'
pipe = DailyKronosPipeline(sym, horizon=10)
try:
    pipe.ml = joblib.load('models/ml_INFY.NS.pkl')
    pipe.trained = True
    print('Loaded existing model')
except:
    print('Training...')
    pipe.train(period='3y')

pred = pipe.predict()
print('Symbol:', pred['symbol'])
print('Kronos:', pred['kronos_dir'], '(' + str(pred['kronos_ret']) + '%)')
print('ML:', str(round(pred['ml_prob']*100,1)) + '% (' + pred['ml_dir'] + ')')
print('Mood:', pred['mood'], '(' + str(pred['mood_score']) + ')')
print('Direction:', pred['direction'], '| Conf:', str(pred['confidence']) + '% | Trade:', 'YES' if pred['tradeable'] else 'NO')