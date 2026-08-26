from daily_kronos_pipeline import DailyKronosPipeline
import joblib

symbols = ['TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'RELIANCE.NS']
for sym in symbols:
    try:
        pipe = DailyKronosPipeline(sym, horizon=10)
        pipe.ml = joblib.load('models/ml_' + sym + '.pkl')
        pipe.trained = True
        pred = pipe.predict()
        print(sym + ': ' + pred['gate'] + ' | ' + pred['direction'] + ' | Conf: ' + str(pred['confidence']) + '% | Kronos: ' + pred['kronos_dir'] + '(' + str(pred['kronos_ret']) + '%) ML: ' + pred['ml_dir'] + '(' + str(pred['ml_prob']) + ') Mood: ' + pred['mood'])
    except Exception as e:
        print(sym + ': Error - ' + str(e))