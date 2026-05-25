"""
Мониторинг обучения DT. Читает results.json и выводит таблицу метрик.
Запуск: python monitor.py [интервал_секунд]  (по умолчанию 15)
"""
import json, math, os, sys, time

RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'results.json')
DATASETS = ['1k', '2k', '5k', '10k']
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 15


def show(results):
    print(f"\n{'dataset':>8}  {'epochs':>6}  {'loss':>8}  {'SR':>7}  {'mean_len':>9}")
    print('-' * 46)
    for key in DATASETS:
        if key not in results:
            print(f"  {key:>6}  {'—':>6}  {'—':>8}  {'—':>7}  {'—':>9}  (waiting...)")
            continue
        h = results[key]
        ep   = len(h['train_loss'])
        loss = h['train_loss'][-1] if h['train_loss'] else float('nan')
        sr   = h['success_rate'][-1] if h['success_rate'] else float('nan')
        ml   = h['mean_length'][-1]  if h['mean_length']  else float('nan')
        sr_s = f"{sr:.1%}" if not math.isnan(sr) else "—"
        ml_s = f"{ml:.0f}" if not math.isnan(ml) else "—"
        print(f"  {key:>6}  {ep:>6}  {loss:>8.4f}  {sr_s:>7}  {ml_s:>9}")
    print()


print(f"Watching {RESULTS_FILE}  (refresh every {INTERVAL}s, Ctrl+C to stop)")

while True:
    ts = time.strftime('%H:%M:%S')
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, encoding='utf-8') as f:
                results = json.load(f)
            print(f"[{ts}]")
            show(results)
        except (json.JSONDecodeError, OSError):
            print(f"[{ts}] results.json is being written, retrying...")
    else:
        print(f"[{ts}] Waiting for results.json...")
    time.sleep(INTERVAL)
