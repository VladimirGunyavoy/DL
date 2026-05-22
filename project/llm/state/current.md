# Current State — Текущее состояние проекта

*Последнее обновление: 2026-05-21*

---

## Компоненты

| Компонент | Файл | Статус |
|---|---|---|
| Симулятор | `sim/env.py` | готово (переработан) |
| Эксперт-контроллер | `sim/expert.py` | готово (переработан) |
| Генерация датасета (BC) | `sim/generate_data.py` | готово (устарел, не используется) |
| Генерация датасета (DT) | `tests/test_generation.ipynb` | готово |
| Датасеты HDF5 | `data/dataset_{1k,2k,5k,10k}.h5` | готово |
| DT модель | `dt/model.py` | готово |
| DT датасет | `dt/dataset.py` | готово |
| DT оценка (роллаут) | `dt/evaluate.py` | готово |
| Обучение DT | `tests/train_dt.ipynb` | в процессе — запускается, но не закончено |

---

## Что сделано за сессию 2026-05-21

- [x] Obs переведён на cos/sin: `[dist, cos(angle), sin(angle)]` — shape (3,)
- [x] State переведён на cos/sin: `[x, y, cos_theta, sin_theta]` — shape (4,)
- [x] `sim/env.py` — убран `self.theta`, хранятся `cos_theta/sin_theta`, JIT через Numba
- [x] `sim/expert.py` — JIT через Numba, принимает `cos_t, sin_t` из state
- [x] `ExpertController` — добавлен `noise_std` для выхода из сингулярностей
- [x] `tests/test_generation.ipynb` — генерация датасета для DT (HDF5, RTG, timesteps)
- [x] Датасеты `data/dataset_{1k,2k,5k,10k}.h5` — сгенерированы
- [x] `dt/model.py` — DecisionTransformer (GPT-like, causal mask)
- [x] `dt/dataset.py` — SequenceDataset с предгенерацией кэша и нормализацией
- [x] `dt/evaluate.py` — rollout в симуляторе, success rate + mean length
- [x] `tests/train_dt.ipynb` — цикл обучения, AMP, AdamW, CosineAnnealing

## В процессе

Обучение DT запускается, но не завершено. Проблема: роллаут медленный (CPU-bound).
Следующий шаг: запустить полный прогон по всем 4 датасетам.

## Сломано / Блокеры

- `torch.compile` не работает на Windows (нет Triton) — убрано
- rollout медленный: 20 эп × ~178 шагов × 1 forward pass каждый шаг
