# Plan — Пошаговый план реализации

*Обновлён 2026-05-21: проект переориентирован с BC на Decision Transformer*

---

## Шаги

### Шаг 1 — DiffDriveEnv [x]
**Файл:** `sim/env.py` — **ГОТОВО**
- obs: `[dist, cos(angle), sin(angle)]` shape (3,)
- state: `[x, y, cos_theta, sin_theta]` shape (4,)
- JIT через Numba (`_step_jit`)
- done: `dist < 0.05 AND abs(theta) < 0.1`

---

### Шаг 2 — ExpertController [x]
**Файл:** `sim/expert.py` — **ГОТОВО**
- Геометрический контроллер (не двухфазный — см. decisions.md 004)
- JIT через Numba (`_act_jit`)
- `noise_std` для выхода из сингулярностей при `rp=0`
- `act(obs, state)` — принимает оба аргумента

---

### Шаг 3 — Генерация датасета [x]
**Файл:** `tests/test_generation.ipynb` — **ГОТОВО**
- Формат: HDF5, per-trajectory группы
- Поля: obs(T,3), actions(T,2), rewards(T,), rtg(T,), timesteps(T,), terminals(T,)
- Субсеты: `data/dataset_{1k,2k,5k,10k}.h5`

---

### Шаг 4 — Decision Transformer [x]
**Файл:** `dt/model.py` — **ГОТОВО**
- GPT-like, causal mask
- Интерлив токенов: [RTG, obs, act] × K
- Предсказание action из obs-токенов

---

### Шаг 5 — Обучение [ ]
**Файл:** `tests/train_dt.ipynb` — **В ПРОЦЕССЕ**

Задачи:
- [x] SequenceDataset с нормализацией и кэшем (`dt/dataset.py`)
- [x] Цикл обучения: AdamW, MSE, CosineAnnealing, AMP
- [x] Сохранение лучших весов в `checkpoints/`
- [ ] Полный прогон по 1k / 2k / 5k / 10k
- [ ] Сохранение `results.json`

---

### Шаг 6 — Оценка и визуализация [ ]
**Файл:** `tests/train_dt.ipynb` (ячейка с графиками) — **НЕ НАЧАТО**

Задачи:
- [ ] Графики: train_loss, success_rate, mean_length по датасетам
- [ ] Сохранить `results.png`

---

## Прогресс

```
[Шаг 1] DiffDriveEnv     ██████████ 100%
[Шаг 2] ExpertController  ██████████ 100%
[Шаг 3] Датасет (DT)     ██████████ 100%
[Шаг 4] DT модель        ██████████ 100%
[Шаг 5] Обучение         ████░░░░░░  40%
[Шаг 6] Оценка           ░░░░░░░░░░   0%
```
