# Architecture — Планируемая архитектура проекта

## Структура файлов проекта

```
project/
├── sim/
│   ├── env.py          — DiffDriveEnv: симулятор
│   └── expert.py       — ExpertController: двухфазный контроллер
├── data/
│   └── generate.py     — генерация датасета, сохранение .npz
├── models/
│   └── bc.py           — BCModel: MLP для Behavior Cloning
├── train.py            — цикл обучения, сохранение весов
├── eval.py             — визуализация траекторий, метрики
└── llm/                — этот каталог
```

---

## sim/env.py — DiffDriveEnv

**Состояние робота:** `(x, y, theta)` — позиция и ориентация в 2D

**Действие:** `(v, omega)` — линейная и угловая скорость

**Наблюдение (observation):** `(dist_to_goal, angle_to_goal)` — относительные координаты цели
- `dist_to_goal` — евклидово расстояние до цели
- `angle_to_goal` — угол до цели в системе координат робота (в радианах)

**Почему relative obs:** обобщается на произвольные позиции старта/цели без изменения сети.

**Интерфейс:**
```python
env = DiffDriveEnv(dt=0.1, goal_threshold=0.1)
obs = env.reset(start=(x0, y0, theta0), goal=(gx, gy))
obs, done = env.step(action)   # action = (v, omega)
```

**Кинематика (дискретная, Эйлер):**
```
x_new     = x + v * cos(theta) * dt
y_new     = y + v * sin(theta) * dt
theta_new = theta + omega * dt
```

---

## sim/expert.py — ExpertController

**Двухфазный алгоритм:**

1. **Фаза 1 — поворот к цели:**
   Пока `|angle_to_goal| > angle_threshold`: подавать `omega = k_rot * angle_to_goal`, `v = 0`

2. **Фаза 2 — движение вперёд:**
   Пока `dist_to_goal > goal_threshold`: подавать `v = v_max`, `omega = k_align * angle_to_goal`
   (небольшая коррекция курса во время движения)

**Интерфейс:**
```python
expert = ExpertController(v_max=1.0, k_rot=2.0, k_align=1.0, angle_threshold=0.1)
action = expert.act(obs)   # obs = (dist, angle) → action = (v, omega)
```

---

## data/generate.py — Генерация датасета

**Что делает:**
- Запускает N эпизодов с разными start/goal
- На каждом шаге: `obs → expert.act(obs) → записать (obs, action)`
- Сохраняет в `.npz`: `{'observations': array(N, 2), 'actions': array(N, 2)}`

**Интерфейс:**
```python
generate_dataset(n_episodes=1000, max_steps=200, save_path='data/dataset.npz')
```

---

## models/bc.py — BCModel (MLP)

**Архитектура:**
- Вход: 2 (dist_to_goal, angle_to_goal)
- Скрытые слои: 2 слоя × 64 нейрона, активация ReLU
- Выход: 2 (v, omega)

**Без нормализации входов на первом шаге** — упрощение для старта.

**Интерфейс:**
```python
model = BCModel(input_dim=2, hidden_dim=64, output_dim=2)
action = model(obs_tensor)   # torch.Tensor
```

---

## train.py — Обучение

- Загрузить датасет → создать DataLoader
- MSE loss: `loss = mse(model(obs), action)`
- Adam оптимизатор
- N эпох, сохранить лучшие веса по val_loss

---

## eval.py — Оценка и визуализация

- Загрузить обученную модель
- Прогнать K тестовых эпизодов: эксперт vs BC-модель
- Нарисовать траектории (matplotlib)
- Метрики: success rate, средняя длина траектории
