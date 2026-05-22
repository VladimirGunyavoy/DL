# Dependencies — Внешние зависимости

## Обязательные

| Пакет | Версия | Зачем |
|---|---|---|
| `numpy` | >=1.21 | Кинематика, датасет, массивы |
| `torch` | >=1.13 | MLP модель, обучение, инференс |
| `matplotlib` | >=3.5 | Визуализация траекторий |

## Опциональные (обсудить перед добавлением)

| Пакет | Зачем | Статус |
|---|---|---|
| `gymnasium` | gym-совместимый интерфейс для env | Не добавляли, обсудить если нужно |
| `tqdm` | прогресс-бар при обучении | Не критично, легко добавить |

## Установка

```bash
pip install numpy torch matplotlib
```

Или через conda если используется Skoltech environment:
```bash
conda install numpy pytorch matplotlib -c pytorch
```

## Проверка окружения

```python
import numpy as np
import torch
import matplotlib
print(f"numpy {np.__version__}, torch {torch.__version__}, matplotlib {matplotlib.__version__}")
```
