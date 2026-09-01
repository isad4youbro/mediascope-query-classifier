# recomend

Публичная версия проекта классификации поисковых видеозапросов.

## Структура

- `mediascope/solution.py` — основной класс `PredictionModel`.
- `mediascope/scripts/` — утилиты загрузки данных, локальной валидации и отправки решения.
- `mediascope/README.md` — техническое описание модели.

## Быстрый старт

```bash
cd mediascope
pip install -r requirements.txt
python -c "from solution import PredictionModel; print('ok')"
```

Если нужно локально обучить/проверить модель, сначала положите собственный датасет в `mediascope/data/train.csv`.
