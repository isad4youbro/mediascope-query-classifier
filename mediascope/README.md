# Mediascope Query Classifier

Классификатор поисковых запросов с выходом в формате:

- `TypeQuery` — видеозапрос (`0/1`)
- `Title` — извлеченное/нормализованное название
- `ContentType` — тип контента (`""`, `"фильм"`, `"сериал"`, `"мультфильм"`, `"мультсериал"`, `"прочее"`)

## Архитектура

Гибридный подход:

1. Нормализация запроса
2. Классификация `TypeQuery`
3. Восстановление `Title` (словари + эвристики + KNN retrieval)
4. Классификация `ContentType`
5. Пост-правила для доменных конфликтов

## Файлы

- `solution.py` — основной `PredictionModel`
- `scripts/local_eval.py` — локальная оценка
- `scripts/validate_group_kfold.py` — leak-safe GroupKFold
- `scripts/download_data.py` — загрузка train-данных из API
- `scripts/submit.py` — сборка и отправка bundle

## Локальный запуск

```bash
pip install -r requirements.txt
python -c "from solution import PredictionModel; print(PredictionModel().ready)"
```

## Локальная валидация

1. Положите датасет в `data/train.csv`.
2. Запустите:

```bash
python scripts/local_eval.py --data data/train.csv
python scripts/validate_group_kfold.py --data data/train.csv --folds 5
```

## Переменные окружения

Смотрите `.env.example`.

Публичная версия уже очищена от приватных ключей и внутренних адресов.
