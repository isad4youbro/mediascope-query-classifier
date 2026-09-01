# MediaScope Query Classifier — Technical Notes

## 🇬🇧 English

This document contains the technical notes for the `mediascope/` implementation.

### Prediction pipeline

```text
query
  ↓
normalization
  ↓
TypeQuery prediction
  ↓
Title retrieval / reconstruction
  ↓
ContentType prediction
  ↓
post-processing
  ↓
structured result
```

### Core entry point

`solution.py` contains the public `PredictionModel` implementation.

The main pipeline combines:

- word-level TF-IDF features;
- character-level TF-IDF features;
- Logistic Regression;
- LinearSVC overrides;
- dictionary-based title recovery;
- KNN similarity retrieval;
- domain-specific post-processing.

### Local evaluation

```bash
python scripts/local_eval.py --data data/train.csv
```

### Grouped validation

```bash
python scripts/validate_group_kfold.py --data data/train.csv --folds 5
```

The grouped validation path is kept separate so that the evaluation procedure is explicit and reproducible for a compatible local dataset.

### Public-release policy

The competition dataset and private infrastructure are not part of this repository. The implementation is published as a technical / portfolio version of the solution.

---

## 🇷🇺 Русская версия

Этот файл содержит техническое описание реализации внутри `mediascope/`.

### Pipeline

```text
запрос
  ↓
нормализация
  ↓
TypeQuery
  ↓
retrieval / восстановление Title
  ↓
ContentType
  ↓
post-processing
  ↓
структурированный результат
```

### Основная точка входа

`solution.py` содержит публичную реализацию `PredictionModel`.

В pipeline используются:

- word-level TF-IDF;
- character-level TF-IDF;
- Logistic Regression;
- LinearSVC override;
- словари для восстановления Title;
- KNN similarity retrieval;
- предметные post-processing правила.

### Локальная оценка

```bash
python scripts/local_eval.py --data data/train.csv
```

### GroupKFold

```bash
python scripts/validate_group_kfold.py --data data/train.csv --folds 5
```

Grouped validation вынесена отдельно, чтобы процедура оценки оставалась явной и воспроизводимой на совместимом локальном датасете.

### Публичный релиз

Датасет соревнования и приватная инфраструктура не входят в репозиторий. Публикуется техническая / портфолио-версия решения.
