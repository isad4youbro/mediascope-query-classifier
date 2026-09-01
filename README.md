<div align="center">

# 🎬 MediaScope Query Classifier

### Hybrid NLP Pipeline for Video-Search Query Understanding

`TF-IDF` · `Logistic Regression` · `LinearSVC` · `KNN` · `GroupKFold`

**🥇 1st place — AI Business SPB / MediaScope case**

</div>

[English](#-english) · [Русский](#-русская-версия)

---

## 🇬🇧 English

### Overview

This project is a **hybrid NLP / machine-learning pipeline** for turning noisy video-search queries into structured predictions.

For each query the system predicts:

- `TypeQuery` — whether the request is video-related;
- `Title` — a normalized / recovered title;
- `ContentType` — content type such as `film`, `series`, `cartoon`, `animated_series` or `other`.

The solution was built for the **AI Business SPB / MediaScope** case and achieved **1st place**.

### Why hybrid?

A single text classifier is not enough when the task also requires **entity / title recovery** and domain-specific corrections.

The pipeline combines statistical text features, linear models, retrieval, dictionaries, heuristics and post-processing rules.

```text
Raw query
    │
    ▼
Normalization
    │
    ├──────────────► TypeQuery classifier
    │
    ▼
Title recovery
    ├── dictionaries
    ├── heuristics
    └── KNN similarity retrieval
    │
    ▼
ContentType classifier
    │
    ▼
Domain-specific post-processing
    │
    ▼
TypeQuery + Title + ContentType
```

### 🧠 Main components

#### 1. Query normalization

The original text is cleaned and transformed into a more stable representation before classification and retrieval.

#### 2. TypeQuery classification

Uses **word- and character-level TF-IDF features** with linear classifiers.

The public solution includes:

- Logistic Regression;
- LinearSVC-based overrides;
- model / rule combination for difficult cases.

#### 3. Title recovery

Instead of directly generating a title, the pipeline combines:

- dictionaries and aliases;
- deterministic heuristics;
- KNN / similarity retrieval.

#### 4. ContentType classification

A separate classifier predicts the semantic type of the recovered content, followed by domain-specific post-processing.

### 🧪 Validation

The repository contains a local evaluation script and a GroupKFold validation path:

```bash
python scripts/local_eval.py --data data/train.csv
python scripts/validate_group_kfold.py --data data/train.csv --folds 5
```

`GroupKFold` is used to make validation more conservative when related queries could otherwise leak information between train and validation splits.

### 📁 Repository structure

```text
mediascope-query-classifier/
├── mediascope/
│   ├── solution.py                 # main PredictionModel
│   ├── scripts/
│   │   ├── local_eval.py           # local evaluation
│   │   ├── validate_group_kfold.py # grouped validation
│   │   ├── download_data.py        # data download helper
│   │   └── submit.py               # submission helper
│   ├── data/                       # local dataset (not public)
│   ├── pyproject.toml
│   ├── uv.lock
│   └── README.md                   # technical notes
├── .gitignore
└── README.md
```

### 🚀 Quick start

```bash
cd mediascope
pip install -r requirements.txt
python -c "from solution import PredictionModel; print(PredictionModel().ready)"
```

For local evaluation, place a compatible dataset in `mediascope/data/train.csv`.

### 🏆 Competition result

**1st place — AI Business SPB / MediaScope case.**

The project was designed around a practical search-understanding task rather than a single benchmark model, combining ML with retrieval and domain rules.

### 🔐 Public release

The public repository is sanitized:

- competition datasets are excluded;
- secrets are excluded;
- internal endpoints are not published;
- evaluation is performed against user-provided local data.

### 🛠 Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `TF-IDF` · `Logistic Regression` · `LinearSVC` · `KNN`

---

## 🇷🇺 Русская версия

### Обзор

Это **гибридный NLP / ML-пайплайн** для превращения шумных поисковых запросов по видео в структурированный результат.

Для каждого запроса система предсказывает:

- `TypeQuery` — является ли запрос видеозапросом;
- `Title` — нормализованное / восстановленное название;
- `ContentType` — тип контента: фильм, сериал, мультфильм, мультсериал и т. д.

Проект сделан для кейса **AI Business SPB / MediaScope** и занял **1 место**.

### Почему гибридный подход

Одного текстового классификатора здесь недостаточно: кроме классификации нужно **восстанавливать сущность / название** и исправлять неоднозначные случаи с учётом предметной области.

Поэтому pipeline объединяет TF-IDF, линейные модели, retrieval, словари, эвристики и post-processing правила.

```text
Исходный запрос
      │
      ▼
Нормализация
      │
      ├──────────────► Классификатор TypeQuery
      │
      ▼
Восстановление Title
      ├── словари
      ├── эвристики
      └── KNN similarity retrieval
      │
      ▼
Классификатор ContentType
      │
      ▼
Предметные post-processing правила
      │
      ▼
TypeQuery + Title + ContentType
```

### 🧠 Основные компоненты

**1. Нормализация запроса** — очистка и приведение текста к стабильному представлению.

**2. Классификация TypeQuery** — word/char TF-IDF + линейные модели, включая Logistic Regression и LinearSVC override.

**3. Восстановление Title** — словари, алиасы, эвристики и KNN / similarity retrieval.

**4. ContentType** — отдельная классификация типа контента + предметные post-processing правила.

### 🧪 Валидация

```bash
python scripts/local_eval.py --data data/train.csv
python scripts/validate_group_kfold.py --data data/train.csv --folds 5
```

GroupKFold используется для более консервативной проверки в случаях, когда похожие запросы могут привести к утечке между train и validation.

### 📁 Структура

```text
mediascope-query-classifier/
├── mediascope/
│   ├── solution.py
│   ├── scripts/
│   │   ├── local_eval.py
│   │   ├── validate_group_kfold.py
│   │   ├── download_data.py
│   │   └── submit.py
│   ├── data/
│   ├── pyproject.toml
│   ├── uv.lock
│   └── README.md
├── .gitignore
└── README.md
```

### 🚀 Быстрый запуск

```bash
cd mediascope
pip install -r requirements.txt
python -c "from solution import PredictionModel; print(PredictionModel().ready)"
```

Для локальной оценки нужно положить совместимый датасет в `mediascope/data/train.csv`.

### 🏆 Результат

**1 место — AI Business SPB / MediaScope.**

Проект построен не вокруг одной модели, а вокруг полного pipeline понимания запроса: ML + retrieval + предметные правила.

### 🔐 Публичная версия

Из репозитория исключены:

- оригинальные датасеты соревнования;
- секреты;
- внутренние endpoint'ы;
- environment-specific настройки.

Локальная оценка выполняется на данных, которые пользователь предоставляет сам.

### 🛠 Стек

`Python` · `pandas` · `NumPy` · `scikit-learn` · `TF-IDF` · `Logistic Regression` · `LinearSVC` · `KNN`
