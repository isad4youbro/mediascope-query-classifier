import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def _sset(words: str) -> set[str]:
    return set(words.split())


class PredictionModel:
    batch_size = 128
    ARTIFACT_NAME = "mediascope_model.pkl"

    TOKEN_RE = re.compile(r"[0-9a-zа-я]+")
    LATIN_RE = re.compile(r"[a-z]")

    NOISE_WORDS = _sset(
        "для с и в на из от по к до не а о у без что или это его вы мы их "
        "the for and with in of to an"
    )

    GENERIC_WORDS = _sset(
        "набор комплект портативный портативная портативное электрический электрическая электрическое "
        "универсальный универсальная универсальное детский детская детское детские умный умная умное мини mini "
        "новый новая новое новые большой большая большое маленький маленькая маленькое бестселлер "
        "премиальный премиальная эксклюзивный эксклюзивная эксклюзивное профессиональный профессиональная "
        "смотреть онлайн бесплатно хорошем качестве full hd сезон сезона серия серии все подряд "
        "фильм фильмы сериал сериалы мультфильм мультсериал дорама аниме русский русском языке дата выхода "
        "трейлер актеры актёры отзывы новинки скачать торрент tv шоу"
    )

    SIZE_COLOR_WORDS = _sset(
        "черный белый красный синий зеленый серый розовый голубой желтый оранжевый фиолетовый "
        "коричневый бежевый черная белая красная синяя серая розовая черное белое красное синее "
        "серое xs s m l xl xxl xxxl black white red blue green grey gray pink"
    )

    FALLBACK_VIDEO_WORDS = _sset(
        "смотреть фильм фильмы сериал сериалы сезон серия серии мультфильм мультсериал "
        "дорама аниме кино трейлер"
    )

    FORCE_VIDEO_WORDS = _sset(
        "смотреть сезон серия серии аниме дорама трейлер кино сериал сериалы фильм фильмы мультсериал мультфильм"
    )

    STREAMING_WORDS = _sset("netflix okko premier rutube wink kion start ivi кинопоиск амедиатека")

    VIDEO_NEGATIVE_WORDS = _sset(
        "центр институт университет банк курс курсы школа магазин купить заказать доставка оплата"
    )

    TITLE_NOISE = _sset(
        "смотреть онлайн бесплатно хорошем качестве full hd трейлер скачать торрент русский язык субтитры озвучка "
        "год года новый новинка лучший топ все подряд 1080 720 480 4k uhd blu ray dvd extended cut premiere "
        "премьера вышел вышла дата выход актер актеры актёры отзыв рецензия wiki wikipedia серия серии сезон "
        "сезоны часть эпизод episode season"
    )

    SERIAL_WORDS = _sset("сезон сезона сезоны серия серии season episode episodes")
    SHOW_WORDS = _sset("выпуск выпуски шоу эфир канал тв tv прямой матч трансляция")
    FILM_WORDS = _sset("фильм фильмы кино")

    ARTIFACT_KEYS = [
        "type_model",
        "content_model",
        "content_svc_model",
        "type_threshold",
        "query_to_title",
        "query_to_content",
        "lex_norm",
        "lex_orig",
        "lex_freq",
        "lex_tok_cnt",
        "knn_neighbors",
        "knn_title_threshold",
        "knn_content_threshold",
        "knn_overlap_threshold",
        "heuristic_title_min_ratio",
        "title_token_ratio",
        "knn_query_norm",
        "knn_titles",
        "knn_contents",
    ]

    def __init__(self) -> None:
        self.type_threshold = 0.09
        self.max_lexicon_titles = 50000
        self.lex_token_weight = 10

        self.max_heuristic_tokens = 5
        self.min_heuristic_chars = 2

        self.knn_neighbors = 3
        self.knn_title_threshold = 0.70
        self.knn_content_threshold = 0.95
        self.knn_overlap_threshold = 0.50
        self.heuristic_title_min_ratio = 1.05

        self.ready = False

        artifact = Path(__file__).resolve().with_name(self.ARTIFACT_NAME)
        if artifact.exists():
            try:
                self._load_artifact(artifact)
                self.ready = True
            except Exception:
                self.ready = False

        if not self.ready:
            train_df = self._load_train_data()
            if train_df is not None:
                try:
                    self._fit(train_df)
                    self.ready = True
                    self._save_artifact(artifact)
                except Exception:
                    self.ready = False

        self._apply_runtime_overrides()

    def _apply_runtime_overrides(self) -> None:
        overrides = {
            "TYPE_THRESHOLD": ("type_threshold", float),
            "KNN_TITLE_THRESHOLD": ("knn_title_threshold", float),
            "KNN_CONTENT_THRESHOLD": ("knn_content_threshold", float),
            "KNN_OVERLAP_THRESHOLD": ("knn_overlap_threshold", float),
            "MAX_HEURISTIC_TOKENS": ("max_heuristic_tokens", int),
        }
        for env_name, (attr, cast) in overrides.items():
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            try:
                setattr(self, attr, cast(raw))
            except Exception:
                pass

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["QueryText"]].copy()
        queries = out["QueryText"].fillna("").astype(str).tolist()

        if not self.ready:
            type_pred = np.array([self._fallback_type(q) for q in queries], dtype=np.int8)
            out["TypeQuery"] = type_pred.astype(int)
            out["Title"] = [self._product_type_extract(q) if t == 1 else "" for q, t in zip(queries, type_pred)]
            out["ContentType"] = [self._fallback_content(q, int(t)) for q, t in zip(queries, type_pred)]
            return out

        q_norm = [self._norm(q) for q in queries]
        type_pred = (self.type_model.predict_proba(q_norm)[:, 1] >= self.type_threshold).astype(np.int8)

        for i, qn in enumerate(q_norm):
            if type_pred[i] == 0 and self._has_force_video_words(qn):
                type_pred[i] = 1
            if type_pred[i] == 0:
                lex_title = self._lex_match(qn)
                if lex_title and len(self._tokenize(self._norm(lex_title))) >= 2:
                    type_pred[i] = 1

        content_pred = self.content_model.predict(q_norm).tolist()
        svc_pred = self.content_svc_model.predict(q_norm).tolist()
        knn_title, knn_content, knn_sim = self._knn_batch(q_norm)

        title_pred: list[str] = []
        for i, (raw, qn, tp) in enumerate(zip(queries, q_norm, type_pred)):
            if tp == 0:
                content_pred[i] = ""
                title_pred.append("")
                continue

            title = self.query_to_title.get(qn, "")
            source = "dict"

            if not title:
                title = self._lex_match(qn)
                source = "lex" if title else ""

            if not title:
                title = self._product_type_extract(raw)
                source = "heuristic" if title else ""

            is_latin = self._has_latin(qn)
            tok_cnt = len(self._tokenize(self._norm(title))) if title else 0
            overlap_min = max(0.25, self.knn_overlap_threshold - (0.15 if is_latin else 0.0))

            if (
                title
                and knn_title[i]
                and knn_sim[i] >= self.knn_title_threshold
                and (is_latin or tok_cnt >= 2 or knn_sim[i] >= 0.90)
                and self._token_overlap_ratio(self._norm(title), self._norm(knn_title[i])) >= overlap_min
            ):
                title = knn_title[i]
                source = "knn"

            if source == "heuristic" and title and self._title_token_ratio_max(self._norm(title)) < self.heuristic_title_min_ratio:
                title = ""
                source = ""

            if len(title) < self.min_heuristic_chars:
                title = ""

            if qn in self.query_to_content:
                content_pred[i] = self.query_to_content[qn]
            elif self._prefer_svc(qn, title, content_pred[i], svc_pred[i]):
                content_pred[i] = svc_pred[i]

            if source == "knn" and knn_sim[i] >= self.knn_content_threshold and knn_content[i]:
                content_pred[i] = knn_content[i]

            content_pred[i] = self._apply_content_rules(qn, content_pred[i])
            title_pred.append(title)

        out["TypeQuery"] = type_pred.astype(int)
        out["Title"] = title_pred
        out["ContentType"] = content_pred
        return out

    @staticmethod
    def _features(char_range: tuple[int, int]) -> FeatureUnion:
        return FeatureUnion(
            [
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=char_range,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

    def _fit(self, train_df: pd.DataFrame) -> None:
        data = train_df.copy().fillna("")
        for c in ["QueryText", "Title", "ContentType"]:
            data[c] = data[c].astype(str)

        self.type_model = Pipeline(
            [
                ("tfidf", self._features((3, 5))),
                ("clf", LogisticRegression(max_iter=800, C=3.0, class_weight="balanced")),
            ]
        )
        self.type_model.fit(data["QueryText"].map(self._norm), data["TypeQuery"].astype(int))

        pos = data[data["TypeQuery"].astype(int) == 1].copy()

        feat = self._features((2, 5))
        self.content_model = Pipeline(
            [
                ("tfidf", feat),
                ("clf", LogisticRegression(max_iter=1400, C=3.0)),
            ]
        )
        self.content_model.fit(pos["QueryText"].map(self._norm), pos["ContentType"])

        self.content_svc_model = Pipeline(
            [
                ("tfidf", feat),
                ("clf", LinearSVC(C=1.0)),
            ]
        )
        self.content_svc_model.fit(pos["QueryText"].map(self._norm), pos["ContentType"])

        titled = pos[pos["Title"].str.strip() != ""].copy()
        titled["q_norm"] = titled["QueryText"].map(self._norm)

        by_query = titled.groupby("q_norm").agg(
            Title=("Title", lambda s: s.value_counts().index[0]),
            ContentType=("ContentType", lambda s: s.value_counts().index[0]),
        )

        self.query_to_title = by_query["Title"].to_dict()
        self.query_to_content = by_query["ContentType"].to_dict()

        self.knn_query_norm = by_query.index.tolist()
        self.knn_titles = by_query["Title"].tolist()
        self.knn_contents = by_query["ContentType"].tolist()
        self._build_knn_index()

        title_tok: Counter[str] = Counter()
        empty_tok: Counter[str] = Counter()

        for t in titled["Title"]:
            title_tok.update(set(self._tokenize(self._norm(t))))
        for q in pos[pos["Title"].str.strip() == ""]["QueryText"]:
            empty_tok.update(set(self._tokenize(self._norm(q))))

        self.title_token_ratio = {
            tok: (cnt + 1.0) / (empty_tok.get(tok, 0) + 1.0)
            for tok, cnt in title_tok.items()
        }

        self.lex_norm = []
        self.lex_orig = []
        self.lex_freq = []
        self.lex_tok_cnt = []
        self.lex_tok_index = defaultdict(list)

        entries: list[tuple[str, str, int, int, tuple[str, ...]]] = []
        for title, freq in titled["Title"].value_counts().items():
            t_norm = self._norm(title)
            tokens = self._tokenize(t_norm)
            if tokens:
                entries.append((t_norm, title, int(freq), len(tokens), tuple(dict.fromkeys(tokens))))

        entries.sort(key=lambda x: (-x[3], -x[2], -len(x[0])))

        for i, (t_norm, t_orig, freq, tok_cnt, uniq_tokens) in enumerate(entries[: self.max_lexicon_titles]):
            self.lex_norm.append(t_norm)
            self.lex_orig.append(t_orig)
            self.lex_freq.append(freq)
            self.lex_tok_cnt.append(tok_cnt)
            for tok in uniq_tokens:
                self.lex_tok_index[tok].append(i)

    def _prefer_svc(self, query_norm: str, title: str, lr_label: str, svc_label: str) -> bool:
        if svc_label == lr_label:
            return False

        q_tokens = set(self._tokenize(query_norm))
        has_anim = "мульт" in query_norm or bool(q_tokens & {"аниме", "мультфильм", "мультсериал"})
        has_serial = bool(q_tokens & self.SERIAL_WORDS)
        has_show = bool(q_tokens & self.SHOW_WORDS)
        has_film = bool(q_tokens & self.FILM_WORDS)

        if has_anim:
            return svc_label in {"мультфильм", "мультсериал"}
        if has_show and not title:
            return svc_label in {"прочее", ""}
        if has_serial and not has_film:
            return svc_label in {"сериал", "мультсериал"}

        return (not title) and (svc_label in {"прочее", ""})

    def _apply_content_rules(self, query_norm: str, label: str) -> str:
        q_tokens = set(self._tokenize(query_norm))

        has_anim = "мульт" in query_norm
        has_anime = "аниме" in q_tokens
        has_dorama = bool(q_tokens & {"дорама", "дорамы", "kdrama", "k-drama"})
        has_serial = bool(q_tokens & self.SERIAL_WORDS)
        has_film = bool(q_tokens & self.FILM_WORDS)

        if has_anime:
            return "мультфильм" if has_film and not has_serial else "мультсериал"
        if has_anim and has_serial:
            return "мультсериал"
        if has_anim and label in {"", "прочее", "сериал", "фильм"}:
            return "мультфильм"
        if has_dorama and not has_film:
            return "сериал"
        if has_serial and not has_film and label == "фильм":
            return "сериал"
        if has_film and not has_serial and label == "сериал":
            return "фильм"

        return label

    def _build_knn_index(self) -> None:
        self.knn_vectorizer = None
        self.knn_index = None

        if not getattr(self, "knn_query_norm", None):
            return

        corpus = [self._focus_norm(qn) or qn for qn in self.knn_query_norm]
        self.knn_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
        matrix = self.knn_vectorizer.fit_transform(corpus)

        n_neighbors = min(self.knn_neighbors, matrix.shape[0])
        if n_neighbors > 0:
            self.knn_index = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors).fit(matrix)

    def _knn_batch(self, query_norm: list[str]) -> tuple[list[str], list[str], np.ndarray]:
        n = len(query_norm)
        if n == 0:
            return [], [], np.zeros(0, dtype=np.float32)

        if self.knn_index is None or self.knn_vectorizer is None:
            return [""] * n, [""] * n, np.zeros(n, dtype=np.float32)

        q_matrix = self.knn_vectorizer.transform([self._focus_norm(qn) or qn for qn in query_norm])
        dists, idxs = self.knn_index.kneighbors(q_matrix, return_distance=True)

        out_titles = [""] * n
        out_content = [""] * n
        out_sim = np.zeros(n, dtype=np.float32)

        for i, (dist_row, idx_row) in enumerate(zip(dists, idxs)):
            t_scores: dict[str, float] = defaultdict(float)
            c_scores: dict[str, float] = defaultdict(float)
            best = 0.0

            for d, j in zip(dist_row, idx_row):
                sim = float(1.0 - d)
                if sim <= 0.0:
                    continue
                best = max(best, sim)
                if self.knn_titles[j]:
                    t_scores[self.knn_titles[j]] += sim * sim
                if self.knn_contents[j]:
                    c_scores[self.knn_contents[j]] += sim

            out_sim[i] = best
            if t_scores:
                out_titles[i] = max(t_scores.items(), key=lambda x: (x[1], len(x[0])))[0]
            if c_scores:
                out_content[i] = max(c_scores.items(), key=lambda x: x[1])[0]

        return out_titles, out_content, out_sim

    def _focus_norm(self, text_norm: str) -> str:
        return " ".join(
            w for w in self._tokenize(text_norm)
            if len(w) > 1 and w not in self.NOISE_WORDS and w not in self.GENERIC_WORDS
        )

    def _token_overlap_ratio(self, a_norm: str, b_norm: str) -> float:
        a_tokens = set(self._tokenize(a_norm))
        b_tokens = set(self._tokenize(b_norm))
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / float(min(len(a_tokens), len(b_tokens)))

    def _title_token_ratio_max(self, title_norm: str) -> float:
        tokens = set(self._tokenize(title_norm))
        if not tokens:
            return 0.0
        return max(self.title_token_ratio.get(tok, 1.20) for tok in tokens)

    @classmethod
    def _has_latin(cls, text_norm: str) -> bool:
        return bool(cls.LATIN_RE.search(text_norm))

    def _lex_match(self, query_norm: str) -> str:
        q_tokens = set(self._tokenize(query_norm))
        cand: set[int] = set()

        for tok in q_tokens:
            cand.update(self.lex_tok_index.get(tok, []))

        if not cand:
            return ""

        padded = f" {query_norm} "
        best_idx = -1
        best_score = -1

        for i in cand:
            if f" {self.lex_norm[i]} " not in padded:
                continue
            score = self.lex_tok_cnt[i] * self.lex_token_weight + min(self.lex_freq[i], 20)
            if score > best_score:
                best_idx, best_score = i, score

        return "" if best_idx < 0 else self.lex_orig[best_idx]

    def _product_type_extract(self, text: str) -> str:
        out: list[str] = []
        for w in self._clean_title_aggressive(text).split():
            if len(w) <= 1 or w in self.NOISE_WORDS or w in self.GENERIC_WORDS or w in self.TITLE_NOISE:
                continue
            out.append(w)
            if len(out) >= self.max_heuristic_tokens:
                break
        return " ".join(out)

    def _has_force_video_words(self, query_norm: str) -> bool:
        tokens = set(self._tokenize(query_norm))
        if not tokens:
            return False
        if tokens & self.STREAMING_WORDS:
            return True
        if "vk" in tokens and "видео" in tokens:
            return True
        if "кино" in tokens and tokens & self.VIDEO_NEGATIVE_WORDS:
            tokens = set(tokens)
            tokens.discard("кино")
        return bool(tokens & self.FORCE_VIDEO_WORDS)

    def _clean_title_aggressive(self, text: str) -> str:
        t = self._norm(text)
        t = re.sub(r"\d+[\.,]?\d*\s*(см|мм|м|мл|л|г|кг|шт|вт|в|а|штук|пар|x|х|°|%|d|cm|mm|ml|kg|w|v|pcs|pc|m|g)\b", " ", t)
        t = re.sub(r"\d+[\.,]?\d*", " ", t)
        t = re.sub(r"\b(xs|s|m|l|xl|xxl|xxxl)\b", " ", t)

        for c in self.SIZE_COLOR_WORDS:
            t = re.sub(r"\b" + re.escape(c) + r"\b", " ", t)

        return re.sub(r"\s+", " ", t).strip()

    def _fallback_type(self, query: str) -> int:
        q_norm = self._norm(query)
        q_tokens = set(self._tokenize(q_norm))
        return int(self._has_force_video_words(q_norm) or bool(q_tokens & self.FALLBACK_VIDEO_WORDS))

    def _fallback_content(self, query: str, type_pred: int) -> str:
        if type_pred == 0:
            return ""

        q = self._norm(query)
        if "мульт" in q and ("сериал" in q or "сезон" in q):
            return "мультсериал"
        if "мульт" in q:
            return "мультфильм"
        if "сериал" in q or "сезон" in q or "серия" in q:
            return "сериал"
        if "фильм" in q or "кино" in q:
            return "фильм"
        return "прочее"

    def _save_artifact(self, path: Path) -> None:
        payload = {k: getattr(self, k) for k in self.ARTIFACT_KEYS}
        payload["lex_tok_index"] = dict(self.lex_tok_index)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_artifact(self, path: Path) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)

        for key in self.ARTIFACT_KEYS:
            if key in payload:
                setattr(self, key, payload[key])

        self.lex_tok_index = defaultdict(list, payload.get("lex_tok_index", {}))
        self._build_knn_index()

    def _load_train_data(self) -> pd.DataFrame | None:
        candidates: list[Path] = []

        if os.environ.get("MEDIASCOPE_TRAIN_PATH"):
            candidates.append(Path(os.environ["MEDIASCOPE_TRAIN_PATH"]))

        base = Path(__file__).resolve().parent
        cwd = Path.cwd()
        candidates += [
            base / "data" / "train.csv",
            cwd / "data" / "train.csv",
            base / "train.csv",
            cwd / "train.csv",
            base / "train_cache.csv",
            cwd / "train_cache.csv",
        ]

        for p in candidates:
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                if {"QueryText", "TypeQuery", "Title", "ContentType"}.issubset(df.columns):
                    return df
            except Exception:
                pass

        return None

    @classmethod
    def _norm(cls, s: str) -> str:
        s = str(s).lower().replace("ё", "е")
        s = re.sub(r"[^0-9a-zа-я]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def _tokenize(cls, s: str) -> list[str]:
        return cls.TOKEN_RE.findall(s)
