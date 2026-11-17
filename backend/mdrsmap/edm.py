"""MDR S-map 的各個建模步驟,由 classifier 依序呼叫。

步驟依序是目標嵌入維度、CCM 特徵選擇、多視角隨機搜尋、S-map 加權距離與 logistic 模型。
每個步驟吃的表由 classifier 準備:第一欄是 pyEDM 要求的時間欄,
其餘欄位已做 min-max 正規化,最後一列是要預測的那一天。
pyEDM 以空白分隔欄位名,所以欄位名不能含空白。
"""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
import pandas as pd
from pyEDM import CCM, Embed, EmbedDimension, Simplex
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit

TIME = "Time"

# 論文固定的亂數種子,用在多視角隨機搜尋。saga 每次 fit 也用同一個種子,
# 同一天的模型才不會因為前面算過幾天而不同
RANDOM_SEED = 87

_LOGISTIC_GRID = {"l1_ratio": [0.9, 0.7, 0.5, 0.3, 0.1], "C": [100, 50, 10, 1, 0.1]}


class View(NamedTuple):
    """多視角隨機搜尋找到的一個視角。

    columns 是幾個延遲欄位的組合;rho 是用它做 Simplex 時,預測與實際的相關係數。
    """

    rho: float
    columns: tuple[str, ...]


class Logistic(NamedTuple):
    """選定的 S-map theta 與彈性網路 logistic 的參數。"""

    theta: float
    c: float
    l1_ratio: float


def target_embedding_dimension(
    data: pd.DataFrame, target: str, validation_days: int
) -> int:
    """回傳 target 的最佳嵌入維度:驗證區間上 Simplex 預測 rho 最高的那一維。"""
    n = len(data)
    result = EmbedDimension(
        dataFrame=data,
        columns=target,
        lib=f"1 {n}",
        pred=f"{n - validation_days - 1} {n - 1}",
        showPlot=False,
    )
    return int(result["E"][result["rho"] == result["rho"].max()].iloc[0])


def select_features(
    data: pd.DataFrame, target: str, target_dimension: int, max_dimension: int
) -> list[str]:
    """以 CCM 挑出與 target 有因果關聯的特徵,順序與表的欄位順序相同。

    每個特徵先挑 CCM 最後一點 rho 最高的嵌入維度。在那一維下,
    rho 必須隨 Library 長度單調上升(Kendall tau 檢定),
    而且最後一點的 rho 必須超過臨界值,這個特徵才入選。
    """
    n = len(data)
    t = stats.t.ppf(0.95, n - 1)
    critical_rho = t / (n - 2 + t**2)
    lib_sizes = f"{target_dimension + 10} {n - 10} 10"

    selected = []
    for feature in data.columns[1:]:
        if feature == target:
            continue
        column = f"{target}:{feature}"
        by_dimension = {
            e: CCM(
                dataFrame=data,
                E=e,
                columns=feature,
                target=target,
                libSizes=lib_sizes,
                random=False,
                showPlot=False,
            )
            for e in range(1, max_dimension + 1)
        }
        best = pd.Series(
            {e: result[column].iloc[-1] for e, result in by_dimension.items()}
        ).idxmax()
        ccm = by_dimension[best]
        # 負的 rho 視為沒有關聯,檢定前先截到 0
        rho = ccm[column].clip(lower=0)
        p_value = stats.kendalltau(ccm["LibSize"], rho).pvalue
        if p_value < 0.05 and rho.iloc[-1] > critical_rho:
            selected.append(feature)
    return selected


def embed(
    data: pd.DataFrame, target: str, features: tuple[str, ...], max_lag: int
) -> pd.DataFrame:
    """把特徵與 target 展開成延遲欄位,丟掉延遲不足的開頭幾列。

    回傳的表第一欄是時間欄,其餘欄名形如 `close(t-0)`、`close(t-1)`。
    """
    embedded = Embed(
        dataFrame=data, E=max_lag, tau=-1, columns=" ".join([*features, target])
    )
    embedded.insert(0, TIME, data[TIME])
    return embedded.dropna().reset_index(drop=True)


def random_views(
    embedded: pd.DataFrame,
    target: str,
    target_dimension: int,
    validation_days: int,
    trials: int,
    top: int,
) -> tuple[View, ...]:
    """隨機抽 trials 次延遲欄位的組合,回傳 rho 最高的 top 個。

    每個組合用 Simplex 預測 target 當天的值來算 rho。
    組合的欄位數等於 target 的嵌入維度。抽到重複的組合會重複出現在結果裡。
    """
    rng = np.random.RandomState(RANDOM_SEED)
    current = f"{target}(t-0)"
    candidates = np.array([c for c in embedded.columns[1:] if c != current])
    m = len(embedded)

    views = []
    for _ in range(trials):
        columns = candidates[
            rng.choice(len(candidates), target_dimension, replace=False)
        ]
        result = Simplex(
            dataFrame=embedded,
            columns=" ".join(columns),
            target=current,
            lib=f"1 {m}",
            pred=f"{m - validation_days - 1} {m - 1}",
            embedded=True,
            showPlot=False,
        )
        rho = result["Observations"].corr(result["Predictions"])
        views.append(View(float(rho), tuple(str(c) for c in columns)))

    # stable:rho 相同的視角保留抽到的先後順序,結果才不受排序演算法影響
    ranked = pd.Series([view.rho for view in views]).sort_values(
        ascending=False, kind="stable"
    )
    return tuple(views[i] for i in ranked.index[:top])


def weighted_distances(embedded: pd.DataFrame, views: tuple[View, ...]) -> np.ndarray:
    """回傳列與列之間的距離矩陣:各視角的歐氏距離,以 rho 占比為權重加總。"""
    rho = pd.Series([view.rho for view in views])
    weights = rho / rho.sum()
    distances = np.zeros((len(embedded), len(embedded)))
    for view, weight in zip(views, weights, strict=True):
        distances += squareform(
            pdist(embedded[list(view.columns)].to_numpy(), metric="euclidean") * weight
        )
    return distances


def select_logistic(
    table: pd.DataFrame,
    target: str,
    distances: np.ndarray,
    thetas: tuple[float, ...],
    horizon: int,
    validation_days: int,
) -> Logistic:
    """對每個 theta 以 grid search 挑 logistic 的參數,回傳驗證區間準確率最高的組合。

    table 是只留當天延遲欄位的表。準確率相同時取先試到的。
    """
    best: tuple[float, Logistic] | None = None
    for theta in thetas:
        x, y = _weighted_training_set(table, target, distances, theta, horizon)
        split = PredefinedSplit(
            [-1] * (len(x) - validation_days) + [0] * validation_days
        )
        search = GridSearchCV(_logistic(), _LOGISTIC_GRID, cv=split, scoring="accuracy")
        with _known_sklearn_warnings_ignored():
            search.fit(x, y)
        if best is None or search.best_score_ > best[0]:
            params = search.best_params_
            best = (
                search.best_score_,
                Logistic(theta, params["C"], params["l1_ratio"]),
            )
    return best[1]


def predict(
    table: pd.DataFrame,
    target: str,
    distances: np.ndarray,
    logistic: Logistic,
    horizon: int,
) -> int:
    """以 S-map 加權訓練 logistic 模型,回傳最後一列那一天的 Direction。"""
    x, y = _weighted_training_set(table, target, distances, logistic.theta, horizon)
    with _known_sklearn_warnings_ignored():
        model = _logistic(C=logistic.c, l1_ratio=logistic.l1_ratio).fit(x, y)
    return int(model.predict(table.iloc[[-1]])[0])


def _logistic(**params: float) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        penalty="elasticnet",
        solver="saga",
        fit_intercept=True,
        intercept_scaling=0.1,
        random_state=RANDOM_SEED,
        **params,
    )


@contextmanager
def _known_sklearn_warnings_ignored() -> Iterator[None]:
    """忽略論文的模型設定下,sklearn 每次 fit 都可能發出的兩種警告。

    - `penalty` 的寫法從 sklearn 1.8 起不建議使用;版本已刻意鎖在移除它之前
    - saga 在預設的迭代次數內不一定收斂;迭代次數是論文的設定,改了結果就不同

    sklearn 內部會重置「同一個警告只顯示一次」的紀錄,不忽略的話每次 fit 都會印一次。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="'penalty' was deprecated", category=FutureWarning
        )
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        yield


def _weighted_training_set(
    table: pd.DataFrame,
    target: str,
    distances: np.ndarray,
    theta: float,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """以離最後一列的距離加權每一列,回傳訓練用的特徵與答案。

    第 t 列的答案是 t + horizon 列的 target,也就是第 t 天的 Direction。
    最後 horizon + 1 列不放進訓練資料:最後 horizon 列的答案還沒揭曉,
    多丟的一列沿用論文程式的做法。
    """
    current = f"{target}(t-0)"
    last = len(table) - 1
    to_last = distances[last]
    scale = np.mean(np.delete(to_last, last))
    weight = np.sqrt(np.exp(-theta * to_last / scale))

    weighted = table.copy()
    weighted["answer"] = weighted[current].shift(-horizon)
    weighted = weighted.multiply(weight, axis=0)
    # Direction 乘上權重後不再是 0 或 1,改回二元值
    for column in (current, "answer"):
        weighted[column] = weighted[column].ne(0).astype(float)
    weighted = weighted[: -(horizon + 1)]
    return weighted.drop(columns="answer"), weighted["answer"]
