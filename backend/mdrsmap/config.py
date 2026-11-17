"""演算法的設定值,集中在一處,預設為論文的值。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """演算法的設定值,預設為論文的值。

    調小這些值可以讓 Search 快很多,測試就是這麼做的;調整後的結果不再對應論文。
    """

    top_views: int = 100
    """多視角隨機搜尋後,保留幾個 rho 最高的視角來算加權距離。"""

    max_embedding_dimension: int = 10
    """CCM 特徵選擇時,每個特徵最多試到幾維的嵌入。"""

    max_lag: int = 10
    """每個特徵展開成幾個延遲欄位,包含當天。"""

    thetas: tuple[float, ...] = (1, 2, 4, 7, 11, 16, 22)
    """S-map 加權距離的 theta 候選值。"""

    validation_days: int = 60
    """Library 最後幾個交易日當作驗證區間,用來挑嵌入維度、視角與模型參數。"""

    random_search_trials: int = 10_000
    """多視角隨機搜尋總共抽幾次視角。這一步佔了 Search 大部分的時間。"""
