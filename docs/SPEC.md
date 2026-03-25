# DomainML Specifications (SPEC.md)

本仕様書は、Definition of Done (DoD) に基づき、提供が求められた5項目を網羅しています。

## 1. パッケージ構成図 (Directory Structure)

```text
domainml/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── metadata.py        # FeatureMetadata 定義
│   └── pipeline.py        # MetaPipeline, MetaColumnTransformer 定義
├── models/
│   ├── __init__.py
│   ├── base.py            # DomainEstimator 定義
│   └── wrappers.py        # 未対応モデル用事後補正ラッパー (Isotonic等)
├── constraints/
│   ├── __init__.py
│   ├── monotonicity.py    # Strict/Soft 単調性制約, Extrapolation 処理
│   └── manifold.py        # グラフ正則化 (Laplacian)
├── analysis/
│   ├── __init__.py
│   └── conflict_detector.py # 制約競合の事前検知機能
└── meta/
    ├── __init__.py
    └── recommender.py     # (Phase 2) メタ制約推薦ロジック
```

## 2. 主要クラス定義とシグネチャ

### `FeatureMetadata`
```python
from typing import List, Optional, Literal

class FeatureMetadata:
    def __init__(self, 
                 feature_names: List[str],
                 monotonicities: Optional[List[Literal['inc', 'dec', 'none']]] = None,
                 constraint_types: Optional[List[Literal['strict', 'soft', 'none']]] = None,
                 groups: Optional[List[int]] = None,
                 manifold_flags: Optional[List[bool]] = None,
                 control_flags: Optional[List[bool]] = None,
                 extrapolation_sigma: float = 3.0):
        # 変数初期化処理...
        
    def slice(self, indices: List[int]) -> 'FeatureMetadata': ...
    def merge(self, other: 'FeatureMetadata') -> 'FeatureMetadata': ...
```

### `MetaPipeline`
```python
from sklearn.pipeline import Pipeline

class MetaPipeline(Pipeline):
    def fit(self, X, y=None, metadata: Optional['FeatureMetadata'] = None, **fit_params): ...
    def transform(self, X, metadata: Optional['FeatureMetadata'] = None): ...
    def fit_transform(self, X, y=None, metadata: Optional['FeatureMetadata'] = None, **fit_params): ...
```

### `DomainEstimator`
```python
from sklearn.base import BaseEstimator

class DomainEstimator(BaseEstimator):
    def fit(self, X, y, metadata: Optional['FeatureMetadata'] = None) -> 'DomainEstimator': ...
    def predict(self, X) -> np.ndarray: ...
    def get_constraints_from_metadata(self, metadata: 'FeatureMetadata'): ...
```

## 3. 核心実装コード (Draft API Concept)

### 単調性制約ミックスインとメタデータ伝播
```python
class MonotonicityMixin:
    def apply_strict_monotonicity(self, X, y, metadata: FeatureMetadata):
        import cvxpy as cp
        # CVXPYによる勾配制約付き最適化の例 (線形/一般化線形モデル向け)
        n_features = X.shape[1]
        w = cp.Variable(n_features)
        b = cp.Variable()
        
        loss = cp.sum_squares(X @ w + b - y)
        constraints = []
        for i, val in enumerate(metadata.monotonicities):
            if val == 'inc' and metadata.constraint_types[i] == 'strict':
                constraints.append(w[i] >= 0)
            elif val == 'dec' and metadata.constraint_types[i] == 'strict':
                constraints.append(w[i] <= 0)
        
        prob = cp.Problem(cp.Minimize(loss), constraints)
        prob.solve()
        return w.value, b.value

    def _extrapolate_constraints(self, X, metadata: FeatureMetadata):
        # データの平均と標準偏差を計算し、± x sigma の仮想点(Support Points)を生成。
        # 予測関数がその領域でも単調性を満たすように損失関数へペナルティを加算するロジック
        pass
```

## 4. 使用例 (Usage Example)

```python
import numpy as np
import pandas as pd
from domainml.core.metadata import FeatureMetadata
from domainml.core.pipeline import MetaPipeline
from domainml.models.base import DomainEstimator

# 1. データとメタデータの準備
X_df = pd.DataFrame({'price': [100, 200, 300], 'age': [5, 3, 1]})
y = np.array([50, 40, 30])
metadata = FeatureMetadata(
    feature_names=['price', 'age'],
    monotonicities=['none', 'dec'],      # ageが上がるほど目的関数は下がる
    constraint_types=['none', 'strict'], # 厳密な制約
    extrapolation_sigma=5.0              # ±5σまで制約順守
)

# 2. パイプラインの構築
pipeline = MetaPipeline([
    ('estimator', DomainEstimator()) # メタデータを受け取って制約付き最適化を実行
])

# 3. 学習と推論
pipeline.fit(X_df, y, metadata=metadata)
preds = pipeline.predict(X_df)
```

## 5. 技術的課題と解決策

1. **計算コスト (strict 制約と CVXPY のスケーラビリティ)**
   - **課題**: `strict` 制約をかける場合、各点のヘシアンやヤコビアンを計算する最適化ソルバーが必要となり、大規模データではメモリと計算時間がボトルネックになる。
   - **解決策**:
     サンプル数が多い場合は Stocastic Gradient Descent (SGD) での `soft` 制約（ペナルティ項）や、射影勾配法（Projected Gradient Descent）へ自動フォールバックするロジックを搭載する。
2. **メタデータと変換器（Transformer）の整合性**
   - **課題**: PCA や PolynomialFeatures などの scikit-learn 手法を通すと、元の特徴量の意味が消失し、単調性などのメタデータが適用できなくなる。
   - **解決策**:
     `MetaColumnTransformer` により、変換パスと非変換パスを分岐させ、線形結合であれば係数を伝播（例：係数が正なら単調性も維持）させる機能を一部導入。非線形変換される変数群については、メタデータの `monotonicity` フラグを `none` にフォールバックし、Warning を出力する。
3. **外挿(±xシグマ)領域での制約維持と過学習**
   - **課題**: 外挿領域での単調性を微分可能に保つには、学習データに含まれない大量のサポートポイントで損失を評価する必要があり、バイアスを生じさせる。
   - **解決策**:
     サポートポイントにおけるペナルティ項（Soft Constraint）の重みを外挿距離（$\sigma$値）に応じて指数関数的に崩壊させることで、過剰な正規化を防ぐ設計とする。
