# DomainML

![Tests](https://img.shields.io/badge/tests-65%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Domain-knowledge integrated machine learning library for scikit-learn.**

DomainML は、「ドメインエキスパートが提供する事前知識（単調性制約・特徴量群・多様体仮説）」を scikit-learn パイプラインに直接統合し、高精度かつ解釈性の高い予測を実現するための拡張ライブラリです。

---

## Features (v0.3.0)

### 1. ドメイン知識の型安全な定義

```python
from domainml import FeatureMetadata, ConstraintStrength, MonotonicityDirection

meta = FeatureMetadata(
    feature_names=["temperature", "pressure", "flow_rate"],
    monotonicities=["inc", "dec", "none"],
    constraint_types=["strict", "soft", "none"],
    groups=[0, 0, -1],           # temperature と pressure は同一グループ
    control_flags=[False, False, True],  # flow_rate は実験制御変数
    extrapolation_sigma=[2.0, 2.0, 3.0]  # 特徴量ごとの外挿範囲
)
```

- **`ConstraintStrength`**: `STRICT` / `SOFT` / `PREFERENCE` の 3 段階列挙型
- **`MonotonicityDirection`**: `INCREASING` / `DECREASING` / `NONE` の方向列挙型

### 2. 制約エンジン（自動フォールバック）

```python
from domainml import MonotonicityEngine

engine = MonotonicityEngine(HistGradientBoostingRegressor())
engine.fit(X, y, metadata=meta)
```

- 決定木系: ネイティブ `monotone_constraints` / `monotonic_cst` を自動選択
- 線形モデル: **CVXPY** による厳密制約最適化 (`MonotonicLinearRegression`)
- カーネル法: 仮想制約点での勾配制約 (`KernelMonotonicity`)
- その他: `IsotonicRegression` による事後補正 (`MonotonicityWrapper`)

### 3. 特徴量群制約（Group Lasso）

```python
from domainml import GroupConstraintEngine, GroupStandardScaler

scaler = GroupStandardScaler(groups={0: [0, 1], 1: [2, 3]})
X_scaled = scaler.fit_transform(X)  # グループ全体を同一尺度でスケール

engine = GroupConstraintEngine(groups={0: [0, 1]})
coef_regularized = engine.apply_group_lasso_penalty(coef, lambda_group=0.5)
```

### 4. 多様体仮説エンジン

```python
from domainml import ManifoldAssumptionEngine

manifold = ManifoldAssumptionEngine(
    manifold_variables=[0, 1],     # 観測変数（多様体構造を持つ）
    non_manifold_variables=[2],    # 制御変数（除外）
)
L = manifold.build_laplacian_regularization(X)
```

### 5. 制約競合検出

```python
from domainml import LinearCoefConflictChecker

detector = LinearCoefConflictChecker(correlation_threshold=0.8)
conflicts = detector.detect_conflicts(X, y, meta)
# 3種類の競合を検出:
# - coef_monotonicity_conflict (係数とドメイン制約の逆行)
# - statistical_monotonicity_conflict (高相関特徴量の相反制約)
# - mathematical_infeasibility (線形計画法による充足不能検出)
```

### 6. 制約充足度評価と交差検証

```python
from domainml import satisfaction_score, constrained_cv

score = satisfaction_score(model, X_val, meta)   # 0.0 〜 1.0
results = constrained_cv(engine, X, y, meta, cv=5)
```

---

## Quick Start

```bash
pip install domainml-meta
```

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from domainml import FeatureMetadata, MonotonicityEngine, satisfaction_score

X = np.random.rand(100, 2)
y = 3 * X[:, 0] - 2 * X[:, 1]

meta = FeatureMetadata(
    feature_names=["f1", "f2"],
    monotonicities=["inc", "dec"],
    constraint_types=["strict", "strict"]
)

engine = MonotonicityEngine(HistGradientBoostingRegressor())
engine.fit(X, y, metadata=meta)

score = satisfaction_score(engine, X, meta)
print(f"制約充足度: {score:.3f}")  # → 1.0 に近い値
```

---

## Architecture

```
FeatureMetadata (ドメイン知識の器)
       ↓
MetaPipeline (メタデータ伝播パイプライン)
       ↓
MonotonicityEngine (自動フォールバック制約エンジン)
  ├── 線形モデル → MonotonicLinearRegression (CVXPY)
  ├── 決定木    → native monotone_constraints
  ├── カーネル  → KernelMonotonicity (仮想制約点)
  └── その他    → MonotonicityWrapper (IsotonicRegression)
       ↓
satisfaction_score / constrained_cv (評価・交差検証)
```

## Roadmap (Phase 4 予定)

- [ ] GPR（ガウス過程回帰）への単調性制約統合
- [ ] 複数特徴量間の相互作用制約
- [ ] 制約付き AutoML インターフェース
