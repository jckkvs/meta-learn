# DomainML

![Tests](https://img.shields.io/badge/tests-113%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Domain-knowledge integrated machine learning library for scikit-learn.**

DomainML は、「ドメインエキスパートが提供する事前知識（単調性制約・特徴量群・多様体仮説）」を scikit-learn パイプラインに直接統合し、高精度かつ解釈性の高い予測を実現するための拡張ライブラリです。

---

## Install

```bash
# 基本インストール
pip install domainml-meta

# CVXPY (線形モデルの厳密制約最適化) を含む場合
pip install domainml-meta[convex]
```

---

## Features (v0.3.0)

### 1. ドメイン知識の型安全な定義

```python
from domainml import FeatureMetadata, ConstraintStrength, MonotonicityDirection, ManifoldConfig

meta = FeatureMetadata(
    feature_names=["temperature", "pressure", "flow_rate"],
    monotonicities=["inc", "dec", "none"],
    constraint_types=["strict", "soft", "none"],
    groups=[0, 0, -1],
    control_flags=[False, False, True],
    extrapolation_sigma=[2.0, 2.0, 3.0],
)

# グループ別多様体仮説を登録（アプローチ4）
meta.update_group_manifold("physical", [0, 1], intrinsic_dim=1)
```

### 2. 制約エンジン（自動フォールバック）

```python
from domainml import MonotonicityEngine
from sklearn.ensemble import HistGradientBoostingRegressor

engine = MonotonicityEngine(HistGradientBoostingRegressor())
engine.fit(X, y, metadata=meta)
```

| モデル種別 | 制約方式 |
|---|---|
| 決定木系 | ネイティブ `monotone_constraints` を自動選択 |
| 線形モデル | CVXPY による厳密制約最適化 |
| カーネル法 | 仮想制約点での勾配制約 |
| その他 | IsotonicRegression による事後補正 |

### 3. 特徴量群制約（Group Lasso）

```python
from domainml import GroupConstraintEngine, GroupStandardScaler

scaler = GroupStandardScaler(groups={0: [0, 1], 1: [2, 3]})
X_scaled = scaler.fit_transform(X)

engine = GroupConstraintEngine(groups={0: [0, 1]})
coef_reg = engine.apply_group_lasso_penalty(coef, lambda_group=0.5)
```

### 4. 多様体仮説の4アプローチ

#### アプローチ1: 前処理（次元削減）

```python
from domainml import ManifoldPreprocessor

prep = ManifoldPreprocessor(n_components=5, method="lle", append=True)
X_aug = prep.fit_transform(X)  # 元特徴量 + 低次元埋め込みを連結
```

#### アプローチ2: 正則化（Laplacian）

```python
from domainml import ManifoldRegularizer

reg = ManifoldRegularizer(manifold_config={"n_neighbors": 10, "regularization_weight": 0.1})
reg.fit(X)
# CVXPY 最適化の損失関数に: loss + reg.get_regularization_term(f)
```

#### アプローチ3: 多様体距離カーネル

```python
from domainml.constraints.manifold_kernel import ManifoldAwareKernel
from sklearn.svm import SVR

kernel = ManifoldAwareKernel(method="diffusion", n_neighbors=10)
kernel.fit(X_train)
K_train = kernel(X_train, X_train)  # 多様体距離に基づくカーネル行列
svr = SVR(kernel="precomputed").fit(K_train, y_train)
```

#### アプローチ4: グループ別多様体（階層的）

```python
meta.update_group_manifold("electronic", [0, 1, 2], intrinsic_dim=2)
meta.update_group_manifold("steric",     [3, 4],    intrinsic_dim=1)
# → 各グループの manifold_flags が自動で True に設定される
```

### 5. 制約競合検出

```python
from domainml import LinearCoefConflictChecker

detector = LinearCoefConflictChecker(correlation_threshold=0.8)
conflicts = detector.detect_conflicts(X, y, meta)
# 3 種類の競合を返す:
#   coef_monotonicity_conflict / statistical_monotonicity_conflict / mathematical_infeasibility
```

### 6. 診断・評価ツール

```python
from domainml import satisfaction_score, constrained_cv
from domainml.analysis.diagnostics import plot_manifold_projection

score = satisfaction_score(model, X_val, meta)       # 制約充足度 [0, 1]
fig   = plot_manifold_projection(X, metadata=meta,   # 多様体を2D可視化
                                  target_values=y, method="isomap")
results = constrained_cv(engine, X, y, meta, cv=5)   # 制約付き交差検証
```

---

## Architecture

```
FeatureMetadata (ドメイン知識 + ManifoldConfig)
       ↓
MetaPipeline (メタデータ伝播)
  ├── [オプション] ManifoldPreprocessor (次元削減前処理)
  ├── [オプション] ManifoldRegularizer  (Laplacian 正則化)
  └── MonotonicityEngine (自動フォールバック制約エンジン)
       ├── 線形 → MonotonicLinearRegression (CVXPY)
       ├── ツリー → native monotone_constraints
       ├── カーネル → KernelMonotonicity / ManifoldAwareKernel
       └── その他 → MonotonicityWrapper (IsotonicRegression)
       ↓
satisfaction_score / constrained_cv / plot_manifold_projection
```

---

## Roadmap

- [ ] GPR（ガウス過程回帰）への単調性制約統合
- [ ] 複数特徴量間の相互作用制約
- [ ] 制約付き AutoML インターフェース
- [ ] 化学データ特化: Tanimoto 距離カーネルの組み込み
