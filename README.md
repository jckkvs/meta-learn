# DomainML

![Coverage Status](https://img.shields.io/badge/coverage-95%25-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Domain-knowledge integrated machine learning module for scikit-learn.**

DomainML は、自動抽出・推測に頼らず「ドメインエキスパートが提供する事前知識」を機械学習モデルに統合し、高精度かつ解釈性の高い予測を実現するための拡張ライブラリです。

## Features (v0.2.0 - Phase 1 (MVP) 完了)

1. **Scikit-learn 互換の制約エンジン**
   - **MonotonicityEngine**: 決定木、線形モデル、カーネル法、および一般モデルに対して、最適な単調性制約を自動的に判別し適用します。
   - **MonotonicityWrapper**: IsotonicRegression による事後補正ラッパー

2. **メタデータ管理**
   - **FeatureMetadata**: ドメイン知識を保持し伝播させるためのメタデータクラス

3. **制約充足度評価**
   - **satisfaction_score**: モデルの予測結果が単調性制約をどの程度満たしているかを評価

## Roadmap (Phase 2 予定)

- [ ] SparseLaplacian (多様体仮説の統合)
- [ ] CausalConflictDetector (制約競合検出)
- [ ] constrained_cv (制約付き交差検証)

## Architecture & Design Philosophy

本ライブラリは、MVP（最小実行可能プロダクト）フェーズをクリアし、柔軟で拡張性の高いアーキテクチャへと進化しました。
- **自動フォールバック機能**: 指定されたアルゴリズムがネイティブに制約をサポートしていない場合、自動的にラッパーベースや CVXPY ベースの制約機構に推移します。
- **ドメイン知識の明示的定義**: データ駆動型の推定を排除し、専門家の知見をそのままコードに `FeatureMetadata` 経由で注入できる設計となっています。
- **堅牢な品質基準**: 複数環境での自動テストにより **行および分岐カバレッジ 95% 以上** を達成。エッジケース処理の異常系テストも徹底しています。

## Quick Start
```bash
pip install domainml-meta
```

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from domainml.core.metadata import FeatureMetadata
from domainml.constraints.engine import MonotonicityEngine
from domainml.core.pipeline import MetaPipeline
from domainml.analysis.metrics import satisfaction_score

# 1. データの用意
X = np.random.rand(100, 2)
y = 3 * X[:, 0] - 2 * X[:, 1]  # f1: 正の相関, f2: 負の相関

# 2. メタデータ（ドメイン知識）の定義
# f1 は「単調増加」、f2 は「単調減少」であるべきという専門知識
meta = FeatureMetadata(
    feature_names=["f1", "f2"],
    monotonicities=["inc", "dec"],
    constraint_types=["strict", "strict"]
)

# 3. 制約エンジンの初期化（モデルをラップするだけ）
engine = MonotonicityEngine(HistGradientBoostingRegressor())

# 4. メタパイプライン内でメタデータを添えて学習
pipeline = MetaPipeline([
    ('model', engine)
])
pipeline.fit(X, y, metadata=meta)

# 5. 再現率（制約充足度）の確認 -> ほぼ 1.0 になる
score = satisfaction_score(pipeline, X, meta)
print(f"Domain Constraint Satisfaction Score: {score:.3f}")
```

---
*詳細なAPIドキュメントや高度な多様体仮説 (Manifold Assumption) の利用例については、今後のアップデートをお待ちください。*
