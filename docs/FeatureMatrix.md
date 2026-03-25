# DomainML Feature Matrix

| ID | 章・節・機能概要 | 機能名 | MUST/OPTIONAL | 実装予定ファイル/クラス | テストID | 性能指標 | 依存 |
|---|---|---|---|---|---|---|---|
| F-001 | 3.1. メタデータ管理 | `FeatureMetadata` 実装 | MUST | `domainml/core/metadata.py` | T-001 | カバレッジ100% | - |
| F-002 | 3.2. メタデータ伝播 | `MetaPipeline` 実装 | MUST | `domainml/core/pipeline.py` | T-002 | 分岐カバレッジ≥75% | F-001 |
| F-003 | 3.3. 拡張推定器 API | `DomainEstimator` 実装 | MUST | `domainml/models/base.py` | T-003 | 分岐カバレッジ≥75% | F-001, F-002 |
| F-004 | 4.1. 単調性エンジン (strict) | `StrictMonotonicityMixin` | MUST | `domainml/constraints/monotonicity.py` | T-004 | ソルバー収束/テストパス | - |
| F-005 | 4.1. 単調性エンジン (soft) | `SoftMonotonicityMixin` | MUST | `domainml/constraints/monotonicity.py` | T-005 | ソルバー収束/テストパス | - |
| F-006 | 4.1. 外挿領域対応 (±xσ) | `ExtrapolationConstraint` | MUST | `domainml/constraints/monotonicity.py` | T-006 | データ外の制約順守確認 | F-004, F-005 |
| F-007 | 4.2. 多様体仮説統合 | `ManifoldIntegration` | MUST | `domainml/constraints/manifold.py` | T-007 | Laplacian制約計算精度 | - |
| F-008 | 4.3. 競合検出器 | `ConstraintConflictDetector` | MUST | `domainml/analysis/conflict_detector.py` | T-008 | 矛盾検出の正確性 | F-001 |
| F-009 | 4.1. 未対応モデル事後補正 | `MonotonicityWrapper` | OPTIONAL | `domainml/models/wrappers.py` | T-009 | Isotonic補正精度 | - |
| F-010 | 3.5. メタ推論/推奨エンジン | `ConstraintRecommender` | OPTIONAL | `domainml/meta/recommender.py` | T-010 | 過去タスクからの推薦精度 | F-001 |

_*※ OPTIONAL 機能については第2フェーズ（またはコア機能完了後）に実装を予定*_
