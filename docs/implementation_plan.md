# Goal Description
scikit-learn では表現できなかった「ドメイン知識（単調性制約、多様体仮説、特徴量群等）」を明示的に統合できる機械学習ライブラリ「DomainML」を開発する。`FeatureMetadata` オブジェクトを用いた制約の保持・伝播、`MetaPipeline` による前処理への追従、そして最適化ソルバーやカスタム勾配降下法を用いた `DomainEstimator` により、外挿領域(±x シグマ)を含めた強力な制約付き学習を可能にする。

## User Review Required
> [!IMPORTANT]
> 以下の事項について、ユーザーによるレビューと合意が必要です。
> 1. **全モデルへの厳密な単調性制約の適用難易度**: KNNやRandomForest等、勾配法や数理最適化で定式化しにくいモデルに対しては「Isotonic Regression等による事後補正（ラッパー）」アプローチをデフォルトとする方針でよろしいでしょうか。
> 2. **最適化バックエンド**: 厳密制約（Strict）には `cvxpy` や `scipy.optimize` を、軟制約（Soft）や多変量では `PyTorch` などの勾配法ライブラリを部分的に導入する検討がありますが、依存ライブラリ増加の許容範囲はいかがでしょうか（まずは scipy/cvxpy で完結させる方針を推奨します）。

## Proposed Changes

### Core Package: `domainml.core`
#### [NEW] `domainml/core/metadata.py`
- `FeatureMetadata` クラスの実装。特徴量名、制約タイプ（strict/soft/なし）、単調性（増/減）、特徴量群ID、多様体フラグ、制御フラグ、外挿範囲を管理するデータ構造。

#### [NEW] `domainml/core/pipeline.py`
- `MetaPipeline` および `MetaColumnTransformer` クラス。scikit-learn互換のインターフェースを持ちながら、`fit_transform` および `transform` 時に `metadata` を受け渡し、次元削減やスケーリングに追随してメタデータを更新・伝播させるロジック。

### Models & Estimators: `domainml.models`
#### [NEW] `domainml/models/base.py`
- `DomainEstimator` 基底クラス。`fit(X, y, metadata=None)` を実装し、渡されたメタデータに基づき内部の損失最適化器に制約構成情報を引き渡すインターフェース。

### Constraints & Regularization: `domainml.constraints`
#### [NEW] `domainml/constraints/monotonicity.py`
- 単調性制約エンジン。
- `cvxpy` を用いた厳密（strict）な単調性制約最適化（GLM等に適用）。
- 損失関数ペナルティを用いた軟（soft）単調性制約定式化。
- 外挿範囲（$\pm x \sigma$）に対する制約補外ロジックの実装。

#### [NEW] `domainml/constraints/manifold.py`
- 多様体仮説統合モジュール。
- Laplacian Regularized Least Squares (LapRLS) などのグラフ正則化項を算出し、メタデータで「制御変数」とされているものをグラフ構築から除外するロジック。

### Analysis & Validation: `domainml.analysis`
#### [NEW] `domainml/analysis/conflict_detector.py`
- 制約競合検出器（`ConstraintConflictDetector`）。
- 学習実行前に、特徴量間の相関行列と単調性制約を比較し、矛盾（強い正相関があるのに片方は単調増加、片方は単調減少が指定されている等）を検知して警告を発する解析器。

## Verification Plan

### Automated Tests
- **Unit Tests (`tests/unit/`)**:
  - `metadata.py`: `FeatureMetadata`のコピー、更新、スライシングの正確性。
  - `pipeline.py`: `MetaPipeline`を通したメタデータの伝播テスト。
  - `monotonicity.py`: ダミーデータに対するStrict/Soft単調性制約の挙動と、外挿（$\pm 3 \sigma$）での制約順守確認。
  - `conflict_detector.py`: 意図的に相反する制約を与えた際のWarning発生テスト。
- **カバレッジと堅牢性**:
  - `pytest --cov=domainml --cov-branch --cov-report=html` にて分岐カバレッジ75%以上を保証。
  - `mutmut run` によりMutation Score 60%以上を達成（CI/ローカルにて実施）。

### Manual Verification
- サンプルスクリプトを用いた動作確認。
- `scikit-learn` のネイティブ機能との互換性（例: メタデータ無しでも正しくフォールバック動作するか）の確認。
