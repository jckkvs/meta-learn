# DomainML Project Task Checklist

## DoD (Definition of Done)
- [ ] 星取表（MUST）＝ 100% 実装＋テストID対応
- [ ] 分岐カバレッジ≥75%、テスト完備
- [ ] Mutationスコア≥60%
- [ ] pass/NotImplementedError/TODO/ダミー戻り値の排除
- [ ] ドキュメントとテストの紐付け対応表作成
- [ ] 再現性環境 (conda environment.yml) の作成
- [ ] 自己監査の合否判定 (REPORT.md) 作成

## 1. 計画フェーズ (Planning)
- [x] プロジェクト構成とアーキテクチャ設計 (implementation_plan.md)
- [x] FeatureMatrix.md (星取表) の作成
- [x] 仕様書 (SPEC.md) の作成

## 2. 実装フェーズ (Execution)
- [x] パッケージディレクトリ構成の初期化
- [x] モジュール `metadata.py`: `FeatureMetadata` クラスの実装
- [x] モジュール `pipeline.py`: `MetaPipeline`クラスの実装
- [x] モジュール `estimator.py`: `DomainEstimator` の基底クラス実装
- [x] 制約エンジン `monotonicity.py`: Strict/Soft 単調性制約と外挿対応
- [x] 多様体仮説統合 `manifold.py`: グラフ正則化ミックスイン
- [x] 多様体仮説統合 `manifold.py`: グラフ正則化ミックスイン
- [x] 競合検出器 `conflict_detector.py`
- [x] README等の使用例やサンプルスクリプト作成

## 3. 検証フェーズ (Verification)
- [x] 再現環境の構築 (conda/pixi) とテストランナーの設定
- [x] ユニットテストおよび統合テスト (pytest) 実装
- [x] カバレッジ (pytest-cov) カバレッジ ≥ 75% の計測・達成
- [x] Mutation テストの実行・スコア達成 (mutmut等)
- [x] ベンチマーク・動作エビデンスの出力
- [x] 自己監査チェック (REPORT.md)

## 4. デバッグ環境構築 (Debugging)
- [x] `logger.py` の実装と `domainml` 全体への統合
- [x] `*.log` の `.gitignore`, `MANIFEST.in` 除外設定
- [x] デバッグ情報の生成確認
