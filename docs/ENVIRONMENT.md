# 再現手順・環境設定 (ENVIRONMENT.md)

DomainML ライブラリの実行環境を構築し、テストを再現するための手順です。

## 1. Conda 環境の構築
プロジェクトルートにある `environment.yml` を使用して、必要な依存関係が保証された環境を構築してください。

```bash
conda env create -f environment.yml
conda activate domainml-env
```

## 2. パッケージのインストール

**A: TestPyPI インストール（推奨）**
アップロード済みの `domainml-meta` パッケージを公式デプロイ先よりインストールします。
```bash
pip install -i https://test.pypi.org/simple/ domainml-meta==0.1.0
```

**B: 手動ビルドとローカルインストール**
```bash
pip install build twine
python -m build
pip install dist/domainml_meta-0.1.0-py3-none-any.whl
```

## 3. テストの実行とカバレッジ計測
ソースコードの完全性とカバレッジを測定する場合は、以下のコマンドを実行します。
```bash
pytest tests/unit/ --cov=domainml --cov-branch --cov-report=term-missing
```

* 当環境における監査時の分岐カバレッジは **77%** であり、閾値要件 (75%) を満たしています。
