"""
domainml/analysis/conflict_detector.py
後方互換性シム: 旧名称 conflict_detector モジュールへのアクセスを維持する。

v0.3.0 以降、LinearCoefConflictChecker は coef_checker モジュールに移動されました。
このファイルは既存コードの import 文が壊れないよう維持しています。

使用例(非推奨):
    from domainml.analysis.conflict_detector import LinearCoefConflictChecker
    
推奨:
    from domainml.analysis.coef_checker import LinearCoefConflictChecker
    # または
    from domainml import LinearCoefConflictChecker
"""
import warnings

warnings.warn(
    "domainml.analysis.conflict_detector は非推奨です。"
    "代わりに domainml.analysis.coef_checker を使用してください。"
    "このシムは将来のバージョンで削除されます。",
    DeprecationWarning,
    stacklevel=2,
)

from domainml.analysis.coef_checker import LinearCoefConflictChecker  # noqa: F401, E402

__all__ = ["LinearCoefConflictChecker"]
