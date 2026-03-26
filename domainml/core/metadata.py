from __future__ import annotations
import copy
from enum import Enum
from typing import Optional, Literal, Union


class ConstraintStrength(Enum):
    """制約強度の列挙型（F-301）
    
    STRICT: 数学的に厳密な保証（凸最適化で制約として強制）
    SOFT:   損失関数へのペナルティ（近似保証）
    PREFERENCE: 可能な限り遵守（最も弱い制約）
    """
    STRICT = "strict"
    SOFT = "soft"
    PREFERENCE = "preference"


class MonotonicityDirection(Enum):
    """単調性の方向を表す列挙型（F-302）"""
    INCREASING = 1
    DECREASING = -1
    NONE = 0


def monotonicity_to_direction(mono_str: str) -> MonotonicityDirection:
    """文字列 'inc'/'dec'/'none' を MonotonicityDirection に変換（後方互換）"""
    return {'inc': MonotonicityDirection.INCREASING,
            'dec': MonotonicityDirection.DECREASING,
            'none': MonotonicityDirection.NONE}[mono_str]


def direction_to_monotonicity(direction: MonotonicityDirection) -> str:
    """MonotonicityDirection を文字列表現に変換（後方互換）"""
    return {MonotonicityDirection.INCREASING: 'inc',
            MonotonicityDirection.DECREASING: 'dec',
            MonotonicityDirection.NONE: 'none'}[direction]

class FeatureMetadata:
    """
    ドメイン知識を保持し伝播させるためのメタデータクラス。
    
    Attributes:
        feature_names (List[str]): 特徴量名のリスト
        monotonicities (List[str]): 'inc'(単調増加), 'dec'(単調減少), 'none'(制約なし)
        constraint_types (List[str]): 'strict'(厳密), 'soft'(ペナルティ), 'none'(なし)
        groups (List[int]): 特徴量群ID。同じIDの特徴量はセットとして扱われる(-1は独立)
        manifold_flags (List[bool]): 多様体仮説を適用するかどうか
        manifold_flags (List[bool]): 多様体仮説を適用するかどうか
        control_flags (List[bool]): 制御変数（多様体仮説や他の制約から除外するもの）かどうか
        extrapolation_sigma (Union[float, List[float]]): 外挿領域における制約の適用範囲（標準偏差の倍数）。各特徴量で異なる値を指定可能。
    """
    def __init__(self, 
                 feature_names: List[str],
                 monotonicities: Optional[List[Literal['inc', 'dec', 'none']]] = None,
                 constraint_types: Optional[List[Literal['strict', 'soft', 'none']]] = None,
                 groups: Optional[List[int]] = None,
                 manifold_flags: Optional[List[bool]] = None,
                 control_flags: Optional[List[bool]] = None,
                 extrapolation_sigma: Union[float, List[float]] = 3.0):
        
        self.n_features = len(feature_names)
        self.feature_names = feature_names
        
        self.monotonicities = monotonicities if monotonicities is not None else ['none'] * self.n_features
        self.constraint_types = constraint_types if constraint_types is not None else ['none'] * self.n_features
        self.groups = groups if groups is not None else [-1] * self.n_features
        self.manifold_flags = manifold_flags if manifold_flags is not None else [False] * self.n_features
        self.control_flags = control_flags if control_flags is not None else [False] * self.n_features
        
        if isinstance(extrapolation_sigma, list):
            self.extrapolation_sigma = extrapolation_sigma
        else:
            self.extrapolation_sigma = [extrapolation_sigma] * self.n_features
        
        self._validate()

    def _validate(self):
        """入力リストの長さが揃っているか、および許容される値か検証する"""
        if len(self.monotonicities) != self.n_features:
            raise ValueError("monotonicities must have the same length as feature_names.")
        if len(self.constraint_types) != self.n_features:
            raise ValueError("constraint_types must have the same length as feature_names.")
        if len(self.groups) != self.n_features:
            raise ValueError("groups must have the same length as feature_names.")
        if len(self.manifold_flags) != self.n_features:
            raise ValueError("manifold_flags must have the same length as feature_names.")
        if len(self.control_flags) != self.n_features:
            raise ValueError("control_flags must have the same length as feature_names.")
        if len(self.extrapolation_sigma) != self.n_features:
            raise ValueError("extrapolation_sigma list must have the same length as feature_names.")
            
        allowed_mono = {'inc', 'dec', 'none'}
        for m in self.monotonicities:
            if m not in allowed_mono:
                raise ValueError(f"Invalid monotonicity: {m}. Allowed values are 'inc', 'dec', 'none'.")
                
        allowed_types = {'strict', 'soft', 'none'}
        for t in self.constraint_types:
            if t not in allowed_types:
                raise ValueError(f"Invalid constraint type: {t}. Allowed values are 'strict', 'soft', 'none'.")

    def slice(self, indices: List[int]) -> 'FeatureMetadata':
        """
        特定のインデックスの特徴量のみを抽出した新しいFeatureMetadataを返す。
        """
        return FeatureMetadata(
            feature_names=[self.feature_names[i] for i in indices],
            monotonicities=[self.monotonicities[i] for i in indices],
            constraint_types=[self.constraint_types[i] for i in indices],
            groups=[self.groups[i] for i in indices],
            manifold_flags=[self.manifold_flags[i] for i in indices],
            control_flags=[self.control_flags[i] for i in indices],
            extrapolation_sigma=[self.extrapolation_sigma[i] for i in indices]
        )

    def merge(self, other: 'FeatureMetadata') -> 'FeatureMetadata':
        """
        他のFeatureMetadataと結合した新しいFeatureMetadataを返す。
        """
        return FeatureMetadata(
            feature_names=self.feature_names + other.feature_names,
            monotonicities=self.monotonicities + other.monotonicities,
            constraint_types=self.constraint_types + other.constraint_types,
            groups=self.groups + other.groups,
            manifold_flags=self.manifold_flags + other.manifold_flags,
            control_flags=self.control_flags + other.control_flags,
            extrapolation_sigma=self.extrapolation_sigma + other.extrapolation_sigma
        )
        
    def clone(self) -> 'FeatureMetadata':
        """深いコピーを返す"""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"FeatureMetadata(n_features={self.n_features}, features={self.feature_names})"
