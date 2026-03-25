import logging
import os
import sys

def setup_logger(name="domainml"):
    """
    開発用途のデバッグロガーを初期化する。
    ファイル 'domainml_debug.log' と標準出力の両方に詳細なDEBUGレベルで出力する。
    本ファイルはバージョン1.0リリース時に削除・無効化を想定。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 既存のハンドラがある場合はクリアして重複を防ぐ
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
    )
    
    # ファイル出力 (Git/PyPIの除外対象)
    log_file = "domainml_debug.log"
    try:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"Warning: Could not setup FileHandler for {log_file} - {e}")

    # コンソール出力
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO) # コンソールはINFOレベルにして画面の氾濫を防ぐ
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()
