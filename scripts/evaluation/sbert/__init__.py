import os
from pathlib import Path

# 获取当前文件(__init__.py)所在的目录
package_root = Path(__file__).resolve().parents[3]

# 基于包目录设置模型路径
models_dir = os.environ.get('SENTENCE_TRANSFORMERS_HOME', str(package_root / 'outputs' / 'cache' / 'sentence_transformers'))
os.makedirs(models_dir, exist_ok=True)  # 确保目录存在

# 设置环境变量
os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', models_dir)
os.environ.setdefault('TFHUB_DOWNLOAD_PROGRESS', '1')
os.environ.setdefault('TFHUB_CACHE_DIR', models_dir)
