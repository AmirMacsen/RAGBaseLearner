#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('dienstag/chinese-macbert-base', cache_dir="D:\\ai\\modelscope_models")