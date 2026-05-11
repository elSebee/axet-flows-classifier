import os
from huggingface_hub import hf_hub_download
import urllib3
import httpx

# Parche SSL corporativo
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["CURL_CA_BUNDLE"] = ""
original_init = httpx.Client.__init__
def new_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = new_init

print("[*] Descargando Phi-3-mini (GGUF Q4_K_M) - Solo pesa ~2.2GB...")

model_dir = "axet_classifier/data/models"
os.makedirs(model_dir, exist_ok=True)

model_path = hf_hub_download(
    repo_id="bartowski/Phi-3-mini-4k-instruct-GGUF",
    filename="Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    local_dir=model_dir
)

print(f"[*] ¡Descarga completada! Modelo guardado en: {model_path}")
