"""Comprueba que el nodo ComfyUI-Trellis2 extrae bien las caracteristicas de DINOv3.

El nodo recorre las capas de DINOv3 a mano (extract_features). Si la version de
transformers instalada cambio la API, las caracteristicas salen mal sin dar error
y TRELLIS.2 genera formas sin sentido. Se compara con la pasada normal del modelo.
Uso (desde C:\\AMD-AI\\Pixal3D-ROCm):
    python_embeded\\python.exe prueba_dinov3.py
"""
import os
import sys

import torch
import torch.nn.functional as F
import transformers

raiz = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(raiz, "ComfyUI", "custom_nodes", "ComfyUI-Trellis2"))
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor  # noqa: E402

print("transformers", transformers.__version__, "| torch", torch.__version__)
modelo = os.path.join(raiz, "ComfyUI", "models", "facebook", "dinov3-vitl16-pretrain-lvd1689m")
ext = DinoV3FeatureExtractor(modelo, image_size=512)
ext.cuda()
torch.manual_seed(0)
img = torch.rand(1, 3, 512, 512, device="cuda")
img = ext.transform(img)

with torch.no_grad():
    nodo = ext.extract_features(img).float()
    salida = ext.model(pixel_values=img.to(ext.model.embeddings.patch_embeddings.weight.dtype),
                       output_hidden_states=True)
    ref = F.layer_norm(salida.hidden_states[-1].float(), nodo.shape[-1:])

print("forma nodo", tuple(nodo.shape), "| forma referencia", tuple(ref.shape))
if nodo.shape == ref.shape:
    cos = F.cosine_similarity(nodo, ref, dim=-1)
    print(f"similitud coseno: media {cos.mean().item():.4f}, minima {cos.min().item():.4f}")
    print("error max", (nodo - ref).abs().max().item())
    print("RESULTADO:", "OK (el nodo extrae bien DINOv3)" if cos.min().item() > 0.99
          else "MAL (las caracteristicas del nodo no coinciden con DINOv3)")
else:
    print("RESULTADO: MAL (formas distintas)")
