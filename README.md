# comfyui-stablsr

Put the StableSR webui_786v_139.ckpt model into Comyfui/models/stablesr/  
Put the StableSR stablesr_768v_000139.ckpt model into Comyfui/models/checkpoints/

Download the ckpt from HuggingFace https://huggingface.co/Iceclear/StableSR/

There is a setup json in /examples/ to load the workflow into Comfyui

# Turbo setting

Reference link https://github.com/pkuliyi2015/sd-webui-stablesr/issues/57 you can significally increase speed with sd 2.1 turbo.

Put the [sd_turbo.safetensors](https://huggingface.co/stabilityai/sd-turbo/blob/main/sd_turbo.safetensors) model into Comyfui/models/checkpoints/  
Put the stablesr_webui_sd-v2-1-512-ema-000117.ckpt in [webui_512v_models.zip](https://huggingface.co/Iceclear/StableSR/blob/main/webui_512v_models.zip) into Comyfui/models/stablesr/  

Yon can try turbo workflow in /examples/.

The example uses https://github.com/ethansmith2000/comfy-todo node for faster speeds.
