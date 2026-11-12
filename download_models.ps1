$root = "E:\Myst_Studio"
function Get-File($url,$dest){
  Write-Host "Downloading $url -> $dest"
  Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
}

# Zero-cost inference engine (no Ollama)
Get-File "https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp.exe" "$root\01_Engines\koboldcpp\koboldcpp.exe"

# OEM GGUF models (free)
Get-File "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" "$root\02_Myst_Series\Myst_Nano_2B\Myst-Nano-2B.gguf"
Get-File "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf" "$root\02_Myst_Series\Myst_Prime_4B\Myst-Prime-4B.gguf"
Get-File "https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf" "$root\02_Myst_Series\Myst_Realm_8B\Myst-Realm-8B.gguf"
Get-File "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf" "$root\02_Myst_Series\Myst_Architect_7B\Myst-Architect-7B.gguf"

# Cloudflare tunnel (free public HTTPS)
Get-File "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" "$root\11_Myst_AI\bin\cloudflared.exe"

Write-Host "All downloads complete."
