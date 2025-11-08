param([Parameter(Mandatory=$true)][string]$AnthropicKey,[string]$Model="claude-3-5-haiku-latest",[string]$PromptLog="data\clear\prompt_dump.jsonl")
$ErrorActionPreference="Stop"
$env:ANTHROPIC_API_KEY=$AnthropicKey
$env:ANTHROPIC_MODEL=$Model
$env:PROMPT_LOG=$PromptLog
$env:HF_API_KEY=""
$env:HUGGINGFACEHUB_API_TOKEN=""
$env:OPENAI_API_KEY=""
if(!(Test-Path -Path "data\clear")){New-Item -ItemType Directory -Force -Path "data\clear" | Out-Null}
if(!(Test-Path -Path "data\ramehr")){New-Item -ItemType Directory -Force -Path "data\ramehr" | Out-Null}
python .\scripts\run_with_auth_patch.py models\CLEAR\run_dataset_ner.py --notes data\notes.jsonl --targets data\targets.yml --out data\clear\ner_raw.jsonl
python .\scripts\run_with_auth_patch.py models\CLEAR\run_ner_cosine_similarity.py --ner data\clear\ner_raw.jsonl --targets data\targets.yml --out data\clear\ner_cos.jsonl --thr 0.85
python .\scripts\run_with_auth_patch.py models\CLEAR\run_llm_filter_cosine_sim_ner_output.py --in data\clear\ner_cos.jsonl --out data\clear\ner_filtered.jsonl
python .\scripts\run_with_auth_patch.py models\CLEAR\run_extraction.py --notes data\notes.jsonl --ner data\clear\ner_filtered.jsonl --window 160 --out data\clear\evidence.jsonl
python .\scripts\run_with_prompt_dump.py models\CLEAR\run_llm_assertion.py --evidence data\clear\evidence.jsonl --schema data\schema_8fields.json --out data\clear\output.jsonl --csv data\clear\output.csv
python .\scripts\convert_clear_to_ramehr.py
Write-Host "DONE"
