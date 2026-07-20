#!/usr/bin/env bash
# Reason Ready — turnkey base-model fetch (a size-aware catalog).
#
#   ./scripts/fetch-models.sh                 # the baseline: embed-0.6b + rerank-0.6b
#   ./scripts/fetch-models.sh embed-4b        # one model by name
#   ./scripts/fetch-models.sh 4b              # both 4B models (embedder + reranker)
#   ./scripts/fetch-models.sh embedder        # the baseline embedder (embed-0.6b)
#   ./scripts/fetch-models.sh all             # every size (LARGE: ~50 GB)
#   ./scripts/fetch-models.sh --list          # show the catalog with sizes
#   ./scripts/fetch-models.sh --check 4b      # verify on disk, download nothing
#
# The candle backends load weights from a local directory (docs/MODELS.md); this
# script is what puts real weights there. Weights are too big to vendor in git,
# so they are pulled on demand and verified byte-exact against the manifest.
# Idempotent and resumable: a byte-exact file is skipped, a partial is resumed.
#
# The 0.6B pair is the baseline — CPU-runnable and the intended fine-tuning base.
# 4B/8B are the quality ceiling and effectively want a GPU (an 8B loads as f32 on
# CPU = ~32 GB RAM). A fine-tuned checkpoint is not in this catalog: it is just a
# local weights dir — point RRO_EMBEDDER_WEIGHTS straight at it.
#
# Environment knobs:
#   RRO_MODELS_DIR   where models land        (default: <repo>/models)
#   HF_ENDPOINT      Hugging Face base URL     (default: https://huggingface.co;
#                    set to e.g. https://hf-mirror.com behind a firewall)
#   HF_REV           git revision to pull      (default: main)
#   HF_TOKEN         bearer token for HF        (optional; for rate limits)
#
# Weights are apache-2.0 (Qwen/Qwen3-Embedding-*, Qwen/Qwen3-Reranker-*).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${RRO_MODELS_DIR:-$ROOT/models}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_REV="${HF_REV:-main}"

# ── the catalog ─────────────────────────────────────────────────────────────
# Which HF repo + local dir each short name maps to.  name  repo  dir
CATALOG="$(cat <<'EOF'
embed-0.6b  Qwen/Qwen3-Embedding-0.6B  qwen3-embedding-0.6b
embed-4b    Qwen/Qwen3-Embedding-4B    qwen3-embedding-4b
embed-8b    Qwen/Qwen3-Embedding-8B    qwen3-embedding-8b
rerank-0.6b Qwen/Qwen3-Reranker-0.6B   qwen3-reranker-0.6b
rerank-4b   Qwen/Qwen3-Reranker-4B     qwen3-reranker-4b
rerank-8b   Qwen/Qwen3-Reranker-8B     qwen3-reranker-8b
EOF
)"

# The files to pull per model, with exact byte sizes (the integrity check).
# Captured 2026-07-16 from revision `main`. Only what the candle loaders read
# (config.json + tokenizer.json + every model*.safetensors, plus the shard index
# and the small sentence-transformers/pooling descriptors the card ships).
# columns:  name  file  bytes
MANIFEST="$(cat <<'EOF'
embed-0.6b config.json 727
embed-0.6b config_sentence_transformers.json 215
embed-0.6b modules.json 349
embed-0.6b tokenizer_config.json 9706
embed-0.6b tokenizer.json 11423705
embed-0.6b model.safetensors 1191586416
embed-4b config.json 727
embed-4b config_sentence_transformers.json 215
embed-4b modules.json 349
embed-4b tokenizer_config.json 7256
embed-4b tokenizer.json 11422947
embed-4b model.safetensors.index.json 30431
embed-4b model-00001-of-00002.safetensors 4965826464
embed-4b model-00002-of-00002.safetensors 3077765624
embed-8b config.json 729
embed-8b config_sentence_transformers.json 215
embed-8b modules.json 349
embed-8b tokenizer_config.json 7256
embed-8b tokenizer.json 11422947
embed-8b model.safetensors.index.json 30432
embed-8b model-00001-of-00004.safetensors 4900037024
embed-8b model-00002-of-00004.safetensors 4915959512
embed-8b model-00003-of-00004.safetensors 4983067656
embed-8b model-00004-of-00004.safetensors 335570376
rerank-0.6b config.json 727
rerank-0.6b config_sentence_transformers.json 325
rerank-0.6b tokenizer_config.json 9706
rerank-0.6b tokenizer.json 11422654
rerank-0.6b model.safetensors 1191588280
rerank-4b config.json 727
rerank-4b config_sentence_transformers.json 325
rerank-4b tokenizer_config.json 9706
rerank-4b tokenizer.json 11422654
rerank-4b model.safetensors.index.json 32819
rerank-4b model-00001-of-00002.safetensors 4058781760
rerank-4b model-00002-of-00002.safetensors 3984833200
rerank-8b config.json 729
rerank-8b config_sentence_transformers.json 294
rerank-8b tokenizer_config.json 9706
rerank-8b tokenizer.json 11422654
rerank-8b model.safetensors.index.json 32878
rerank-8b model-00001-of-00005.safetensors 4027618768
rerank-8b model-00002-of-00005.safetensors 4060268160
rerank-8b model-00003-of-00005.safetensors 4043508680
rerank-8b model-00004-of-00005.safetensors 3003274088
rerank-8b model-00005-of-00005.safetensors 1242472576
EOF
)"

ALL_MODELS="embed-0.6b embed-4b embed-8b rerank-0.6b rerank-4b rerank-8b"
BASELINE="embed-0.6b rerank-0.6b"

# ── helpers ─────────────────────────────────────────────────────────────────
filesize() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || wc -c <"$1" 2>/dev/null || echo -1; }

human() {
  awk -v b="$1" 'BEGIN{ split("B KB MB GB TB",u); i=1; while(b>=1024 && i<5){b/=1024;i++}
    printf (i==1?"%d %s":"%.1f %s"), b, u[i] }'
}

catalog_field() { # $1=model $2=col(2=repo,3=dir)
  awk -v m="$1" -v c="$2" '$1==m{print $c; exit}' <<< "$CATALOG"
}

model_bytes() { # total bytes for a model
  awk -v m="$1" '$1==m{s+=$3} END{print s+0}' <<< "$MANIFEST"
}

known_model() { grep -qE "^$1 " <<< "$CATALOG"; }

# Resolve a selection word to model short-names.
resolve() {
  case "$1" in
    baseline|"")   echo "$BASELINE" ;;
    all|both)      echo "$ALL_MODELS" ;;
    embedder)      echo "embed-0.6b" ;;
    reranker)      echo "rerank-0.6b" ;;
    embed|embed-all)   echo "embed-0.6b embed-4b embed-8b" ;;
    rerank|rerank-all) echo "rerank-0.6b rerank-4b rerank-8b" ;;
    0.6b) echo "embed-0.6b rerank-0.6b" ;;
    4b)   echo "embed-4b rerank-4b" ;;
    8b)   echo "embed-8b rerank-8b" ;;
    *) if known_model "$1"; then echo "$1"; else
         echo "unknown selection: $1" >&2
         echo "try one of: baseline | all | embedder | reranker | 0.6b | 4b | 8b | <model-name> (--list)" >&2
         return 2
       fi ;;
  esac
}

# ── args ────────────────────────────────────────────────────────────────────
CHECK_ONLY=0; LIST=0; SELECTIONS=()
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --list)  LIST=1 ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -* ) echo "unknown flag: $arg" >&2; exit 2 ;;
    * )  SELECTIONS+=("$arg") ;;
  esac
done

if [[ "$LIST" == "1" ]]; then
  echo "Reason Ready model catalog (target: $MODELS_DIR)"
  printf "  %-12s %-26s %10s\n" "name" "hf repo" "size"
  while read -r m repo _dir; do
    [[ -z "${m:-}" ]] && continue
    printf "  %-12s %-26s %10s\n" "$m" "$repo" "$(human "$(model_bytes "$m")")"
  done <<< "$CATALOG"
  echo "groups: baseline (default) | 0.6b | 4b | 8b | embed | rerank | all"
  exit 0
fi

# Expand selections (default = baseline) into a de-duped, catalog-ordered set.
WANT=""
[[ ${#SELECTIONS[@]} -eq 0 ]] && SELECTIONS=(baseline)
for s in "${SELECTIONS[@]}"; do WANT+=" $(resolve "$s")"; done
MODELS=""
for m in $ALL_MODELS; do [[ " $WANT " == *" $m "* ]] && MODELS+=" $m"; done

# ── download one file, resuming if partial, verifying size after ────────────
have_cli() { command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; }
hf_cli()   { if command -v hf >/dev/null 2>&1; then hf "$@"; else huggingface-cli "$@"; fi; }

download_one() {
  local repo="$1" file="$2" want="$3" dest="$4"
  local url="$HF_ENDPOINT/$repo/resolve/$HF_REV/$file?download=true"
  if have_cli && [[ "$HF_ENDPOINT" == "https://huggingface.co" ]]; then
    hf_cli download "$repo" "$file" --revision "$HF_REV" --local-dir "$(dirname "$dest")" >/dev/null
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 2 --retry-connrefused -C - \
      ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} -O "$dest" "$url"
  else
    echo "ERROR: need one of hf / huggingface-cli / curl / wget to download." >&2; return 3
  fi
  local got; got="$(filesize "$dest")"
  if [[ "$got" != "$want" ]]; then
    echo "  ✗ $file: got $got bytes, expected $want — incomplete or wrong revision" >&2; return 4
  fi
}

# ── run ─────────────────────────────────────────────────────────────────────
total=0; for m in $MODELS; do total=$((total + $(model_bytes "$m"))); done
echo "── Reason Ready: model fetch ──────────────────────────────────"
echo "target:   $MODELS_DIR"
echo "endpoint: $HF_ENDPOINT   rev: $HF_REV"
echo "models:  $MODELS  (total $(human "$total"))"
if have_cli; then echo "method:   huggingface CLI (resumable, LFS-aware)"
else echo "method:   $(command -v curl >/dev/null 2>&1 && echo curl || echo wget) (resumable)"; fi
echo

missing=0 fetched=0 verified=0
for m in $MODELS; do
  repo="$(catalog_field "$m" 2)"; dir="$(catalog_field "$m" 3)"
  target="$MODELS_DIR/$dir"; mkdir -p "$target"
  echo "▸ $m  ($repo -> $dir, $(human "$(model_bytes "$m")"))"
  while read -r r_model r_file r_bytes; do
    [[ -z "${r_model:-}" || "$r_model" != "$m" ]] && continue
    dest="$target/$r_file"
    if [[ -f "$dest" && "$(filesize "$dest")" == "$r_bytes" ]]; then
      echo "  ✓ $r_file ($(human "$r_bytes")) — present"; verified=$((verified+1)); continue
    fi
    if [[ "$CHECK_ONLY" == "1" ]]; then
      echo "  … $r_file ($(human "$r_bytes")) — MISSING"; missing=$((missing+1)); continue
    fi
    echo "  ↓ $r_file ($(human "$r_bytes"))"
    download_one "$repo" "$r_file" "$r_bytes" "$dest"
    echo "  ✓ $r_file — verified"; fetched=$((fetched+1))
  done <<< "$MANIFEST"
  echo
done

if [[ "$CHECK_ONLY" == "1" ]]; then
  [[ "$missing" -gt 0 ]] && { echo "$missing file(s) missing. Run without --check to download."; exit 1; }
  echo "All selected model files present and byte-exact."; exit 0
fi

echo "── done: $fetched downloaded, $verified already present ───────"
echo
echo "Point the engine at what you fetched, e.g.:"
for m in $MODELS; do
  dir="$(catalog_field "$m" 3)"
  case "$m" in
    embed-*)  echo "  RRO_EMBEDDER=candle-qwen           RRO_EMBEDDER_WEIGHTS=$MODELS_DIR/$dir" ;;
    rerank-*) echo "  RRO_RERANKER=candle-cross-encoder  RRO_RERANKER_WEIGHTS=$MODELS_DIR/$dir" ;;
  esac
done
echo "Or one command (baseline):  RRO_REAL=1 ./scripts/quickstart.sh"
