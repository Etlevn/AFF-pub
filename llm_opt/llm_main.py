import json
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm_opt import llm_prmt
from llm_opt import llm_config as config
from llm_opt.llm_parse import obtain_valid_json_from_llm, load_json_relaxed


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def optimize_once() -> Path:
    # Generate prompt
    print(f"[LLM_OPT] RUN_DIR={config.RUN_DIR}", flush=True)
    prompt = llm_prmt.build_prompt(config.RUN_DIR)
    print(f"[LLM_OPT] Prompt ready, length={len(prompt)} chars", flush=True)

    # Call the high-order function to uniformly obtain JSON
    print("[LLM_OPT] Calling LLM ...", flush=True)
    cleaned, parsed = obtain_valid_json_from_llm(prompt)
    print(f"[LLM_OPT] obtain_valid_json_from_llm -> parsed={'yes' if parsed is not None else 'no'}", flush=True)
    # If the parsing is still not successful, use local loose parsing to further repair
    if parsed is None and isinstance(cleaned, str):
        print("[LLM_OPT] Fallback to load_json_relaxed ...", flush=True)
        cleaned2, parsed2 = load_json_relaxed(cleaned)
        cleaned, parsed = cleaned2, parsed2
        print(f"[LLM_OPT] load_json_relaxed -> parsed={'yes' if parsed is not None else 'no'}", flush=True)

    # Save
    base = Path(__file__).resolve().parents[1]
    out_dir = base / "analysis" / config.RUN_DIR / "llm"
    ensure_dir(out_dir)
    out_path = out_dir / "llm_opt.json"

    print(f"[LLM_OPT] Saving result to: {out_path}", flush=True)
    if parsed is not None:
        # Unify the top-level structure: if it is {"factors": [...]}, take its list to avoid subsequent reading of the empty list
        if isinstance(parsed, dict) and isinstance(parsed.get("factors"), list):
            parsed = parsed["factors"]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned if isinstance(cleaned, str) else str(cleaned))
    return out_path


def main() -> None:
    out = optimize_once()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
