"""Export a portable HTML/JSON evidence review without modifying the input run."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import tempfile

from .evidence_review import ReviewError, digest, load_review, strict_json


def validate_review_bundle(output):
    """A completion marker binds the two fully written report files to their hashes."""
    root = Path(output).resolve()
    with (root/"review-complete.json").open("rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ReviewError("Review completion marker is oversized.")
    marker = strict_json(raw)
    if (not isinstance(marker, dict) or marker.get("kind") != "traffic_evidence_review_bundle" or
            marker.get("complete") is not True or not isinstance(marker.get("files"), dict) or
            set(marker["files"]) != {"review.json", "review.html"}):
        raise ReviewError("Review completion marker is invalid.")
    for name, expected in marker["files"].items():
        path = root/name
        if path.resolve().parent != root:
            raise ReviewError("Review bundle file escapes its directory.")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ReviewError("Review bundle file does not match its completion hash: "+name)
    return marker


def _write_bundle(destination, payload, page):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".traffic-review-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        (stage/"review.json").write_text(payload, encoding="utf-8")
        (stage/"review.html").write_text(page, encoding="utf-8")
        marker = {"schema_version": 1, "kind": "traffic_evidence_review_bundle", "complete": True,
                  "files": {name: digest((stage/name).read_bytes()) for name in ("review.json", "review.html")}}
        (stage/"review-complete.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
        # Exclusive directory/file creation prevents competing repeats overwriting a report.
        destination.mkdir(exist_ok=False)
        created = []
        try:
            for name in ("review.json", "review.html", "review-complete.json"):
                with (stage/name).open("rb") as source, (destination/name).open("xb") as target:
                    created.append(destination/name)
                    shutil.copyfileobj(source, target)
        except BaseException:
            # Only remove paths created by this publication attempt; never recurse.
            for path in reversed(created):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                destination.rmdir()
            except OSError:
                pass
            raise


def render_html(document, images=None):
    esc = lambda v: html.escape(str(v), quote=True)
    images = images or {}
    cards = []
    for review in document["events"]:
        event = review["event"]
        image = images.get(event.get("thumbnail"))
        thumbnail = ('<img alt="Exported evidence thumbnail" src="data:image/png;base64,' +
                     base64.b64encode(image).decode("ascii") + '">') if image else "<p>No evidence thumbnail available.</p>"
        flags = "SYNTHETIC" if event.get("synthetic") is True else "Provenance unrecorded" if "synthetic" not in event else "Non-synthetic label"
        fact_items = []
        for fact in review["facts"]:
            refs = fact.get("citations", [{"pointer": fact["pointer"], "recorded": True}])
            references = ", ".join(
                f'<code>{esc(fact.get("pointer_base", "event"))}:{esc(ref["pointer"])}</code>' +
                (" (not recorded)" if ref.get("recorded") is False else "") for ref in refs)
            fact_items.append(f'<li><b>{esc(fact["id"])}</b>: {esc(fact["text"])} <span class="reference">{references}</span></li>')
        facts = "".join(fact_items)
        warnings = "".join(f'<li>{esc(w["text"])}</li>' for w in review["warnings"])
        questions = "".join(f'<li>{esc(q["text"])}</li>' for q in review["review_questions"])
        model = review["language_model"]
        emphasis = ""
        if model["state"] == "selected":
            selected = set(model["fact_ids"])
            emphasized = "".join(f'<li>{esc(f["text"])}</li>' for f in review["facts"] if f["id"] in selected)
            selected_questions = set(model["question_ids"])
            emphasized += "".join(f'<li>{esc(q["text"])}</li>' for q in review["review_questions"] if q["id"] in selected_questions)
            emphasis = f'<h3>Local model suggested focus</h3><ul>{emphasized}</ul>'
        reference = json.dumps(review["evidence_ref"], ensure_ascii=False)
        raw = json.dumps(event, ensure_ascii=False, indent=2, allow_nan=False)
        track_excerpt = ""
        for field, label in (("stop_context", "Stop-window CSV excerpt"), ("track_context", "Track CSV excerpt")):
            context = review.get(field)
            if context:
                raw_context = json.dumps(context, ensure_ascii=False, indent=2, allow_nan=False)
                track_excerpt += f'<details><summary>{label} · {context["sample_count"]} matching records</summary><p>{esc(context["limitation"])}</p><pre>{esc(raw_context)}</pre></details>'
        stop = review.get("stop_review")
        stop_note = (f'<p>Stop record consistency: {esc(stop["state"])}. CSV coverage: {esc(stop["csv_coverage"])}. '
                     f'{esc(stop["limitation"])}</p>') if stop else ""
        cards.append(f'''<article data-status="{esc(event['status'])}">
<h2>{esc(event['event_id'])} · {esc(event['type'])}</h2>
<p class="badge">{esc(event['status'])} · {esc(flags)} · {esc(event['timestamp_seconds'])} s</p>
{thumbnail}<h3>Evidence explanation</h3><ul>{facts}</ul>
<ul class="warnings">{warnings}</ul>{stop_note}{emphasis}
<h3>Human review</h3><ul>{questions}</ul>
<p>Local model: {esc(model['state'])}. {esc(model.get('reason', ''))}</p>
<p class="reference">Source citation: <code>{esc(reference)}</code></p>
{track_excerpt}<details><summary>Original event (unchanged)</summary><pre>{esc(raw)}</pre></details></article>''')
    warning_text = "".join(f'<li>{esc(w["text"])}</li>' for w in document["warnings"])
    limits = "".join(f'<li>{esc(item)}</li>' for item in document["limitations"])
    files = "".join(f'<tr><td>{esc(f["path"])}</td><td>{esc(f["bytes"])}</td><td><code>{esc(f["sha256"])}</code></td></tr>' for f in document["source"]["files"])
    counts = ", ".join(f"{key}: {value}" for key, value in document["summary"]["statuses"].items())
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'none'; base-uri 'none'; form-action 'none'">
<title>Traffic evidence review</title><style>
*{{box-sizing:border-box}}body{{font:16px/1.55 system-ui,sans-serif;margin:0;background:#edf2f3;color:#182b35}}
main{{max-width:1080px;margin:auto;padding:28px 20px}}h1{{font-size:2rem}}h2{{font-size:1.3rem;overflow-wrap:anywhere}}
article,.panel{{background:white;border:1px solid #cad7dc;border-radius:12px;padding:22px;margin:22px 0}}
.badge{{display:inline-block;background:#e2edf0;border-radius:8px;padding:6px 12px;font-weight:650}}
img{{display:block;max-width:100%;max-height:480px;border-radius:8px}}li{{margin:8px 0}}code{{font-size:.85em;overflow-wrap:anywhere}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f7;padding:12px}}.warnings{{color:#7a3515}}table{{width:100%;border-collapse:collapse}}td{{border-top:1px solid #ddd;padding:8px;vertical-align:top}}
@media print{{body{{background:white}}article{{break-inside:avoid}}}}
</style></head><body><main><h1>Traffic evidence review</h1>
<p>Run {esc(document['source']['run_id'])} · {esc(document['generated_at'])}</p>
<p>{esc(document['summary']['event_count'])} exported assessments · {esc(counts)}</p>
<section class="panel"><h2>Interpretation limits</h2><ul>{limits}</ul><ul class="warnings">{warning_text}</ul>
<p>Local-model suggestions only select existing evidence and review questions. All facts and warnings remain visible.</p></section>
{''.join(cards) or '<p>No events were exported. This is not evidence that the scene was clear.</p>'}
<section class="panel"><h2>Source file hashes</h2><p>Hashes identify exported evidence bytes. They do not independently verify the original footage or truth of its labels.</p>
<table><thead><tr><th>File</th><th>Bytes</th><th>SHA-256</th></tr></thead><tbody>{files}</tbody></table></section>
</main></body></html>'''


def export_review(run, output, *, local_llm=False, max_events=5000, max_llm_events=8, model=None,
                  llm_timeout=120, local_provider="ollama", model_file=None):
    root, destination = Path(run).resolve(), Path(output).resolve()
    if destination == root or root in destination.parents or destination in root.parents:
        raise ReviewError("Output must be a separate directory outside the source run.")
    # Refuse replacement so a mistaken path can never overwrite an earlier review.
    if destination.exists():
        raise ReviewError("Output already exists; choose a new review directory.")
    document = load_review(root, max_events=max_events)

    def snapshot_bytes(ref):
        path = root / ref["path"]
        if path.resolve().parent != root:
            raise ReviewError("Evidence path changed while creating the review.")
        with path.open("rb") as stream:
            data = stream.read(ref["bytes"] + 1)
        if len(data) != ref["bytes"] or digest(data) != ref["sha256"]:
            raise ReviewError("Evidence changed while creating the review; retry from a completed stable run.")
        return data

    if local_llm:
        from .review_llm import DEFAULT_MODEL, LocalLlamaCppSelector, LocalReviewSelector, apply_local_selection
        if local_provider == "llama-cpp":
            if model_file is None:
                raise ReviewError("The local llama.cpp provider requires --model-file for provenance.")
            selector = LocalLlamaCppSelector(model_file=model_file, model=model or "traffic-review", timeout=llm_timeout)
        elif local_provider == "ollama":
            selector = LocalReviewSelector(model=model or DEFAULT_MODEL, timeout=llm_timeout)
        else:
            raise ReviewError("Unsupported local provider.")
        apply_local_selection(document, selector, max_events=max_llm_events)
    images = {}
    for review in document["events"]:
        ref = review.get("thumbnail_ref")
        if ref and ref["path"] not in images:
            images[ref["path"]] = snapshot_bytes(ref)
    # Recheck all consumed bytes after optional slow inference, before publication.
    verified_sources = {}
    for ref in document["source"]["files"]:
        data = snapshot_bytes(ref)
        if ref["path"] in ("motion-incidents.jsonl","motion-incidents.json","motion-tracks.csv"):
            verified_sources[ref["path"]] = data
    from .review_grounding import validate_grounding
    document["grounding"] = validate_grounding(document,verified_sources)
    payload = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
    page = render_html(document, images)
    _write_bundle(destination, payload, page)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed motion run containing motion-incidents exports")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the source run")
    parser.add_argument("--local-llm", action="store_true", help="Use an already running local Ollama model for evidence focus")
    parser.add_argument("--model", help="Installed Ollama model name; default qwen3-vl:4b")
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--max-llm-events", type=int, default=8)
    parser.add_argument("--llm-timeout", type=float, default=120, help="Per-request local model timeout, seconds (maximum 300)")
    parser.add_argument("--local-provider", choices=["ollama", "llama-cpp"], default="ollama")
    parser.add_argument("--model-file", type=Path, help="Existing GGUF file for llama.cpp model provenance")
    args = parser.parse_args(argv)
    if args.model and not args.local_llm:
        parser.error("--model requires --local-llm")
    if args.model_file and (not args.local_llm or args.local_provider != "llama-cpp"):
        parser.error("--model-file requires --local-llm --local-provider llama-cpp")
    try:
        document = export_review(args.run, args.output, local_llm=args.local_llm, max_events=args.max_events,
                                 max_llm_events=args.max_llm_events, model=args.model, llm_timeout=args.llm_timeout,
                                 local_provider=args.local_provider, model_file=args.model_file)
    except (OSError, ValueError) as exc:
        print("Review failed: " + str(exc), file=sys.stderr)
        return 2
    model_states = {}
    for review in document["events"]:
        state = review["language_model"]["state"]
        model_states[state] = model_states.get(state, 0) + 1
    print(json.dumps({"output": str(args.output.resolve()), **document["summary"], "local_model_states": model_states}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
