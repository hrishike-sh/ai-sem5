"""Replay real saved synthetic artifacts through current completion contracts.

No detector, model, HTTP/preview server or shared-source write is performed.
I/O faults are injected into new local copies; original records stay read-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exercise(worker,vision_file,output):
    from traffic_ai import local_vision,review_cli,review_llm
    from traffic_ai.evidence_review import load_review,ReviewError
    from ivp.scene_reviews import SceneReviews
    from ivp.evidence_reviews import EvidenceReviews
    worker,vision_file=Path(worker).resolve(),Path(vision_file).resolve()
    original=load_review(worker)
    if not original["events"] or not all(c["event"].get("synthetic") is True for c in original["events"]):
        raise ValueError("Choose an existing explicitly synthetic completed worker run.")
    vision=json.loads(vision_file.read_text(encoding="utf-8"))
    if vision.get("synthetic") is not True or vision.get("state")!="complete":
        raise ValueError("Choose an existing successful explicitly synthetic vision record.")
    image=vision_file.parent/"review-input.png"
    if digest(image)!=vision["review_image_sha256"]:raise ValueError("Saved vision image differs from its record.")
    output.mkdir(exist_ok=False)
    source_bytes={ref["path"]:(worker/ref["path"]).read_bytes() for ref in original["source"]["files"]}
    if any(hashlib.sha256(source_bytes[r["path"]]).hexdigest()!=r["sha256"] for r in original["source"]["files"]):
        raise ValueError("Worker source changed before its diagnostic copy.")
    originals={str(worker/ref["path"]):ref["sha256"] for ref in original["source"]["files"]}
    originals.update({str(vision_file):digest(vision_file),str(image):digest(image)})
    def clone(name):
        folder=output/name;folder.mkdir()
        for name,data in source_bytes.items():(folder/name).write_bytes(data)
        return folder
    source=clone("source-copy")
    record={"kind":"real_synthetic_artifact_contract_acceptance","created_at":datetime.now(timezone.utc).isoformat(),
            "state":"running","source_run":str(worker),"vision_record":str(vision_file),"original_hashes":originals,
            "classification":"Actual worker/vision records replayed; filesystem faults/provider returns are injected. No new inference or UI/browser rendering.","checks":[]}
    receipt=output/"acceptance.json"
    def check(name,passed,**details):
        record["checks"].append({"name":name,"passed":bool(passed),**details})
        receipt.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"check":name,"passed":bool(passed)}),flush=True)
    baseline=output/"baseline"
    document=review_cli.export_review(source,baseline)
    review_cli.validate_review_bundle(baseline)
    prior={p.name:p.read_bytes() for p in baseline.iterdir()}
    def previous_intact():return prior=={p.name:p.read_bytes() for p in baseline.iterdir()}
    check("actual_worker_events_preserved",[e["event"] for e in document["events"]]==[e["event"] for e in original["events"]],events=len(original["events"]))
    copy=review_cli.shutil.copyfileobj
    for error in (OSError,KeyboardInterrupt):
        for filename in ("review.json","review.html","review-complete.json"):
            target=output/(error.__name__+"-"+filename.replace(".","-"))
            def interrupted(src,dst,*args,**kwargs):
                if Path(dst.name).name==filename:
                    dst.write(b"{partial");raise error("Injected publication interruption")
                return copy(src,dst,*args,**kwargs)
            raised=False
            with patch.object(review_cli.shutil,"copyfileobj",interrupted):
                try:review_cli.export_review(source,target)
                except error:raised=True
            cleaned=not target.exists()
            retried=review_cli.export_review(source,target) if cleaned else None
            verified=review_cli.validate_review_bundle(target)["complete"] if retried else False
            check("interrupted_"+error.__name__+"_"+filename,raised and cleaned and verified and previous_intact(),
                  retry_same_path=bool(verified),previous_valid_preserved=previous_intact(),fault="Injected into actual report-file publication")
    target=output/"marker-flush-failure"
    def after_marker(src,dst,*args,**kwargs):
        result=copy(src,dst,*args,**kwargs)
        if Path(dst.name).name=="review-complete.json":
            dst.flush();raise OSError("Injected failure after full marker copy")
        return result
    raised=False
    with patch.object(review_cli.shutil,"copyfileobj",after_marker):
        try:review_cli.export_review(source,target)
        except OSError:raised=True
    check("failure_after_marker_flush_not_published",raised and not target.exists() and previous_intact())
    target=output/"raced-destination"
    render=review_cli.render_html
    def competing_destination(*args,**kwargs):
        target.mkdir()
        for name,data in prior.items():(target/name).write_bytes(data)
        return render(*args,**kwargs)
    raised=False
    with patch.object(review_cli,"render_html",competing_destination):
        try:review_cli.export_review(source,target)
        except FileExistsError:raised=True
    check("concurrent_completed_destination_preserved",raised and prior=={p.name:p.read_bytes() for p in target.iterdir()})
    early=clone("early-source-change");target=output/"early-review"
    class MutatingSelector:
        def __init__(self,**kwargs):pass
        def select(self,*args):
            data=json.loads((early/"status.json").read_text());data["acceptance_fault"]=True
            (early/"status.json").write_text(json.dumps(data))
            return {"state":"selected","fact_ids":["status"],"question_ids":[]}
    raised=False
    with patch.object(review_llm,"LocalReviewSelector",MutatingSelector):
        try:review_cli.export_review(early,target,local_llm=True)
        except ReviewError:raised=True
    absent=not target.exists()
    (early/"status.json").write_bytes(source_bytes["status.json"])
    review_cli.export_review(early,target);review_cli.validate_review_bundle(target)
    check("source_change_before_publication_rejected_and_retry_valid",raised and absent and previous_intact())
    late=clone("late-source-change");target=output/"late-review"
    def late_mutation(*args,**kwargs):
        data=json.loads((late/"status.json").read_text());data["acceptance_fault"]=True
        (late/"status.json").write_text(json.dumps(data))
        return render(*args,**kwargs)
    with patch.object(review_cli,"render_html",late_mutation):snapshot=review_cli.export_review(late,target)
    review_cli.validate_review_bundle(target)
    integrity=EvidenceReviews(output,None,None).integrity("diagnostic",late,snapshot["source"]["files"])
    check("late_mutation_is_saved_snapshot_not_current_source",integrity["state"]=="changed" and previous_intact(),source_integrity=integrity["state"],note="Portable bundle remains an intact earlier snapshot; primary current-source check detects the change.")
    with ThreadPoolExecutor(max_workers=1) as executor:
        scenes=SceneReviews(output,executor)
        def replay(*args,**kwargs):return deepcopy(vision)
        def start():
            attempt=scenes.start(None,vision["timestamp"],"synthetic-frame",vision["review_image_sha256"])
            scenes.future.result(timeout=10)
            return attempt["id"],scenes.get(attempt["id"])
        with patch.object(local_vision,"review_frame",replay):
            good_id,good=start();good_path=scenes.folder(good_id)/"review.json";good_bytes=good_path.read_bytes()
            check("actual_vision_record_replay_complete",good["state"]=="complete" and good["review"]==vision)
            for name,bad in (("empty",{}),("not-complete",{**vision,"state":"unavailable"}),
                             ("bad-observation",{**vision,"observation":{}}),("wrong-frame",{**vision,"timestamp":vision["timestamp"]+1})):
                identifier,_=start();(scenes.folder(identifier)/"review.json").write_text(json.dumps(bad))
                rejected=False
                try:scenes.get(identifier)
                except ValueError:rejected=True
                check("semantic_scene_artifact_"+name,rejected and good_path.read_bytes()==good_bytes)
            identifier,_=start()
            inconsistent=deepcopy(vision)
            category="bus" if "bus" not in inconsistent["observation"]["visible_features"] else "car"
            inconsistent["observation"]["visible_features"]=[category]
            inconsistent["observation"]["summary"]="Unverified scene feature categories: "+category+"."
            (scenes.folder(identifier)/"review.json").write_text(json.dumps(inconsistent))
            rejected=False
            try:scenes.get(identifier)
            except ValueError:rejected=True
            check("bounded_selection_and_display_must_agree",rejected and good_path.read_bytes()==good_bytes,
                  fault="Only display categories changed; original stored selection retained.",
                  scope="Internal consistency, not creator authentication or category accuracy.")
            state=scenes.state;once=[False]
            def failed_complete(folder,value,message):
                if value=="complete" and not once[0]:once[0]=True;raise OSError("Injected complete-status write failure")
                return state(folder,value,message)
            with patch.object(scenes,"state",failed_complete):bad_id,bad=start()
            exists=(scenes.folder(bad_id)/"review.json").exists()
            retry_id,retry=start()
            check("root_review_without_complete_status_not_current",exists and bad["state"]=="unavailable" and "review" not in bad and retry["state"]=="complete" and good_path.read_bytes()==good_bytes,
                  readable_orphan_is_not_advertised=True,fresh_retry=retry_id!=bad_id,previous_valid_preserved=good_path.read_bytes()==good_bytes)
            def partial_provider(*args,output,**kwargs):
                output.mkdir();(output/"review.json").write_text("{")
                raise OSError("Injected internal model-record write failure")
            with patch.object(local_vision,"review_frame",partial_provider):bad_id,bad=start()
            retry_id,retry=start()
            check("partial_model_record_retry_retains_previous",bad["state"]=="unavailable" and "review" not in bad and not (scenes.folder(bad_id)/"review.json").exists() and retry["state"]=="complete" and good_path.read_bytes()==good_bytes,
                  fresh_retry=retry_id!=bad_id,partial_internal_file_retained=True)
    check("all_original_fixture_bytes_preserved",all(digest(path)==value for path,value in originals.items()))
    check("previous_portable_report_preserved",previous_intact())
    record.update(state="complete",passed=sum(c["passed"] for c in record["checks"]),total=len(record["checks"]),completed_at=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    return int(record["passed"]!=record["total"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary",type=Path)
    parser.add_argument("--worker-run",type=Path,required=True)
    parser.add_argument("--vision-record",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute-snapshot",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.execute_snapshot:return exercise(args.worker_run,args.vision_record,args.output/"checks")
    if args.primary is None:parser.error("--primary is required")
    from validate_ai_snapshot import validate
    assembled=validate(args.primary,args.output,run_tests=False)
    snapshot=Path(assembled["snapshot"])
    script=snapshot/"scripts/validate_review_artifacts.py"
    script.write_bytes(Path(__file__).read_bytes())
    command=[sys.executable,str(script),"--execute-snapshot","--worker-run",str(args.worker_run.resolve()),
             "--vision-record",str(args.vision_record.resolve()),"--output",str(args.output.resolve())]
    scratch=args.output/"runtime-temp";scratch.mkdir()
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","TEMP":str(scratch.resolve()),"TMP":str(scratch.resolve()),"TRAFFIC_AI_HOME":str((args.output/"ai-runtime").resolve())}
    with (args.output/"artifact-acceptance.log").open("wb") as log:
        result=subprocess.run(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    primary=args.primary.resolve()
    changed=[name for name,value in assembled["primary_files"].items() if name.endswith(".py") and (not (primary/name).is_file() or digest(primary/name)!=value)]
    record={"kind":"artifact_contract_snapshot_acceptance","created_at":datetime.now(timezone.utc).isoformat(),"test_exit_code":result.returncode,
            "primary_python_unchanged":not changed,"changed_primary_paths":changed,"script_sha256":digest(script),
            "snapshot":str(snapshot),"command":command,"passed":result.returncode==0 and not changed,
            "runtime_changes":"None. This acceptance adds no production validation layer or completion marker."}
    (args.output/"artifact-acceptance.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(record));return int(not record["passed"])


if __name__=="__main__":raise SystemExit(main())
