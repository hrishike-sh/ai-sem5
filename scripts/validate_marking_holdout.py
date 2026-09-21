"""Read-only original-frame spot checks with a frozen marking-review protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ivp.survey import survey


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', type=Path)
    parser.add_argument('--protocol', default='heldout-protocol-v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    phase = args.phase.resolve()
    protocol_path = phase/args.protocol
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    initial = json.loads((phase/'heldout-protocol-v1.json').read_text(encoding='utf-8'))
    paint_path, survey_path = ROOT/'ivp/paint.py', ROOT/'ivp/survey.py'
    if digest(paint_path) != initial['algorithm_sha256'] or digest(survey_path) != initial['survey_sha256']:
        raise ValueError('Algorithm differs from frozen review version; preserve results and declare a new evaluation phase.')
    baseline_path = Path(initial['baseline_v1_source'])
    if digest(baseline_path) != initial['baseline_v1_sha256']:
        raise ValueError('Historical comparison implementation changed.')
    baseline = ModuleType('marking_legacy_comparison')
    exec(compile(baseline_path.read_text(encoding='utf-8'), str(baseline_path), 'exec'), baseline.__dict__)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case in protocol['cases']:
        frame = phase/case['frame']
        source = Path(case['source'])
        if digest(frame) != case['frame_sha256'] or digest(source) != case['source_sha256']:
            raise ValueError('Source or original review frame changed: ' + case['id'])
        rgb = np.array(Image.open(frame).convert('RGB'))
        source_width, source_height = case.get('source_display_size', case['analysis_size'])
        sx, sy = source_width/rgb.shape[1], source_height/rgb.shape[0]
        transform = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
        report, images = survey(rgb, case['roi_normalized'], transform)
        previous, old_images = baseline.survey(rgb, case['roi_normalized'], transform)
        dest = output/case['id']
        dest.mkdir()
        for name, image in images.items():
            Image.fromarray(image).save(dest/(name+'.png'))
        Image.fromarray(old_images['cleaned-mask']).save(dest/'baseline-mask.png')
        Image.fromarray(old_images['annotated-survey']).save(dest/'baseline-overlay.png')
        left, top, right, bottom = report['roi_xyxy_exclusive']
        reviewed = Image.fromarray(images['selected-road'])
        draw = ImageDraw.Draw(reviewed)
        checks = []
        for region in case['review_regions']:
            x1, y1, x2, y2 = region['xyxy_analysis_exclusive']
            if not (left <= x1 < x2 <= right and top <= y1 < y2 <= bottom):
                raise ValueError('Review region is outside its frozen ROI: ' + region['id'])
            local = (slice(y1-top, y2-top), slice(x1-left, x2-left))
            before = float(np.mean(old_images['cleaned-mask'][local] > 0))
            after = float(np.mean(images['cleaned-mask'][local] > 0))
            positive = region['role'] == 'positive'
            gate = protocol['engineering_review_gates']['positive_selected_fraction_min' if positive else 'negative_selected_fraction_max']
            passed = after >= gate if positive else after <= gate
            checks.append({**region, 'baseline_selected_fraction': before, 'current_selected_fraction': after,
                           'frozen_gate': gate, 'passed': passed})
            draw.rectangle((x1-left, y1-top, x2-left-1, y2-top-1), outline='#00b570' if positive else '#f77e30', width=2)
        reviewed.save(dest/'original-review-regions.png')
        (dest/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        records.append({'id': case['id'], 'source_sha256': case['source_sha256'],
                        'frame_sha256': case['frame_sha256'], 'timestamp_seconds': case['decoded_timestamp_seconds'],
                        'roi_normalized': case['roi_normalized'],
                        'baseline_coverage_percent': previous['summary']['coverage_percent'],
                        'current_coverage_percent': report['summary']['coverage_percent'],
                        'state': 'withheld_flat_input' if report['parameters']['flat_input_withheld'] else 'appearance_candidates',
                        'checks': checks, 'all_review_regions_passed': all(c['passed'] for c in checks),
                        'parameters': report['parameters'], 'support': report['support']})
    if digest(paint_path) != initial['algorithm_sha256'] or digest(survey_path) != initial['survey_sha256']:
        raise ValueError('Frozen algorithm changed during validation.')
    sources_unchanged = all(digest(Path(c['source'])) == c['source_sha256']
                            and digest(phase/c['frame']) == c['frame_sha256'] for c in protocol['cases'])
    if not sources_unchanged:
        raise ValueError('Frozen source changed during validation.')
    result = {'kind': 'frozen_original_pixel_marking_spot_checks', 'protocol_sha256': digest(protocol_path),
              'algorithm_sha256': digest(paint_path), 'survey_sha256': digest(survey_path),
              'algorithm_unchanged_from_frozen_version': True, 'source_frames_unchanged': sources_unchanged,
              'predictor_receives_review_labels': False,
              'uses_ai': False, 'cases': records, 'cases_passed': sum(c['all_review_regions_passed'] for c in records),
              'cases_total': len(records), 'review_regions_passed': sum(c['passed'] for r in records for c in r['checks']),
              'review_regions_total': sum(len(r['checks']) for r in records),
              'all_review_regions_passed': all(c['all_review_regions_passed'] for c in records),
              'limits': ['Manual original-pixel review regions, not independent expert or full pixel truth.',
                         'Held out only from the recent two-source tuning; media may share physical scenes.',
                         'Review fractions and gates are not semantic precision, recall, IoU or legal-rule accuracy.',
                         'Masks are computed before region scoring; thresholds/gates were not tuned to these outputs.']}
    (output/'acceptance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'cases_passed': result['cases_passed'], 'cases_total': result['cases_total'],
                      'regions_passed': result['review_regions_passed'], 'regions_total': result['review_regions_total'],
                      'failures': [{'case': r['id'], 'region': c['id'], 'selected_fraction': c['current_selected_fraction']}
                                   for r in records for c in r['checks'] if not c['passed']]}))
    if not result['all_review_regions_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
