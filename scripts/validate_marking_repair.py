"""Score a frozen repair against unchanged known regions, explicitly as regression."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ivp.survey import survey


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze', default='frozen-implementation-v1.json')
    args = parser.parse_args()
    phase = args.phase.resolve()
    frozen = json.loads((phase/args.freeze).read_text(encoding='utf-8'))
    for name, sha in frozen['files'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Repair differs from frozen implementation: '+name)
    protocol_path = phase/'regression-protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    for name, sha in protocol['parent_protocols'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Original review protocol changed: '+name)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case in protocol['cases']:
        frame, source, baseline = ROOT/case['frame'], Path(case['source']), ROOT/case['baseline_mask']
        for path, expected in [(frame, case['frame_sha256']), (source, case['source_sha256']),
                               (baseline, case['baseline_mask_sha256'])]:
            if digest(path) != expected:
                raise ValueError('Frozen original or baseline changed: '+str(path))
        rgb = np.array(Image.open(frame).convert('RGB'))
        sw, sh = case['source_display_size'] if 'source_display_size' in case else case['analysis_size']
        sx, sy = sw/rgb.shape[1], sh/rgb.shape[0]
        transform = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
        report, images = survey(rgb, case['roi_normalized'], transform)
        report['source_binding'] = {'source_sha256': case['source_sha256'], 'frame_sha256': case['frame_sha256'],
                                    'timestamp_seconds': case['decoded_timestamp_seconds'],
                                    'transform_to_source_pixel_centres': transform.tolist()}
        previous = np.array(Image.open(baseline).convert('L'))
        if previous.shape != images['cleaned-mask'].shape:
            raise ValueError('Baseline and current crops have different geometry.')
        dest = output/case['id']
        dest.mkdir()
        for name, image in images.items():
            Image.fromarray(image).save(dest/(name+'.png'))
        Image.fromarray(previous).save(dest/'pre-repair-mask.png')
        overlay = images['selected-road'].copy()
        overlay[previous > 0] = (overlay[previous > 0]*.35+np.array([20, 180, 210])*.65).astype(np.uint8)
        Image.fromarray(overlay).save(dest/'pre-repair-overlay.png')
        left, top, right, bottom = report['roi_xyxy_exclusive']
        reviewed = Image.fromarray(images['selected-road'])
        draw = ImageDraw.Draw(reviewed)
        checks = []
        for region in case['review_regions']:
            x1, y1, x2, y2 = region['xyxy_analysis_exclusive']
            if not (left <= x1 < x2 <= right and top <= y1 < y2 <= bottom):
                raise ValueError('Original review region no longer lies within ROI.')
            local = (slice(y1-top, y2-top), slice(x1-left, x2-left))
            before, after = float(np.mean(previous[local] > 0)), float(np.mean(images['cleaned-mask'][local] > 0))
            gate = region.get('gate', protocol['positive_min_default'] if region['role'] == 'positive' else protocol['negative_max_default'])
            passed = after >= gate if region['role'] == 'positive' else after <= gate
            checks.append({**region, 'pre_repair_selected_fraction': before, 'post_repair_selected_fraction': after,
                           'frozen_gate': gate, 'passed': passed})
            draw.rectangle((x1-left, y1-top, x2-left-1, y2-top-1), outline='#00b570' if region['role'] == 'positive' else '#f77e30', width=2)
        reviewed.save(dest/'original-review-regions.png')
        (dest/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if report['fragments']:
            with (dest/'fragments.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(report['fragments'][0]))
                writer.writeheader()
                writer.writerows(report['fragments'])
        records.append({'id': case['id'], 'evidence_role': case['evidence_role'], 'checks': checks,
                        'status': report['status'], 'eligible_for_geometry': report['eligible_for_geometry'],
                        'eligible_for_rules': report['eligible_for_rules']})
    for name, sha in frozen['files'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Implementation changed during validation.')
    checks = [c for r in records for c in r['checks']]
    result = {'kind': 'known_marking_regression_after_bounded_repair', 'uses_ai': False,
              'protocol_sha256': digest(protocol_path), 'frozen_implementation_sha256': digest(phase/args.freeze),
              'cases': records, 'regions_passed': sum(c['passed'] for c in checks), 'regions_total': len(checks),
              'new_heldout_accuracy': False, 'labels_supplied_to_predictor': False,
              'remaining_failures': [{'case': r['id'], **c} for r in records for c in r['checks'] if not c['passed']],
              'limits': ['Known diagnosis/prototype cases are regression reuse, not independent accuracy.',
                         'Unchanged engineering patch gates do not measure full-scene paint quality or semantic recognition.',
                         'Yellow roofs can remain supported appearance; every proposal remains ineligible for geometry/rules.']}
    (output/'regression.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k in ('regions_passed','regions_total','remaining_failures','new_heldout_accuracy')}))
    if result['remaining_failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
