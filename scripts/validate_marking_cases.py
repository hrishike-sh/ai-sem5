"""Score original-pixel marking surveys on frozen development annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivp.survey import survey


def coverage(mask, rectangle):
    left, top, right, bottom = rectangle
    return float(np.mean(mask[top:bottom, left:right] > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('development_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.development_root.resolve()
    protocol_path = root / 'marking-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    frozen = {r['run']: r for r in json.loads((root / 'paint-prototype-v2.json').read_text())}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for item in protocol['real_development_labels']:
        source = root / item['image']
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != frozen[item['run']]['source_sha256']:
            raise ValueError('Development crop changed: ' + item['run'])
        rgb = np.asarray(Image.open(source).convert('RGB'))
        report, images = survey(rgb, [0, 0, 1, 1])
        previous = np.asarray(Image.open(source.parent / 'cleaned-mask.png'))
        mask = images['cleaned-mask']
        dest = output / item['run']
        dest.mkdir()
        for name, image in images.items():
            Image.fromarray(image).save(dest / (name + '.png'))
        (dest / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        record = {'run': item['run'], 'source_sha256': digest, 'label_scope': item['label_origin'],
                  'before_total_coverage': float(np.mean(previous > 0)),
                  'after_total_coverage': float(np.mean(mask > 0)),
                  'before_positive': coverage(previous, item['positive_paint_xyxy']),
                  'after_positive': coverage(mask, item['positive_paint_xyxy']),
                  'before_negative': coverage(previous, item['negative_pavement_xyxy']),
                  'after_negative': coverage(mask, item['negative_pavement_xyxy']),
                  'positive_min': item['positive_coverage_min'], 'negative_max': item['negative_coverage_max']}
        record['passed'] = (record['after_positive'] >= record['positive_min']
                            and record['after_negative'] <= record['negative_max'])
        records.append(record)
    result = {'method': 'frozen_two_sided_marking_development_acceptance', 'uses_ai': False,
              'manual_annotations_are_independent_ground_truth': False,
              'annotations_used_by_predictor': False,
              'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'source_hashes': {str(p.relative_to(Path(__file__).resolve().parents[1])):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__), Path(__file__).resolve().parents[1] / 'ivp/paint.py',
                                Path(__file__).resolve().parents[1] / 'ivp/survey.py']},
              'cases': records, 'all_passed': all(r['passed'] for r in records)}
    (output / 'acceptance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
