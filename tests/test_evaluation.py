import json
import subprocess
import sys
from pathlib import Path


def test_evaluation_cli_records_all_exposed_regression_examples(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable,str(root/'scripts'/'evaluate.py'),'--output-dir',str(tmp_path)],
                            capture_output=True,text=True,encoding='utf-8',timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path/'results.json').read_text(encoding='utf-8'))
    assert report['sample_count'] == 96
    assert len(report['predictions']) == 96
    assert 'regression' in report['dataset_status'].lower()
    assert 'fallback' in report['hybrid']['per_class']
    assert report['hybrid']['correct'] + report['hybrid']['incorrect'] == 96
