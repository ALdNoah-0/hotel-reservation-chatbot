import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    config = json.loads(Path(path or ROOT / 'data' / 'hotel.json').read_text(encoding='utf-8'))
    ZoneInfo(config['timezone'])
    if not config['rooms']:
        raise ValueError('Configure at least one room category.')
    faq = config.get('hotel_faq', {})
    if not isinstance(faq, dict) or any(value is not None and (not isinstance(value, str) or len(value) > 2000) for value in faq.values()):
        raise ValueError('hotel_faq must contain text answers of at most 2,000 characters, or null for unknown facts.')
    for key, room in config['rooms'].items():
        if not key.isidentifier():
            raise ValueError('Room keys must be simple identifiers.')
        for field in ('nightly_rate','capacity','inventory'):
            minimum = 0 if field == 'inventory' else 1
            if type(room[field]) is not int or room[field] < minimum:
                raise ValueError(f'{key}: {field} must be an integer >= {minimum}.')
    return config


def hotel_today(config):
    return datetime.now(ZoneInfo(config['timezone'])).date()


def money(value):
    return f'PHP {value:,.0f}'
