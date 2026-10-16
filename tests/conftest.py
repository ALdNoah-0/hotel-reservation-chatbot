import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module(name):
    assert importlib.util.find_spec('hotel_chatbot') is not None, 'Chatbot package is not implemented'
    assert importlib.util.find_spec(name) is not None, f'{name} is not implemented'
    return importlib.import_module(name)


@pytest.fixture
def config():
    return module('hotel_chatbot.config').load_config()


@pytest.fixture
def service(tmp_path, config):
    from datetime import date
    return module('hotel_chatbot.reservations').ReservationService(
        tmp_path / 'bookings.sqlite3', config, today=date(2026, 10, 13))


@pytest.fixture
def draft():
    from datetime import date
    return module('hotel_chatbot.models').Draft(
        room_type='deluxe', checkin=date(2026, 12, 24), nights=3,
        guests=2, name='Alex Rivera', email='alex@example.com')
