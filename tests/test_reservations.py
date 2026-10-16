from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date

import pytest
from conftest import module


def test_quote_uses_catalog_and_three_nights(service, draft):
    quote = service.quote(draft, require_identity=True)
    assert quote.nightly_rate == 5000
    assert quote.total == 15000
    assert quote.checkout == date(2026,12,27)


@pytest.mark.parametrize('fields', [
    {'guests':0}, {'guests':3}, {'nights':0}, {'nights':31},
    {'checkin':date(2026,1,1)}, {'checkout':date(2026,12,23)},
    {'checkout':date(2026,12,28)}, {'room_type':'presidential'},
    {'name':''}, {'email':'bad-email'}, {'nights':True},
])
def test_invalid_reservation_never_persists(service, draft, fields):
    with pytest.raises(ValueError):
        service.create(replace(draft, **fields), 'invalid')
    assert service.count() == 0


def test_booking_is_persistent_and_identity_protected(service, draft):
    booked = service.create(draft, 'req-1')
    assert booked['total'] == 15000
    assert service.lookup(booked['reference'], 'ALEX@example.com')['name'] == 'Alex Rivera'
    assert service.lookup(booked['reference'], 'someone@example.com') is None
    fresh = module('hotel_chatbot.reservations').ReservationService(service.path, service.config, today=date(2026,10,13))
    assert fresh.lookup(booked['reference'], draft.email)['status'] == 'active'


def test_confirmation_replay_is_idempotent(service, draft):
    first = service.create(draft, 'same-request')
    second = service.create(draft, 'same-request')
    assert first['reference'] == second['reference']
    assert service.count() == 1


def test_capacity_and_adjacent_stays(service, draft):
    stock = service.config['rooms']['deluxe']['inventory']
    for n in range(stock):
        service.create(draft, f'fill-{n}')
    assert service.available('deluxe', date(2026,12,25), date(2026,12,28)) == 0
    assert service.available('deluxe', date(2026,12,27), date(2026,12,30)) == stock
    with pytest.raises(ValueError, match='available|sold out'):
        service.create(draft, 'overflow')


def test_concurrent_last_room_has_one_winner(service, draft):
    for n in range(service.config['rooms']['deluxe']['inventory'] - 1):
        service.create(draft, f'existing-{n}')
    def attempt(n):
        try:
            service.create(draft, f'concurrent-{n}')
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, range(4)))
    assert results.count(True) == 1
    assert service.count() == service.config['rooms']['deluxe']['inventory']


def test_cancellation_verifies_identity_and_restores_room(service, draft):
    booked = service.create(draft, 'cancel-me')
    with pytest.raises(ValueError):
        service.cancel(booked['reference'], 'wrong@example.com')
    assert service.lookup(booked['reference'], draft.email)['status'] == 'active'
    service.cancel(booked['reference'], draft.email)
    service.cancel(booked['reference'], draft.email)
    assert service.lookup(booked['reference'], draft.email)['status'] == 'cancelled'
    assert service.available('deluxe', date(2026,12,24), date(2026,12,27)) == service.config['rooms']['deluxe']['inventory']


def test_failed_modification_preserves_original(service, draft):
    booked = service.create(draft, 'original')
    occupied = replace(draft, checkin=date(2027,1,1))
    for n in range(service.config['rooms']['deluxe']['inventory']):
        service.create(occupied, f'occupied-{n}')
    with pytest.raises(ValueError):
        service.modify(booked['reference'], draft.email, occupied)
    assert service.lookup(booked['reference'], draft.email)['checkin'] == '2026-12-24'


def test_modification_excludes_itself_and_requotes(service, draft):
    booked = service.create(draft, 'change')
    changed = service.modify(booked['reference'], draft.email, replace(draft,nights=2))
    assert changed['total'] == 10000
    assert changed['checkout'] == '2026-12-26'
    service.cancel(booked['reference'], draft.email)
    with pytest.raises(ValueError):
        service.modify(booked['reference'], draft.email, draft)


def test_sequential_reservations_do_not_overcount_inventory(service, draft):
    # Same room can serve two adjacent stays; spanning availability counts rooms, not rows.
    service.create(replace(draft,nights=1), 'day-1')
    service.create(replace(draft,checkin=date(2026,12,25),nights=1), 'day-2')
    assert service.available('deluxe',date(2026,12,24),date(2026,12,26)) == service.config['rooms']['deluxe']['inventory'] - 1
