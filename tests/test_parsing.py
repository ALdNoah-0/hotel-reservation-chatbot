from datetime import date

import pytest
from conftest import module

TODAY = date(2026, 10, 13)


@pytest.mark.parametrize('text,expected', [
    ('book deluxe for 3 nights for two guests starting December 24', {'room_type':'deluxe','nights':3,'guests':2,'checkin':date(2026,12,24)}),
    ('suite from 2026-12-24 to 2026-12-27', {'room_type':'suite','checkin':date(2026,12,24),'checkout':date(2026,12,27)}),
    ('tomorrow for one night, just myself', {'checkin':date(2026,10,14),'nights':1,'guests':1}),
    ('next Friday for a week', {'checkin':date(2026,10,16),'nights':7}),
    ('check out on December 28', {'checkout':date(2026,12,28)}),
    ('December 24, 2027', {'checkin':date(2027,12,24)}),
    ('January 5', {'checkin':date(2027,1,5)}),
    ('3 adults and 2 children', {'guests':5}),
    ('family room for a family of four', {'room_type':'family','guests':4}),
    ('BR-12345678AB', {'booking_id':'BR-12345678AB'}),
    ('my email is ALEX@example.com', {'email':'alex@example.com'}),
    ('my name is Alex Rivera', {'name':'Alex Rivera'}),
])
def test_extracts_literal_slots(config, text, expected):
    parsed = module('hotel_chatbot.parsing').parse_message(text, TODAY, config)
    assert not parsed.errors
    for key, value in expected.items():
        assert parsed.slots[key] == value


@pytest.mark.parametrize('text', [
    'book for 03/04/2027', 'book for February 30', 'book next month',
    'book this weekend', 'book a deluxe or suite', 'book on 2026-02-30',
    'book for -2 nights', 'book for 0 guests', 'book for three nights and five nights',
    'book the presidential suite',
    'book two rooms for 3 guests',
])
def test_ambiguous_or_invalid_slots_require_clarification(config, text):
    assert module('hotel_chatbot.parsing').parse_message(text, TODAY, config).errors


def test_year_and_reference_are_not_counts(config):
    parsed = module('hotel_chatbot.parsing').parse_message('BR-12345678AB on 2027-01-03', TODAY, config)
    assert 'guests' not in parsed.slots
    assert 'nights' not in parsed.slots


def test_negative_guest_count_is_not_silently_positive(config):
    assert module('hotel_chatbot.parsing').parse_message('for -2 guests', TODAY, config).errors
