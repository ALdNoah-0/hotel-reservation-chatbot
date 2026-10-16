from datetime import date, timedelta
from dataclasses import replace

import pytest
from conftest import module


def test_checkin_date_to_phrase_moves_arrival_not_checkout(config):
    parsed = module('hotel_chatbot.parsing').parse_message('change check-in date to December 28',date(2026,10,13),config)
    assert parsed.slots['checkin'] == date(2026,12,28)
    assert 'checkout' not in parsed.slots


def test_explicit_reverse_date_order_preserves_labels(config):
    parsed = module('hotel_chatbot.parsing').parse_message('check-out December 28, check-in December 24',date(2026,10,13),config)
    assert parsed.slots['checkout'] == date(2026,12,28)
    assert parsed.slots['checkin'] == date(2026,12,24)


def test_name_and_email_in_one_message_are_separate(config):
    parsed = module('hotel_chatbot.parsing').parse_message(
        'my name is Alex Rivera and my email is alex@example.com',date(2026,10,13),config)
    assert parsed.slots['name'] == 'Alex Rivera'
    assert parsed.slots['email'] == 'alex@example.com'


def test_availability_question_during_modification_preserves_verified_draft(service,draft):
    saved = service.create(draft,'preserve')
    recognizer = module('hotel_chatbot.intents').IntentRecognizer()
    bot = module('hotel_chatbot.conversation').HotelChatbot(service,recognizer)
    bot.prepare_management('modify',saved['reference'],draft.email)
    bot.respond('do you have a suite available tomorrow')
    assert bot.mode == 'modify'
    assert bot.reference == saved['reference']
    assert bot.draft.room_type == 'deluxe'
    assert bot.draft.checkin == date(2026,12,24)


def test_modification_of_checkin_preserves_nights(service,draft):
    saved = service.create(draft,'move-arrival')
    bot = module('hotel_chatbot.conversation').HotelChatbot(service,module('hotel_chatbot.intents').IntentRecognizer())
    bot.prepare_management('modify',saved['reference'],draft.email)
    bot.respond('change check-in date to December 28')
    assert bot.pending == 'modify'
    bot.confirm()
    changed = service.lookup(saved['reference'],draft.email)
    assert changed['checkin'] == '2026-12-28'
    assert changed['checkout'] == '2026-12-31'
    assert changed['total'] == 15000


def test_checking_out_neighbourhood_is_a_location_question():
    recognizer = module('hotel_chatbot.intents').IntentRecognizer()
    assert recognizer.predict('what is around the area worth checking out').intent == 'location_inquiry'


@pytest.mark.parametrize('text',['I do not want 2 nights','actually I am not sure about 2 nights'])
def test_disputed_stay_detail_disarms_confirmation(service,draft,text):
    bot = module('hotel_chatbot.conversation').HotelChatbot(service,module('hotel_chatbot.intents').IntentRecognizer())
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    bot.respond(text)
    assert bot.pending is None
    assert bot.blocked
    bot.confirm()
    assert service.count() == 0


def test_calendar_overflow_is_a_validation_error(service,draft):
    with pytest.raises(ValueError):
        service.create(replace(draft,checkin=date(9999,12,31)), 'overflow')


@pytest.mark.parametrize('text',[
    'book a deluxe room for 99 guests for 3 nights starting December 24',
    'book a deluxe room from 2026-12-24 to 2026-12-23 for 2 guests',
    'book a standard room for 2 nights for 1 guest starting 2026-01-01',
    'book a standard room for 2 nights for 1 guest starting 2030-01-01',
])
def test_chat_invalid_or_distant_dates_leave_app_usable(tmp_path,monkeypatch,text):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('HOTEL_DB_PATH',str(tmp_path/'ui.sqlite3'))
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
    at.chat_input[0].set_value(text).run()
    assert not at.exception
    assert at.session_state['bot'].service.count() == 0
    # A correction through the form must remain possible.
    assert len(at.chat_input) == 1
