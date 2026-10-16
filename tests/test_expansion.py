"""Behavior checks for configurable FAQs, request notes, and read-only lookup."""
from dataclasses import replace
import sqlite3

import pytest

from hotel_chatbot.conversation import HotelChatbot
from hotel_chatbot.intents import IntentRecognizer
from hotel_chatbot.reservations import ReservationService


@pytest.fixture(scope='session')
def recognizer():
    return IntentRecognizer()


@pytest.fixture
def bot(service, recognizer):
    return HotelChatbot(service, recognizer)


@pytest.mark.parametrize('question,key,answer',[
    ('Can I bring my dog?', 'pets', 'Pets require advance approval.'),
    ('Do you have wheelchair access?', 'accessibility', 'Step-free entrance and accessible rooms on request.'),
    ('Can I store my luggage before check-in?', 'luggage', 'Luggage storage is available until 8 PM.'),
    ('What time does breakfast start?', 'breakfast_hours', 'Breakfast is served from 7 AM to 10 AM.'),
    ('What are the pool opening hours?', 'pool_hours', 'The pool opens at 8 AM and closes at 8 PM.'),
    ('Can I get an extra bed?', 'extra_beds', 'Extra beds are not available.'),
    ('Do you offer an airport shuttle?', 'shuttle', 'Airport transport must be arranged separately.'),
    ('What documents do I need at check-in?', 'checkin_documents', 'Bring your booking reference and valid ID.'),
    ('What is your policy for children staying?', 'children', 'All children count toward room capacity.'),
])
def test_specific_faq_uses_configured_answer_without_booking(bot, service, question, key, answer):
    bot.config['hotel_faq'] = {key: answer}
    reply = bot.respond(question)
    assert answer in reply
    assert service.count() == 0
    assert bot.mode is None


def test_missing_accessibility_information_stays_unknown(bot):
    bot.config['hotel_faq'] = {'accessibility': None}
    reply = bot.respond('Is the hotel wheelchair accessible?')
    assert 'accessibility' in reply.lower()
    assert 'not configured' in reply.lower()


@pytest.mark.parametrize('question', [
    'Can I bring my dog?', 'Do you have wheelchair access?', 'What time does breakfast start?',
    'What are the pool opening hours?', 'Can I get an extra bed?',
    'What is your policy for children staying?', 'Can I store my luggage before check-in?',
    'Do you offer an airport shuttle?', 'Do you have a restaurant?', 'Do you offer room service?',
    'Is laundry available?', 'What is your smoking policy?', 'Are visitors allowed?',
    'What documents do I need at check-in?', 'Can I get early check-in or late check-out?',
    'What is your cancellation policy?',
])
def test_all_information_picker_questions_are_read_only(bot, service, question):
    from hotel_chatbot.conversation import FALLBACK
    reply = bot.respond(question)
    assert reply != FALLBACK
    assert bot.mode is None
    assert bot.pending is None
    assert service.count() == 0


def test_cancellation_policy_question_does_not_start_cancellation(bot, draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    reply = bot.respond('What is your cancellation policy?')
    assert bot.config['cancellation_policy'] in reply
    assert bot.mode == 'book'
    assert bot.pending == 'book'


def test_request_note_requires_fresh_confirmation_and_persists(bot, service, draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    old_token = bot.request_id
    reply = bot.respond('Special request: Quiet room, please.')
    assert getattr(bot.draft, 'special_requests', None) == 'Quiet room, please.'
    assert bot.request_id != old_token
    assert 'not guaranteed' in reply.lower()
    assert service.count() == 0
    bot.confirm()
    saved = service.lookup(bot.last_booking['reference'], draft.email)
    assert saved['special_requests'] == 'Quiet room, please.'


def test_note_content_cannot_change_dates_or_room(bot, draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    bot.respond('Special request: Suite decorations for December 26, 2026.')
    assert getattr(bot.draft, 'special_requests', None) == 'Suite decorations for December 26, 2026.'
    assert bot.draft.room_type == 'deluxe'
    assert bot.draft.checkin == draft.checkin


def test_note_validation_rejects_oversize_input_before_writing(bot, service, draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    reply = bot.update_draft(special_requests='x' * 501)
    assert '500' in reply
    assert bot.pending is None
    assert service.count() == 0


def test_modification_preserves_notes_and_can_explicitly_clear_them(bot, service, draft):
    bot.update_draft(**{**draft.__dict__, 'special_requests': 'Quiet room'})
    bot.confirm()
    saved = bot.last_booking
    bot.prepare_management('modify', saved['reference'], draft.email)
    assert getattr(bot.draft, 'special_requests', None) == 'Quiet room'
    reply = bot.respond('Clear special requests')
    assert getattr(bot.draft, 'special_requests', None) == ''
    assert bot.pending == 'modify'
    assert service.lookup(saved['reference'], draft.email)['special_requests'] == 'Quiet room'
    bot.confirm()
    assert service.lookup(saved['reference'], draft.email)['special_requests'] == ''


def test_view_booking_requires_identity_and_never_mutates(bot, service, draft):
    saved = service.create(draft, 'view-existing')
    assert 'reference' in bot.respond('Check my reservation status').lower()
    assert saved['checkin'] not in bot.respond(saved['reference'])
    assert 'No matching' in bot.respond('wrong@example.com')
    reply = bot.respond(draft.email)
    assert saved['reference'] in reply
    assert 'active' in reply.lower()
    assert '15,000' in reply
    assert bot.pending is None
    assert service.lookup(saved['reference'], draft.email)['status'] == 'active'


def test_cancelled_booking_can_be_viewed_without_reactivation(bot, service, draft):
    saved = service.create(draft, 'view-cancelled')
    service.cancel(saved['reference'], draft.email)
    reply = bot.respond(f"Show my booking {saved['reference']} for {draft.email}")
    assert saved['reference'] in reply
    assert 'cancelled' in reply.lower()
    assert '15,000' in reply
    assert bot.pending is None
    assert service.lookup(saved['reference'], draft.email)['status'] == 'cancelled'


def test_readonly_lookup_does_not_capture_request_to_rearrange_booking(recognizer):
    assert recognizer.predict('i need to shuffle around my reservation details').intent == 'modify_booking'


def test_legacy_database_adds_empty_notes_without_changing_saved_booking(tmp_path, config, draft):
    path = tmp_path / 'legacy.sqlite3'
    # A complete existing reservation schema, from the previous app version.
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE rooms (id TEXT PRIMARY KEY, room_type TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE reservations (reference TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
            room_id TEXT NOT NULL, room_type TEXT NOT NULL, checkin TEXT NOT NULL, checkout TEXT NOT NULL,
            nights INTEGER NOT NULL, guests INTEGER NOT NULL, name TEXT NOT NULL, email TEXT NOT NULL,
            nightly_rate INTEGER NOT NULL, total INTEGER NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            INSERT INTO rooms VALUES('deluxe-001','deluxe',1);
            INSERT INTO reservations VALUES('BR-LEGACY12345','legacy','deluxe-001','deluxe','2026-12-24','2026-12-27',
            3,2,'Alex Rivera','alex@example.com',5000,15000,'active','old','old');''')
    migrated = ReservationService(path, config, today=draft.checkin)
    row = migrated.lookup('BR-LEGACY12345', draft.email)
    assert row.get('special_requests') == ''
    assert row['total'] == 15000
    assert row['status'] == 'active'
    assert row['created_at'] == 'old'
    ReservationService(path, config, today=draft.checkin)
    assert migrated.count() == 1


@pytest.mark.parametrize('notes', ['x' * 501, None, 'bad\x00note'])
def test_service_rejects_invalid_notes_without_creating_reservation(service, draft, notes):
    with pytest.raises(ValueError, match='Special requests'):
        service.create(replace(draft, special_requests=notes), 'invalid-notes')
    assert service.count() == 0


def test_cancel_policy_wording_preserves_pending_booking(bot, draft):
    bot.update_draft(**draft.__dict__)
    reply = bot.respond('What are the rules to cancel a reservation?')
    assert bot.config['cancellation_policy'] in reply
    assert bot.mode == 'book'
    assert bot.pending == 'book'


@pytest.mark.parametrize('question,key', [
    ('Can I store my luggage before check-out on December 26, 2026?', 'luggage'),
    ('Can I get early check-in on December 26, 2026?', 'early_late'),
])
def test_dated_faq_cannot_change_reviewed_stay(bot, draft, question, key):
    bot.config['hotel_faq'] = {key: 'Please arrange this with reception.'}
    bot.update_draft(**draft.__dict__)
    original = replace(bot.draft)
    token = bot.request_id
    reply = bot.respond(question)
    assert 'Please arrange this with reception.' in reply
    assert bot.draft == original
    assert bot.request_id == token
    assert bot.pending == 'book'


def test_streamlit_notes_are_reviewed_and_saved_after_confirmation(tmp_path, monkeypatch):
    from pathlib import Path
    from datetime import timedelta
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('HOTEL_DB_PATH', str(tmp_path / 'notes-app.sqlite3'))
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=30).run()
    assert not app.exception
    app.selectbox(key='draft_room').set_value('standard')
    app.date_input(key='draft_checkin').set_value(app.session_state['bot'].service.today + timedelta(days=10))
    app.text_input(key='draft_name').set_value('Alex Rivera')
    app.text_input(key='draft_email').set_value('alex@example.com')
    app.text_area(key='draft_requests').set_value('Quiet room, please.')
    app.button(key='review_draft').click().run()
    assert not app.exception
    assert app.session_state['bot'].service.count() == 0
    assert app.session_state['bot'].draft.special_requests == 'Quiet room, please.'
    app.button(key='confirm_action').click().run()
    assert not app.exception
    saved = app.session_state['bot'].last_booking
    assert saved['special_requests'] == 'Quiet room, please.'
    assert any('Quiet room, please.' in message.value for message in app.markdown)
