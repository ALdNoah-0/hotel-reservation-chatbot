from datetime import timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(tmp_path, monkeypatch):
    assert (ROOT / 'app.py').exists(), 'Streamlit app has not been implemented'
    monkeypatch.setenv('HOTEL_DB_PATH',str(tmp_path / 'app.sqlite3'))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py'),default_timeout=30)
    at.run()
    return at


def test_app_renders_chat_and_demo_label(app):
    assert not app.exception
    assert len(app.chat_input) == 1
    assert any('demo' in c.value.lower() for c in app.caption)


def test_chat_remembers_draft_between_reruns(app):
    app.chat_input[0].set_value('book a deluxe room for 3 nights for 2 guests starting December 24').run()
    assert not app.exception
    assert app.session_state['bot'].draft.room_type == 'deluxe'
    app.chat_input[0].set_value('how much would that cost').run()
    assert any('15,000' in m.value for m in app.markdown)
    assert app.session_state['bot'].service.count() == 0


def test_form_books_only_after_confirmation(app):
    today = app.session_state['bot'].service.today
    app.selectbox(key='draft_room').set_value('deluxe')
    app.date_input(key='draft_checkin').set_value(today + timedelta(days=10))
    app.number_input(key='draft_nights').set_value(3)
    app.number_input(key='draft_guests').set_value(2)
    app.text_input(key='draft_name').set_value('Alex Rivera')
    app.text_input(key='draft_email').set_value('alex@example.com')
    app.button(key='review_draft').click().run()
    assert not app.exception
    assert app.session_state['bot'].service.count() == 0
    assert app.session_state['bot'].pending == 'book'
    app.button(key='confirm_action').click().run()
    assert not app.exception
    assert app.session_state['bot'].service.count() == 1
    app.run()
    assert app.session_state['bot'].service.count() == 1


def test_lookup_and_cancel_flow(app):
    from hotel_chatbot.models import Draft
    service = app.session_state['bot'].service
    saved = service.create(Draft(room_type='standard',checkin=service.today + timedelta(days=5),
        nights=2,guests=1,name='Alex Rivera',email='alex@example.com'),'app-existing')
    app.text_input(key='lookup_ref').set_value(saved['reference'])
    app.text_input(key='lookup_email').set_value('alex@example.com')
    app.button(key='lookup_submit').click().run()
    assert not app.exception
    app.button(key='prepare_cancel').click().run()
    assert 'navigation' in app.session_state
    assert app.session_state['navigation'] == 'Reservation assistant'
    assert service.lookup(saved['reference'],'alex@example.com')['status'] == 'active'
    app.button(key='confirm_action').click().run()
    assert not app.exception
    assert service.lookup(saved['reference'],'alex@example.com')['status'] == 'cancelled'
