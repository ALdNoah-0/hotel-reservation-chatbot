from dataclasses import replace

import pytest
from conftest import module


@pytest.fixture(scope='session')
def recognizer():
    return module('hotel_chatbot.intents').IntentRecognizer()


@pytest.fixture
def bot(service, recognizer):
    return module('hotel_chatbot.conversation').HotelChatbot(service, recognizer)


@pytest.mark.parametrize('text,intent', [
    ("I don't want a bot, I want a person",'request_human_agent'),
    ('my room is dirty and the ac is not working','complaint'),
    ('noise from the hallway kept me up all night','complaint'),
    ('how do I fix my wifi router at home','out_of_scope'),
    ('please cancel booking BR-12345678AB','cancel_booking'),
    ('can I change my reservation dates','modify_booking'),
    ('how much would that cost','check_price'),
    ('do you have a pool','amenities_inquiry'),
    ('do you have a suite available tomorrow','check_availability'),
    ('when is check-in','checkin_checkout_time'),
    ('what payment methods are accepted','payment_inquiry'),
    ('what rooms do you offer','room_types_inquiry'),
    ('where are you located','location_inquiry'),
    ('book a deluxe room for tomorrow','book_room'),
    ('I do not want to book a room','fallback'),
    ('asdkj qweoi random gibberish','fallback'),
    ('cancel my booking and reserve another room','fallback'),
    ('can you arrange a place for us to stay','book_room'),
    ('give me a ballpark accommodation budget','check_price'),
    ('I need to pull out of this reservation','cancel_booking'),
    ('can we swap the room in my existing booking','modify_booking'),
    ('when are we allowed to show up','checkin_checkout_time'),
    ('which part of town is the hotel in','location_inquiry'),
    ('do you require money up front','payment_inquiry'),
    ('we have had nothing but trouble here','complaint'),
    ('can I bypass the automated assistant','request_human_agent'),
    ('what comes with our stay','amenities_inquiry'),
    ('can you help us choose from your room options','room_types_inquiry'),
    ('do not cancel my booking','fallback'),
    ('book a room and modify my existing reservation','fallback'),
])
def test_recognizes_requests_and_abstains(recognizer, text, intent):
    assert recognizer.predict(text).intent == intent


def test_booking_waits_for_identity_and_explicit_confirmation(bot, service):
    bot.respond('book a deluxe room for 3 nights starting December 24 for 2 guests')
    assert service.count() == 0
    bot.respond('my name is Alex Rivera')
    assert bot.draft.name == 'Alex Rivera'
    bot.respond('alex@example.com')
    assert bot.pending == 'book'
    assert '15,000' in bot.respond('how much would that cost')
    assert service.count() == 0
    reply = bot.respond('confirm booking')
    assert 'BR-' in reply
    assert service.count() == 1
    bot.respond('confirm booking')
    assert service.count() == 1


def test_bare_answers_complete_multiturn_draft(bot, service):
    bot.respond('I want to make a reservation')
    bot.respond('deluxe')
    bot.respond('December 24')
    bot.respond('3')
    bot.respond('2')
    bot.respond('Alex Rivera')
    bot.respond('alex@example.com')
    assert bot.pending == 'book'
    bot.confirm()
    assert service.count() == 1


def test_form_correction_requires_new_confirmation(bot, service, draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    assert bot.pending == 'book'
    bot.update_draft(nights=2)
    assert '10,000' in bot.summary()
    bot.confirm()
    assert service.count() == 1
    assert service.lookup(bot.last_booking['reference'], draft.email)['total'] == 10000


def test_unknown_message_does_not_apply_incidental_slots(bot):
    bot.respond('book a deluxe room for December 24 for 2 nights for 2 guests')
    prior = replace(bot.draft)
    bot.respond('my homework has 8 guests and 9 nights in the story')
    assert bot.draft == prior


def test_faq_interruption_preserves_booking_and_unsupported_facts_are_unknown(bot):
    bot.respond('book a family room for December 24 for 2 nights for 4 guests')
    prior = replace(bot.draft)
    assert 'Wi-Fi' in bot.respond('do you have wifi')
    assert bot.draft == prior
    assert 'Spa' in bot.respond('do you have a spa')
    assert bot.draft == prior
    bot.config['hotel_faq'] = {**bot.config['hotel_faq'], 'spa': None}
    reply = bot.respond('do you have a spa')
    assert 'not configured' in reply.lower() or 'not listed' in reply.lower()


def test_ambiguous_date_does_not_arm_confirmation(bot, service, draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    reply = bot.respond('change check-in to 03/04/2027')
    assert bot.pending is None
    bot.confirm()
    assert service.count() == 0
    assert 'date' in reply.lower()


def test_human_request_does_not_claim_staff_transfer(bot):
    reply = bot.respond("I don't want a bot, I want a person")
    assert 'contact' in reply.lower() or 'configured' in reply.lower()
    assert 'connecting you' not in reply.lower()
    assert 'transferred' not in reply.lower()


def test_cancellation_requires_lookup_and_confirmation(bot, service, draft):
    saved = service.create(draft,'existing')
    bot.respond(f"cancel booking {saved['reference']}")
    bot.respond('wrong@example.com')
    assert bot.pending is None
    bot.respond(draft.email)
    assert bot.pending == 'cancel'
    assert service.lookup(saved['reference'],draft.email)['status'] == 'active'
    bot.respond('confirm cancellation')
    assert service.lookup(saved['reference'],draft.email)['status'] == 'cancelled'


def test_modification_uses_verified_booking_and_explicit_confirm(bot, service, draft):
    saved = service.create(draft,'modify')
    bot.respond(f"modify booking {saved['reference']} {draft.email}")
    bot.respond('make it 2 nights')
    assert service.lookup(saved['reference'],draft.email)['total'] == 15000
    bot.respond('confirm modification')
    assert service.lookup(saved['reference'],draft.email)['total'] == 10000


def test_reset_never_deletes_bookings(bot, service, draft):
    service.create(draft,'keep')
    bot.respond('book a standard room')
    bot.reset()
    assert bot.mode is None
    assert bot.draft.room_type is None
    assert service.count() == 1


def test_declining_or_negating_confirmation_does_not_write(bot, service, draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    bot.respond('do not confirm booking')
    assert service.count() == 0
    bot.respond('no')
    bot.confirm()
    assert service.count() == 0


def test_faq_slots_do_not_silently_change_pending_quote(bot, draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    bot.respond('what is the cost of a suite')
    assert bot.draft.room_type == 'deluxe'


def test_negated_request_disarms_prepared_booking(bot,service,draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    bot.respond('I do not want to book that room')
    bot.respond('yes')
    assert service.count() == 0


def test_wrong_confirmation_action_never_commits_booking(bot,service,draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    bot.respond('confirm modification')
    assert service.count() == 0


def test_conflicting_actions_disarm_pending_confirmation(bot,service,draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    bot.respond('cancel my booking and reserve a standard room')
    bot.confirm()
    assert service.count() == 0


def test_plain_yes_after_faq_is_not_a_booking_command(bot,service,draft):
    bot.mode = 'book'
    bot.update_draft(**draft.__dict__)
    bot.respond('do you have breakfast')
    bot.respond('yes')
    assert service.count() == 0
