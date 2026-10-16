"""MVP guarantees: every suggested question is answered, FAQ routing, and classifier quality floors."""
import json
import re
from pathlib import Path

import pytest

from hotel_chatbot.conversation import FALLBACK, HotelChatbot
from hotel_chatbot.hotel_info import EXAMPLES, TOPICS, FaqRetriever
from hotel_chatbot.intents import IntentRecognizer, prepare

ROOT = Path(__file__).resolve().parents[1]
APP_QUESTIONS = re.findall(r"'[^']+': '([^']+)',", (ROOT / 'app.py').read_text(encoding='utf-8').split("questions = {")[1].split('}')[0])


@pytest.fixture(scope='module')
def recognizer():
    return IntentRecognizer()


@pytest.fixture
def bot(service, recognizer):
    return HotelChatbot(service, recognizer)


def test_app_dropdown_has_questions():
    assert len(APP_QUESTIONS) >= 40


@pytest.mark.parametrize('question', APP_QUESTIONS)
def test_every_dropdown_question_gets_a_real_answer(bot, service, question):
    reply = bot.respond(question)
    assert reply != FALLBACK
    assert 'not configured' not in reply.lower()
    assert 'not certain' not in reply.lower()
    assert service.count() == 0


def test_every_topic_has_examples_and_a_configured_answer(config):
    assert set(EXAMPLES) == set(TOPICS)
    for key in TOPICS:
        assert isinstance(config['hotel_faq'].get(key), str) and config['hotel_faq'][key].strip(), key


@pytest.mark.parametrize('question,topic', [
    ('can I order food to my room', 'room_service'), ('do you accept animals', 'pets'),
    ('do you accept bitcoin', None), ('can my friend stay overnight', 'visitors'), ('I lost my key card', 'security'),
    ('is it noisy at night', 'noise'), ('is there a safe deposit box at reception', 'security'), ('do you have bathtubs', 'in_room'),
    ('Can my alligator stay with me', 'pets'), ('can I bring my parrot', 'pets'), ('do you serve fish or chicken', 'restaurant'),
    ('can I book a massage', 'spa'), ('what amenities are in the room', 'in_room'),
    ('how can I contact you', 'contact'), ('who are you', 'about_bot'), ('can I request a quiet room', 'special_requests'),
])
def test_faq_questions_route_to_their_topic(recognizer, question, topic):
    assert recognizer.predict(question).topic == topic


@pytest.mark.parametrize('text,intent', [
    ('book a room for 2 people tomorrow for 1 night', 'book_room'),
    ('do you have room for 5 people', 'room_types_inquiry'),
    ("what's a good workout routine", 'out_of_scope'),
])
def test_booking_and_capacity_rules_do_not_collide(recognizer, text, intent):
    assert recognizer.predict(text).intent == intent


@pytest.mark.parametrize('text', ['how many nights can I stay', 'how many days can I stay', 'how long can I stay',
                                  'what is the maximum stay', 'how many nights can I book', 'can I stay for a month'])
def test_length_of_stay_questions_get_the_limit_not_a_booking(bot, service, text):
    reply = bot.respond(text)
    assert '1 to 30 nights' in reply and 'Pet policy' not in reply
    assert bot.mode is None and service.count() == 0


@pytest.mark.parametrize('text', ['how long can I stay', 'can I stay longer', 'can I stay here', 'what is the minimum stay'])
def test_the_word_stay_does_not_retrieve_the_pet_policy(recognizer, text):
    assert recognizer.predict(text).topic != 'pets'


@pytest.mark.parametrize('text,nights', [('book a deluxe room for 2 weeks from december 24 for 2 guests', 14),
                                         ('book a suite for a week', 7), ('my trip in two weeks', None),
                                         ('book for two weeks from now', None)])
def test_weeks_are_parsed_as_stay_length_only_when_they_mean_duration(config, text, nights):
    from datetime import date
    from hotel_chatbot.parsing import parse_message
    assert parse_message(text, date(2026, 10, 16), config).slots.get('nights') == nights


@pytest.mark.parametrize('text', ['can i store my surfboard', 'what are your covid rules', 'are you open during holidays'])
def test_low_confidence_guesses_ask_instead_of_acting(bot, service, text):
    reply = bot.respond(text)
    assert reply.startswith('I am not sure I understood')
    assert bot.mode is None and bot.awaiting is None and service.count() == 0


def test_confident_classifier_routes_still_act(bot):
    assert 'booking reference' in bot.respond('drop my reservation please')


def test_faq_retriever_finds_paraphrases(recognizer):
    assert FaqRetriever().match('how quick is the wireless internet')[0] == 'wifi'
    # Retrieval alone confuses shared words ("allowed"); the full pipeline's rules resolve it.
    assert recognizer.predict('is my puppy allowed to come').topic == 'pets'


def test_preprocessing_masks_entities():
    text = prepare("I'd like to book a Deluxe room on December 24 for 3 nights, alex@example.com BR-12345678AB")
    for token in ('xroom', 'xdate', 'xnum', 'xemail', 'xref', 'would'):
        assert token in text.split()


def test_bare_values_still_answer_the_awaited_field(bot):
    bot.respond('I want to make a reservation')
    for reply in ('suite', 'December 24', '3', '2', 'my name is Alex Rivera', 'alex@example.com'):
        bot.respond(reply)
    assert bot.draft.name == 'Alex Rivera' and bot.draft.nights == 3 and bot.draft.guests == 2
    assert bot.pending == 'book'


def test_held_out_accuracy_floor(recognizer):
    data = json.loads((ROOT / 'data' / 'independent_test.json').read_text(encoding='utf-8'))
    intents = data['intent_examples']
    hybrid = sum(recognizer.predict(e['text']).intent == e['intent'] for e in intents) / len(intents)
    classifier = sum(recognizer.predict_ml(e['text']).intent == e['intent'] for e in intents) / len(intents)
    faq = sum(recognizer.predict(e['text']).topic == e['topic'] for e in data['faq_examples']) / len(data['faq_examples'])
    assert hybrid >= .9 and classifier >= .85 and faq >= .9


def test_training_never_contains_evaluation_messages(recognizer):
    train = {e['text'].lower().strip() for e in recognizer.examples}
    for name in ('regression_holdout.json', 'independent_test.json'):
        data = json.loads((ROOT / 'data' / name).read_text(encoding='utf-8'))
        for key in ('examples', 'intent_examples', 'faq_examples'):
            assert not train & {e['text'].lower().strip() for e in data.get(key, [])}
