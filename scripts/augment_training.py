"""Append the MVP training-data expansion to data/intents.json (idempotent).

These examples were written fresh for varied register (formal, casual, short, typo-prone) and are
checked against both evaluation sets so no evaluation message is ever used for training.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'mvp_augmentation_2026_10'

NEW = {
 'book_room': [
  'i wanna book', 'book me in please', 'can i grab a room for friday', 'need a place to crash for 2 nights',
  'reserve something for me and my wife', 'room for one person please', 'I would like to stay with you in december',
  'sign me up for a deluxe', 'we want to stay 3 nights', 'can I make a booking for my parents',
  'lemme book a suite', 'booking for 2 adults and 2 kids', 'hold a room for me next friday',
  'i need accommodation for a business trip', 'I want to stay at your hotel', 'pls book standard room',
  'we are planning a vacation and need a room', 'get us a room for the holidays', 'book a room under my name',
  'i would like to reserve for the 15th', 'can you put us in a family room', 'start a new reservation',
 ],
 'check_availability': [
  'are there rooms left', 'do you have space for 4 people', 'anything available this saturday',
  'is the suite taken on the 10th', 'can you check if you have rooms', 'got any free rooms tomorrow night',
  'is there room for us on december 31', 'still accepting guests for christmas eve', 'any openings in january',
  'is everything booked up', 'is a family room open for 2 nights', 'what is available on friday',
  'can i check availability', 'do you have a deluxe free', 'are rooms open for new year',
  'check if a standard room is free tonight', 'any space left for the weekend', 'how many rooms are free',
 ],
 'check_price': [
  'how much', 'what does it cost', 'price please', 'how much for one night', 'rates for a family room',
  'what will i pay for 3 nights', 'is it expensive', 'what is the total for a week', 'cost per night',
  'how much do your rooms go for', 'can i get a price estimate', 'what is the room rate', 'how pricey is the suite',
  'tell me the price of a deluxe', 'what do you charge per night', 'whats the cost of staying 2 nights',
  'cheapest option', 'most affordable room', 'how much is the most expensive room',
 ],
 'cancel_booking': [
  'cancel', 'cancel please', 'i want to cancel', 'i no longer need the room', 'please scrap my reservation',
  'we wont be coming, cancel', 'cancel reservation BR-55AA66BB', 'my trip got cancelled so cancel my booking',
  'i have to cancel my stay', 'get rid of my booking', 'terminate my reservation', 'remove my booking',
  'i changed my mind and dont want the room anymore', 'abort my reservation', 'how do i cancel my booking',
  'not coming anymore please cancel', 'cancel my stay for next week', 'i need a cancellation',
 ],
 'modify_booking': [
  'change my booking', 'i need to change my dates', 'can i extend my stay', 'add one more night please',
  'make it 4 guests instead', 'upgrade me to a suite', 'shorten my stay by a night', 'switch to a different room',
  'can i edit my reservation', 'change the number of guests', 'move my check in to the 20th', 'update my reservation dates',
  'i want to stay longer', 'push my booking by one day', 'reschedule please', 'change my room type',
  'modify reservation BR-77CC88DD', 'can i change my check-out date',
 ],
 'greeting': [
  'hey', 'hi there', 'hello hello', 'good day', 'hiii', 'helloo', 'yo', 'sup', 'hey assistant', 'hello, is anyone there',
  'greetings', 'hi, i have a question', 'hello, can you help me', 'good afternoon!', 'hey how are you', 'kumusta',
 ],
 'goodbye': [
  'bye now', 'see ya', 'goodbye and take care', 'i am leaving now', 'later', 'that is all, bye', 'have a good day, bye',
  'im done', 'i will go now', 'ok that is everything, goodbye', 'see you soon', 'take care', 'peace out', 'farewell then',
 ],
 'thanks': [
  'thanks a lot', 'thank u', 'ty', 'thx', 'many thanks', 'thanks for the help', 'appreciate it', 'perfect, thank you',
  'awesome thanks', 'great help', 'thank you very much', 'that helps a lot', 'cool thanks', 'salamat',
 ],
 'checkin_checkout_time': [
  'check in time', 'checkout time', 'when is check in', 'what time is checkout', 'when can i get my room',
  'what time can we arrive', 'until when can i stay on my last day', 'when do i need to leave', 'check-in hours',
  'what time do i have to check out', 'earliest check in', 'latest check out time', 'what time does check in start',
  'can i arrive at night', 'is late arrival ok',
 ],
 'amenities_inquiry': [
  'what amenities do you have', 'hotel facilities', 'do you have a pool', 'is there wifi', 'is breakfast included',
  'do you have parking', 'is there a gym', 'what is included in the room rate', 'what services do you offer',
  'do you have a spa', 'is there a restaurant', 'any facilities for kids', 'do rooms have a tv', 'is there a bar',
  'do you allow pets', 'is there room service', 'what does the hotel offer', 'can i use the pool', 'is there free breakfast',
 ],
 'location_inquiry': [
  'where are you', 'location please', 'whats the address', 'how far are you from the airport', 'which area is the hotel in',
  'are you in manila', 'how do i get to the hotel', 'what is the nearest mall', 'is it walkable to restaurants',
  'send me the address', 'where is the hotel located', 'how far is the city center', 'what is around the hotel',
  'directions to the hotel please', 'is there a train station near you',
 ],
 'payment_inquiry': [
  'how do i pay', 'payment methods', 'do you take cash', 'can i pay online', 'is gcash accepted', 'do you take visa',
  'when do i pay', 'do i pay now or later', 'is a deposit required', 'can i pay at the hotel', 'mastercard ok?',
  'do you accept paymaya', 'do i need a credit card to book', 'can i pay in installments', 'pay on arrival possible?',
 ],
 'cancellation_policy': [
  'cancellation policy', 'can i cancel for free', 'is there a cancellation fee', 'what if i need to cancel later',
  'do you refund if i cancel', 'how late can i cancel', 'is my booking refundable', 'what happens if i do not show up',
  'free cancellation?', 'cancellation terms', 'are there penalties for cancelling', 'refund policy',
 ],
 'view_booking': [
  'show my booking', 'check my reservation', 'what is my reservation status', 'look up booking BR-99EE00FF',
  'can i see my reservation', 'view my booking details', 'is my booking confirmed', 'find my booking',
  'what did i book', 'reservation status please', 'show me my reservation details', 'retrieve my booking',
 ],
 'complaint': [
  'my room is dirty', 'the ac is not working', 'the toilet is broken', 'it is too noisy', 'i am not happy',
  'terrible service', 'i want to complain', 'there are bugs in my room', 'the shower has no hot water',
  'the wifi keeps disconnecting', 'staff ignored me', 'this is unacceptable', 'my room smells bad', 'very disappointed with my stay',
 ],
 'request_human_agent': [
  'talk to a person', 'agent please', 'human please', 'can i speak to someone', 'i need a real human',
  'transfer me to reception', 'let me talk to the manager', 'call me please', 'connect me with staff',
  'i want to speak to customer service', 'is there a live person', 'speak to front desk',
 ],
 'room_types_inquiry': [
  'what rooms do you have', 'room types', 'show me the rooms', 'what kind of rooms are available', 'tell me about the deluxe room',
  'how big is the suite', 'which room is best for a couple', 'what room is good for 4 people', 'difference between rooms',
  'how many people fit in a standard room', 'do you have rooms for families', 'what is the largest room', 'list your rooms',
  'room options please', 'what beds do the rooms have',
 ],
 'out_of_scope': [
  'what is the weather tomorrow', 'tell me a story', 'who is the president', 'can you do my homework', 'what is bitcoin',
  'recommend a movie', 'how old are you', 'sing a song', 'what is 5 times 7', 'write an essay for me', 'what is love',
  'how do i lose weight', 'who will win the election', 'explain quantum physics', 'book a flight for me',
 ],
}


def main():
    path = ROOT / 'data' / 'intents.json'
    corpus = json.loads(path.read_text(encoding='utf-8'))
    held_out = set()
    for name in ('regression_holdout.json', 'independent_test.json'):
        data = json.loads((ROOT / 'data' / name).read_text(encoding='utf-8'))
        for key in ('examples', 'intent_examples', 'faq_examples'):
            held_out.update(e['text'].lower().strip() for e in data.get(key, []))
    corpus['examples'] = [e for e in corpus['examples'] if e.get('source') != SOURCE]
    existing = {e['text'].lower().strip() for e in corpus['examples']}
    added = 0
    for intent, texts in NEW.items():
        for text in texts:
            key = text.lower().strip()
            if key in held_out:
                raise SystemExit(f'Refusing to train on an evaluation message: {text!r}')
            if key not in existing:
                corpus['examples'].append({'text': text, 'intent': intent, 'source': SOURCE})
                existing.add(key)
                added += 1
    path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Added {added} examples; corpus now {len(corpus["examples"])}.')


if __name__ == '__main__':
    main()
