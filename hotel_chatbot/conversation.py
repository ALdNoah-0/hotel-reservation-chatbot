"""Session-local conversation state; service calls own all database mutations."""
import re
import uuid
from dataclasses import fields, replace
from datetime import date

from .config import money
from .intents import normalize
from .hotel_info import configured_answer
from .models import Draft, Prediction
from .parsing import parse_message

FALLBACK = ('I am not certain what you mean. I can help with booking a room, availability, rates, hotel information, '
            'or managing a reservation. Try “What rooms do you have?”, “Is breakfast included?”, or “Book a deluxe room for 2 nights from December 24 for 2 guests”.')
INFORMATIONAL_LABELS = {'cancellation_policy':'cancellation','amenities_inquiry':'hotel amenities','checkin_checkout_time':'check-in and check-out',
                        'room_types_inquiry':'our rooms','payment_inquiry':'payment','location_inquiry':'our location'}
INFORMATIONAL_INTENTS = set(INFORMATIONAL_LABELS)
CLARIFY_BELOW = .55
CLARIFY_EXAMPLES = {
    'book_room':('book a room','Book a deluxe room for 2 nights from December 24 for 2 guests'),
    'check_availability':('check room availability','Is a suite available on December 24 for 2 nights?'),
    'check_price':('get a price','How much is a deluxe room for 3 nights?'),
    'cancel_booking':('cancel a reservation','Cancel my booking'),
    'modify_booking':('change a reservation','Change my booking'),
    'view_booking':('look up a reservation','Check my reservation status'),
}
PROMPTS = {
    'room_type':'Which room category would you like: Standard, Deluxe, Family, or Suite?',
    'checkin':'What is your check-in date? Use YYYY-MM-DD or a month name, such as December 24, 2026.',
    'nights':'How many nights will you stay? You can also give an exact check-out date.',
    'guests':'How many guests are staying, including children?',
    'name':'What name should the reservation be under?',
    'email':'What is your email address? You will need it with your reference to manage the reservation.',
}


class HotelChatbot:
    def __init__(self,service,recognizer):
        self.service = service
        self.recognizer = recognizer
        self.config = service.config
        self.reset()

    def reset(self):
        self.draft = Draft()
        self.mode = None
        self.pending = None
        self.awaiting = None
        self.reference = None
        self.lookup_email = None
        self.verified_booking = None
        self.last_booking = None
        self.last_prediction = Prediction('greeting',1.0,'welcome')
        self.last_slots = {}
        self.last_text = ''
        self.request_id = uuid.uuid4().hex
        self.blocked = False
        self.revision = 0

    def summary(self):
        try:
            q = self.service.quote(self.draft)
        except ValueError as exc:
            return str(exc)
        room = self.config['rooms'][q.room_type]['label']
        notes = ('\n\nSpecial requests (not guaranteed): ' + self.draft.special_requests.strip()) if self.draft.special_requests.strip() else ''
        return (f'{room} · {q.checkin.isoformat()} → {q.checkout.isoformat()} · {q.nights} nights · {q.guests} guests\n\n'
                f'{money(q.nightly_rate)} per night × {q.nights} = **{money(q.total)}**. '
                + self.config['pricing_note'] + notes)

    def update_draft(self,**updates):
        self.pending = None
        self.blocked = False
        self.request_id = uuid.uuid4().hex
        allowed = {f.name for f in fields(Draft)}
        updates = {k:v for k,v in updates.items() if k in allowed}
        if 'nights' in updates and 'checkout' not in updates:
            updates['checkout'] = None
        if 'checkin' in updates and 'checkout' not in updates and self.draft.nights is not None:
            updates['checkout'] = None
        if 'checkout' in updates and updates['checkout'] is not None and 'nights' not in updates:
            updates['nights'] = None
        self.draft = replace(self.draft,**updates)
        self.revision += 1
        if self.mode is None:
            self.mode = 'book'
        return self._next()

    def _next(self):
        if self.blocked:
            return 'Please correct the unclear detail before confirming.'
        required = ['room_type','checkin','nights','guests']
        if self.mode in {'book','modify'}:
            required += ['name','email']
        for key in required:
            value = getattr(self.draft,key)
            if key == 'nights' and self.draft.checkout is not None:
                continue
            if value is None or value == '':
                self.awaiting = key
                return PROMPTS[key]
        self.awaiting = None
        try:
            q = self.service.quote(self.draft,require_identity=self.mode in {'book','modify'})
            free = self.service.available(q.room_type,q.checkin,q.checkout,self.reference if self.mode=='modify' else None)
        except ValueError as exc:
            self.pending = None
            return str(exc)
        if self.mode == 'price':
            return self.summary()
        if free == 0:
            self.pending = None
            return self.summary() + '\n\nNo room in this category is available for those dates. Try a different category or dates.'
        if self.mode == 'availability':
            return self.summary() + f'\n\n{free} rooms are currently available. Availability is checked again when a booking is confirmed. Say “book this room” to start a reservation.'
        self.pending = self.mode
        action = 'modification' if self.mode == 'modify' else 'booking'
        return self.summary() + f'\n\nPlease review your details, then select Confirm {action} or type “confirm {action}”. This is a local demo reservation.'

    def _verify_management(self):
        self.pending = None
        if not self.reference:
            self.awaiting = 'booking_id'
            return 'Please give your booking reference, such as BR-12345678AB.'
        if not self.lookup_email:
            self.awaiting = 'lookup_email'
            return 'What email address was used for that reservation?'
        row = self.service.lookup(self.reference,self.lookup_email)
        if not row:
            self.awaiting = 'lookup_email'
            return 'No matching reservation was found. Check the reference and email, then try again.'
        self.verified_booking = row
        self.awaiting = None
        if self.mode == 'view':
            self.mode = None
            notes = ('\n\nSpecial requests (not guaranteed): ' + row['special_requests']) if row['special_requests'] else ''
            return (f"Reservation **{row['reference']}** · {row['status'].title()}\n\n"
                    f"{row['room_type'].title()} · {row['checkin']} → {row['checkout']} · {row['nights']} nights · {row['guests']} guests\n\n"
                    f"Recorded total: **{money(row['total'])}**." + notes + '\n\nThis lookup did not change your reservation.')
        if row['status'] != 'active':
            return 'That reservation is already cancelled.'
        if self.mode == 'cancel':
            self.pending = 'cancel'
            return (f"Reservation {row['reference']}: {row['room_type'].title()}, {row['checkin']} → {row['checkout']}.\n\n"
                    + self.config['cancellation_policy'] + '\n\nType “confirm cancellation” or use Confirm cancellation to proceed.')
        self.draft = Draft(room_type=row['room_type'],checkin=date.fromisoformat(row['checkin']),
            checkout=date.fromisoformat(row['checkout']),nights=row['nights'],guests=row['guests'],name=row['name'],email=row['email'],special_requests=row['special_requests'])
        self.revision += 1
        return self.summary() + '\n\nWhat would you like to change: dates, nights, room category, guests, or special requests?'

    def prepare_management(self,action,reference,email):
        if action not in {'cancel','modify','view'}:
            raise ValueError('Unknown management action.')
        self.mode = action
        self.reference = reference
        self.lookup_email = email
        self.verified_booking = None
        self.pending = None
        self.blocked = False
        return self._verify_management()

    def confirm(self):
        if not self.pending or self.blocked:
            return 'There is no complete, reviewed action awaiting confirmation. Provide or correct the details first.'
        action = self.pending
        try:
            if action == 'book':
                saved = self.service.create(self.draft,self.request_id)
                reply = f"Your local demo reservation is saved. Reference: **{saved['reference']}**.\n\n" + self.summary()
            elif action == 'cancel':
                saved = self.service.cancel(self.reference,self.lookup_email)
                reply = f"Local demo reservation **{saved['reference']}** is cancelled. No refund was processed."
            else:
                saved = self.service.modify(self.reference,self.lookup_email,self.draft)
                reply = f"Local demo reservation **{saved['reference']}** is updated.\n\n" + self.summary()
        except ValueError as exc:
            self.pending = None
            return str(exc)
        self.last_booking = saved
        self.pending = None
        self.mode = None
        self.awaiting = None
        return reply

    def _contact(self):
        contact = self.config.get('reception_contact')
        return (f'You can reach reception at {contact}. This chat cannot transfer you to a live agent.' if contact
                else 'Reception contact details are not configured, and this chat cannot connect to hotel staff.')

    def _faq(self,intent,text):
        t = normalize(text)
        topic = self.last_prediction.topic
        if intent == 'cancellation_policy':
            return self.config['cancellation_policy']
        if intent in {'amenities_inquiry','checkin_checkout_time','location_inquiry'}:
            answer = configured_answer(self.config,t,topic)
            if answer:
                return answer
        if intent == 'request_human_agent':
            return self._contact()
        if intent == 'complaint':
            return ('I am sorry your stay has not been right, and thank you for telling us. Please let reception know so the team can fix it right away. '
                    + self._contact())
        if intent == 'checkin_checkout_time':
            return f"Check-in starts at {self.config['checkin_time']}, and check-out is by {self.config['checkout_time']}. Ask me about early check-in or late check-out if you need more time."
        if intent == 'room_types_inquiry':
            rooms = self.config['rooms'].values()
            slots = parse_message(text,self.service.today,self.config).slots
            guests = slots.get('guests')
            lead = ''
            if slots.get('room_type') and not guests:
                r = self.config['rooms'][slots['room_type']]
                return (f"**{r['label']}** — {r['description']} Up to {r['capacity']} guests; {money(r['nightly_rate'])}/night.\n\n"
                        'Other rooms: ' + ', '.join(f"{o['label']} (up to {o['capacity']}, {money(o['nightly_rate'])})"
                                                    for k,o in self.config['rooms'].items() if k != slots['room_type']) + '.')
            if guests:
                fitting = [r['label'] for r in rooms if r['capacity'] >= guests]
                lead = (f"For {guests} guests: {' or '.join(fitting)}.\n\n" if fitting
                        else f'No single room fits {guests} guests; the largest rooms take {max(r["capacity"] for r in rooms)}. Please book separate rooms.\n\n')
            return lead + '\n\n'.join(f"**{r['label']}** — {r['description']} Up to {r['capacity']} guests; {money(r['nightly_rate'])}/night." for r in rooms)
        if intent == 'payment_inquiry':
            return 'We accept ' + ', '.join(self.config['payment_methods']) + '. ' + self.config['payment_policy']
        if intent == 'location_inquiry':
            return configured_answer(self.config,'',topic or 'location') or (f"Address: {self.config['address']}." if self.config.get('address') else 'The hotel address is not configured.')
        if intent == 'amenities_inquiry':
            overview = 'Our amenities: ' + '; '.join(f"**{a['label']}** — {a['description']}" for a in self.config['amenities'].values())
            if re.search(r'\b(amenit\w*|facilit\w*|services?|offer|features|what do you have|what does the hotel have)\b',t):
                return overview + '\n\nAsk me about anything specific, such as pets, room service, or the airport transfer.'
            # A specific item with no configured answer: say so plainly rather than imply it exists.
            return ('I do not have information about that specific item, so I cannot confirm the hotel offers it. '
                    + overview + '\n\n' + self._contact().split(' This chat')[0])
        if intent == 'out_of_scope':
            return 'Sorry, I can only help with Hotel Becerel: rooms, rates, availability, reservations, and hotel information. Try “What rooms do you have?”'
        if intent == 'greeting':
            return f"Hello! Welcome to {self.config['name']}. I can help with rooms, rates, availability, reservations, and any question about the hotel. How can I help?"
        if intent == 'thanks':
            return 'You are welcome! Anything else I can help with?'
        if intent == 'goodbye':
            return 'Goodbye, and thank you for choosing Hotel Becerel! Your saved reservations stay on record.'
        return FALLBACK

    def _price(self,slots):
        candidate = replace(self.draft,**{k:v for k,v in slots.items() if k in Draft.__dataclass_fields__})
        room = self.config['rooms'].get(candidate.room_type)
        if not room:
            cheapest = min(self.config['rooms'].values(),key=lambda r: r['nightly_rate'])
            return ('\n\n'.join(f"{r['label']}: {money(r['nightly_rate'])}/night" for r in self.config['rooms'].values())
                    + f"\n\nThe most affordable is {cheapest['label']} at {money(cheapest['nightly_rate'])}/night. "
                    + 'Tell me a room and number of nights for a total.')
        nights = candidate.nights
        if nights is None and candidate.checkin and candidate.checkout:
            nights = (candidate.checkout-candidate.checkin).days
        if nights is not None and type(nights) is int and 1<=nights<=30:
            return f"{room['label']}: {money(room['nightly_rate'])}/night × {nights} nights = **{money(room['nightly_rate']*nights)}**. " + self.config['pricing_note']
        return f"{room['label']}: {money(room['nightly_rate'])}/night. How many nights? " + self.config['pricing_note']

    def _availability(self,slots):
        updates = {k:v for k,v in slots.items() if k in Draft.__dataclass_fields__}
        if ('checkin' in updates or 'nights' in updates) and 'checkout' not in updates and self.draft.nights is not None:
            updates['checkout'] = None
        if 'checkout' in updates and 'nights' not in updates:
            updates['nights'] = None
        candidate = replace(self.draft,**updates)
        try:
            quote = self.service.quote(candidate)
            free = self.service.available(quote.room_type,quote.checkin,quote.checkout,self.reference if self.mode=='modify' else None)
        except ValueError as exc:
            return f'For that availability check: {exc} Your current reservation draft is unchanged.'
        return (f"{self.config['rooms'][quote.room_type]['label']}: {free} rooms available for {quote.checkin} → {quote.checkout}. "
                'Your current reservation draft is unchanged. Inventory is checked again when you confirm a change.')

    def respond(self,text):
        text = text.strip()
        if not text:
            return FALLBACK
        if len(text)>2000:
            return 'Please keep each message under 2,000 characters.'
        t = normalize(text)
        # Only whole-message commands confirm; a sentence mentioning confirmation cannot trigger a write.
        if re.fullmatch(r'yes[.! ]*',t):
            return 'Please use the Confirm button or type the exact action, such as “confirm booking”, after reviewing its summary.'
        if re.fullmatch(r'(?:confirm|confirm (?:booking|reservation|cancellation|modification)|yes[, ]+confirm(?: (?:booking|reservation|cancellation|modification))?)[.! ]*',t):
            named = next((action for word,action in {'booking':'book','reservation':'book','cancellation':'cancel','modification':'modify'}.items() if word in t),None)
            if named and named!=self.pending:
                return 'That confirmation does not match the pending action. Please review its summary and use the matching Confirm button.'
            return self.confirm()
        if re.fullmatch(r'(?:no|no thanks|do not confirm(?: booking)?|do not proceed|never mind|nevermind|stop)[.! ]*',t):
            self.pending = None
            self.blocked = True
            return 'Nothing was saved or changed. Correct or review the draft to prepare a new confirmation.'
        if t in {'reset','start over','new conversation'}:
            self.reset()
            return 'Conversation reset. Saved reservations remain in the local database.'
        # Treat explicit notes as opaque text, never as dates, room choices, or commands.
        note = re.fullmatch(r'(?:add (?:a )?|set |update )?special requests?\s*:\s*(.*)',text,re.I|re.S)
        clear_note = bool(re.fullmatch(r'(?:clear|remove) (?:my )?special requests?[.! ]*',t))
        if note or clear_note:
            if self.mode in {'cancel','view'}:
                return 'Finish the current lookup or cancellation first. To change a saved note, review a reservation modification.'
            return self.update_draft(special_requests='' if clear_note else note.group(1).strip())
        if re.search(r'\b(?:do not|never|not going to|not ready to)\b.{0,30}\b(?:book|reserve|confirm|cancel|modify|change)\b',t):
            self.pending = None
            self.blocked = True
            return 'Nothing was saved or changed. Review or correct your details if you want to continue.'

        prediction = self.recognizer.predict(text)
        self.last_prediction = prediction
        if prediction.intent=='fallback' and prediction.source=='explicit rule':
            self.pending = None
            self.blocked = True
            return 'Your message contains conflicting or negated actions. Nothing was saved or changed. Please choose one action and review the details.'
        parsed = parse_message(text,self.service.today,self.config)
        slots = parsed.slots
        self.last_slots = dict(slots)
        self.last_text = text
        intent = prediction.intent
        # While a field is awaited, data-bearing replies beat statistical (non-rule) topic guesses,
        # e.g. "my name is Alex" must not retrieve the "hotel name" FAQ.
        if (self.awaiting and prediction.source != 'explicit rule' and self.mode in {'book','availability','price','modify','cancel','view'}
                and (set(slots) & set(Draft.__dataclass_fields__) | ({'booking_id','email'} & set(slots))
                     or self.awaiting=='name' and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,119}",text)
                     or self.awaiting in {'nights','guests'} and re.fullmatch(r'\d{1,2}',t))):
            intent = 'fallback'
        # Low-confidence classifier guesses (no rule fired, no stay details given) never start an action;
        # informational guesses are answered but flagged. Cut-off chosen from held-out/probe scores (EVALUATION.md).
        if (prediction.source == 'classifier' and prediction.score < CLARIFY_BELOW and intent != 'fallback' and not self.awaiting
                and not set(slots) & (set(Draft.__dataclass_fields__) | {'booking_id'})):
            if intent in CLARIFY_EXAMPLES:
                action, example = CLARIFY_EXAMPLES[intent]
                return (f'I am not sure I understood. Did you want to {action}? If so, say something like “{example}”. '
                        'Otherwise, please rephrase, or pick a topic under More questions you can ask.')
            if intent in INFORMATIONAL_INTENTS:
                return f'I think you are asking about {INFORMATIONAL_LABELS[intent]}; if not, please rephrase.\n\n' + self._faq(intent,text)
        # Informational questions may contain dates or room names, but do not edit a stay.
        if intent in INFORMATIONAL_INTENTS:
            return self._faq(intent,text)
        if self.mode in {'book','modify','availability','price'} and set(slots).intersection(Draft.__dataclass_fields__) and re.search(r'\b(?:not sure|unsure|uncertain|maybe|perhaps|do not want|not)\b',t):
            self.pending = None
            self.blocked = True
            return 'That stay detail sounds uncertain or negated. Nothing was changed. Please provide the intended detail or review the form before confirming.'
        # Bare values answer the specific field requested in the active flow.
        bare = False
        if self.awaiting in {'booking_id','lookup_email'} and ('booking_id' in slots or 'email' in slots):
            bare = True
        elif self.awaiting in {'room_type','checkin','nights','guests','name','email'} and self.mode in {'book','availability','price'}:
            if re.fullmatch(r'\d{1,2}',t) and self.awaiting in {'nights','guests'}:
                slots[self.awaiting] = int(t)
                bare = True
            # A short reply to a direct question is that value unless a rule recognised an explicit request.
            elif self.awaiting=='name' and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,119}",text) and (intent in {'fallback','greeting','book_room'} or prediction.source!='explicit rule'):
                slots.setdefault('name',text)
                bare = True
            elif self.awaiting in slots and len(text.split())<=8 and (intent in {'fallback','book_room','check_availability'} or prediction.source!='explicit rule'):
                bare = True
        detail_update = self.mode in {'book','availability','price','modify'} and bool(slots) and (
            bare or intent=='fallback' and len(text.split())<=12 or re.search(r'\b(?:make it|instead|change|actually|my name|my email|check[ -]?out)\b',t))
        if self.mode in {'cancel','modify','view'} and self.verified_booking is None:
            if slots.get('booking_id'):
                self.reference = slots['booking_id']
            if slots.get('email'):
                self.lookup_email = slots['email']
            if bare or intent in {'cancel_booking','modify_booking','view_booking'} or 'email' in slots or 'booking_id' in slots:
                return self._verify_management()

        if parsed.errors and (intent in {'book_room','check_availability','modify_booking'} or detail_update or self.mode is not None and re.search(r'\bchange\b',t)):
            self.pending = None
            self.blocked = True
            return ' '.join(parsed.errors)
        if intent in {'out_of_scope','request_human_agent','complaint'}:
            return self._faq(intent,text)
        if bare and detail_update:
            return self.update_draft(**slots)
        if intent == 'view_booking':
            return self.prepare_management('view',slots.get('booking_id'),slots.get('email'))
        if intent=='check_price':
            if parsed.errors:
                return ' '.join(parsed.errors)
            if self.mode is None:
                self.mode = 'price'
                self.draft = replace(self.draft,**{k:v for k,v in slots.items() if k in Draft.__dataclass_fields__})
                self.awaiting = 'nights' if self.draft.nights is None else None
            return self._price(slots)
        if intent=='check_availability' and self.mode in {'book','modify'}:
            return self._availability(slots)
        if intent in {'cancel_booking','modify_booking'} and not detail_update:
            return self.prepare_management('cancel' if intent=='cancel_booking' else 'modify',slots.get('booking_id'),slots.get('email'))
        if intent in {'book_room','check_availability'}:
            if self.mode in {'cancel','modify','view'}:
                self.draft = Draft()
                self.reference = None
                self.verified_booking = None
            self.mode = 'book' if intent=='book_room' else 'availability'
            return self.update_draft(**slots)
        if detail_update:
            return self.update_draft(**slots)
        if intent in {'greeting','goodbye','thanks','checkin_checkout_time','amenities_inquiry','room_types_inquiry','payment_inquiry','location_inquiry','cancellation_policy'}:
            return self._faq(intent,text)
        return FALLBACK
