"""Hotel FAQ answering: precise keyword rules, then TF-IDF retrieval over example questions."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# More specific questions precede general ones; at most two topics are answered per message.
TOPICS = {
    'breakfast_hours': ('breakfast', r'\bbreakfast\b'),
    'pool_hours': ('pool', r'\b(pool|swimming|swim)\b'),
    'gym_hours': ('fitness center', r'\b(gym|fitness)\b|\b(?:can|could|where|place to) (?:i |we )?(?:work ?out|exercise)\b|\b(?:work ?out|exercise) (?:room|area|equipment|facilit\w*)\b'),
    'pets': ('pet policy', r'\b(pets?|dogs?|doggy|puppy|puppies|cats?|kittens?|animals?|birds?|parrots?|rabbits?|bunny|hamsters?|guinea pigs?|ferrets?|'
                           r'goldfish|turtles?|tortoises?|snakes?|lizards?|iguanas?|reptiles?|alligators?|crocodiles?|monkeys?)\b|\bpet[ -]friendly\b'),
    'smoking': ('smoking policy', r'\b(smok\w*|vaping|vape|cigarettes?)\b'),
    'accessibility': ('accessibility', r'\b(accessib\w*|wheelchairs?|step[ -]?free|disabled|disability|mobility|elevators?|lifts?|ramps?)\b'),
    'housekeeping': ('housekeeping', r'\b(housekeeping|cleaned|cleaning|clean (?:my|the) room|turndown|(?:change|fresh|new|extra|more) (?:my )?(?:towels?|sheets?|linens?|pillows?|blankets?))\b'),
    'noise': ('noise and quiet hours', r'^(?:is|are|how)\b.{0,30}\b(?:noisy|noise|quiet|loud)\b|\bquiet hours\b|\bsound ?proof\w*'),
    'medical': ('medical help', r'\b(doctors?|medical|clinic|hospitals?|pharmac\w*|drugstore|first aid|nurse|medicines?)\b'),
    'feedback': ('feedback and reviews', r'\b(reviews?|feedback|suggestions?|compliments?)\b|\brate (?:my|your|the) (?:stay|hotel)\b'),
    'extra_beds': ('extra beds and cribs', r'\b(extra|additional|rollaway|roll-away|spare)\s+beds?\b|\b(cots?|cribs?)\b'),
    'children': ('children', r'\b(child|children|kids?|bab(?:y|ies)|infants?|toddlers?)\b.*\b(polic\w*|allowed|welcome|stay\w*|free|discount|charge|club|play)\b|\b(policy|rules|discounts?|fees?)\b.*\b(child|children|kids?|infants?)\b'),
    'room_service': ('room service', r'\broom service\b|\b(?:order|deliver\w*|bring)\b.{0,25}\b(?:food|meals?|dinner|breakfast|snacks?)\b.{0,20}\broom\b|\bfood\b.{0,20}\b(?:to|in) (?:my |the )?room\b|\bin[- ]room dining\b|\blate[- ]night (?:food|snacks?|meals?)\b|\bmidnight snacks?\b|\bfood at night\b|\bdeliver\w*\b.{0,25}\b(?:cake|flowers)\b|\b(?:cake|flowers)\b.{0,15}\bdeliver\w*'),
    'dietary': ('dietary options', r'\b(vegetarian|vegan|halal|kosher|gluten|allerg\w*|dietary|lactose)\b'),
    'outside_food': ('outside food', r'\b(?:bring|outside|own)\b.{0,20}\b(?:food|snacks?|drinks?|alcohol)\b|\b(?:food|snacks?|drinks?)\b.{0,15}\bfrom outside\b|\bdurian\b'),
    'restaurant': ('dining', r'\b(restaurants?|dining|dinner|lunch|somewhere to eat|place to eat|eat in the hotel|cafe|coffee shop)\b'),
    'bar': ('bar and lounge', r'\b(bar|lounge|cocktails?|happy hour|nightlife)\b'),
    'spa': ('spa', r'\b(spa|massages?|sauna|wellness|facials?)\b'),
    'laundry': ('laundry', r'\b(laundry|dry[ -]?cleaning|wash(?:ing)? (?:my )?clothes|ironing)\b'),
    'luggage': ('luggage storage', r'\b(luggage|baggage|bags?|suitcases?)\b.*\b(stor\w*|keep|leave|drop|hold|check[ -]?in|check[ -]?out|early|after)\b|\b(store|keep|leave|drop|hold)\b.*\b(luggage|baggage|bags?|suitcases?)\b'),
    'shuttle': ('airport transfer', r'\b(shuttle|transfers?|airport (?:transfer|transport|pickup|pick[ -]?up|service))\b|\bpick\w*\s*(?:me |us |guests )?up\b.{0,25}\bairport\b|\bairport\b.{0,25}\bpick\w*\s*up\b|\bairport\b.{0,20}\bdrop[ -]?off\b|\bdrop[ -]?off\b.{0,25}\bairport\b'),
    'visitors': ('visitor policy', r'\b(visitors?|visit(?:ing)? me|visiting guests?|outside guests?|friends?.{0,20}(?:visit|come over|come see|my room)|(?:come|coming) (?:see|visit) me)\b|\bfamily\b.{0,15}\bcome see me\b|\bstay overnight\b|\bsleep ?over\b|\bovernight guests?\b|\bpart(?:y|ies)\b'),
    'checkin_documents': ('check-in documents', r'\b(documents?|identification|valid id|passport|ids?)\b.*\b(check[ -]?in|need|bring|required?|show)\b|\b(bring|need|show|required?)\b.*\b(documents?|identification|passport|\bid\b)|\bminimum age\b|\bage requirements?\b|\bage (?:requirement|limit) (?:to|for) (?:check|book)|\bhow old\b.{0,25}\b(?:check|book)'),
    'early_late': ('early check-in and late check-out', r'\b(early|late|later|earlier)\b.*\b(check[ -]?in|check[ -]?out|checkout|checkin|arrival|departure|arrive)\b|\bcheck[ -]?in\b.{0,10}\b(?:at|by|before)\s+(?:[1-9]|1[01])\s*(?:am|a\.m\.|in the morning)\b|\barrive\b.{0,20}\b(?:[1-9]|1[01])\s*(?:am|in the morning)\b'),
    'length_of_stay': ('length of stay', r'\bhow (?:many|long)\b.{0,25}\b(?:can|could|may|allowed to|is the)\b.{0,15}\b(?:stay|book|reserve)\b|\bhow many (?:nights|days|weeks)\b.{0,25}\b(?:max(?:imum)?|min(?:imum)?|allowed|limit)\b|\b(?:max(?:imum)?|min(?:imum)?|longest|shortest)\b.{0,20}\b(?:stay|nights?|days?|length|booking)\b|\blimit\b.{0,30}\b(?:stay|nights?|days?)\b|\blength of (?:my |the |a )?stay\b|^(?:can|could|may) (?:i|we) stay\b.{0,25}\b(?:weeks?|months?|long-?term)\b|\blong[- ]term stays?\b'),
    'special_requests': ('special requests', r'\bspecial requests?\b|\b(?:can|could|may|how (?:do|can)) (?:i|we)\b.{0,15}\b(?:request|ask for)\b|\bquiet room\b'),
    'in_room': ('in-room amenities', r'\bwhat(?: is|\'s| comes| do you have)? in (?:the|my|a|each|your) rooms?\b|\b(?:in-?room|room) (?:amenities|features|facilities)\b|\b(?:amenities|features|facilities)\b.{0,12}\bin (?:the|my|a|each|your) rooms?\b|\b(tvs?|television|air[ -]?con\w*|aircon|minibars?|mini[ -]bars?|fridges?|refrigerators?|kettles?|hair ?dryers?|towels?|toiletr\w*|shampoo|soap|hot water|irons?|bathtubs?|bath tubs?|in[- ]room safe|safety deposit|safe in the room)\b'),
    'view_balcony': ('views and balconies', r'\b(balcon\w*|views?|high floor|top floor|window)\b'),
    'wifi': ('Wi-Fi', r'\b(wi-?fi|internet|wireless)\b'),
    'parking': ('parking', r'\b(parking|park (?:my|the|a) (?:car|motorbike|motorcycle|bike|bicycle)|bikes?|bicycles?|motorbikes?|motorcycles?|car park|garage|ev charg\w*|electric (?:car|vehicle))\b'),
    'business': ('business and meetings', r'\b(business cent(?:er|re)|meeting rooms?|conference|host (?:a )?meeting|events? (?:space|venue|hall)|venues?|weddings?|function (?:room|hall)|banquet|printing|printer)\b'),
    'security': ('safety and security', r'\b(safe|safety|security|cctv|curfew|24[ -]?(?:hour|hr)s? (?:reception|front desk)|safe deposit box|lost (?:my )?(?:room )?(?:key|card|keycard|key card)|locked out)\b'),
    'contact': ('contact details', r'\b(phone|telephone|hotline|contact (?:number|details|info\w*|you|the hotel)|how (?:can|do) i (?:contact|reach|call)|call (?:you|the hotel)|your email|email address of the hotel|what is your email|languages?|speak (?:english|tagalog|filipino))\b'),
    'nearby': ('nearby places', r'\b(nearby|near (?:the hotel|by|here|you)|close by|around the (?:hotel|area)|tourist|attractions?|things to do|sightseeing|beach|malls?|shopping|atms?|money changer|sim cards?|convenience stores?|7-eleven|grocery|supermarket)\b|\bwhat is near\b'),
    'location': ('location', r'\b(address|located|location|directions|what city|which city|map)\b|\bwhere\b.{0,15}\b(?:hotel|you|are you)\b|\bhow (?:do|can) i (?:get|find)\b|\bhow far\b|\b(?:from|to) the airport\b'),
    'taxes_fees': ('taxes and fees', r'\b(tax|taxes|vat|resort fee|service (?:fee|charge)|hidden fees?|extra fees?|currency)\b'),
    'discounts': ('discounts', r'\b(discounts?|promos?|promo codes?|promotions?|(?:any|best|good|hot|special) deals?|coupons?|vouchers?|special offers?)\b'),
    'about_hotel': ('about the hotel', r'\b(?:tell me about|describe|what is|about) (?:the|this|your) hotel\b|\bhotel name\b|\bname of (?:the|this|your) hotel\b|\bhow many (?:floors|storeys|stories|rooms does)\b|\brooftop\b(?!.{0,10}\b(?:pool|bar)\b)'),
    'about_bot': ('what I can do', r'^(?:help|help me|menu|options)[?.! ]*$|\bwhat can you (?:do|help)\b|\bwho are you\b|\bare you (?:a |an )?(?:bot|robot|ai|human|real)\b|\bhow (?:do|does) (?:this|you) work\b|\bwhat do you do\b'),
}

# Example questions per topic train the retrieval model. They are worded differently from the
# regression and independent test sets, which are never used for fitting.
EXAMPLES = {
    'breakfast_hours': ['what time is breakfast', 'breakfast hours', 'when is breakfast served', 'what time does breakfast end'],
    'pool_hours': ['what time does the pool open', 'pool hours', 'when does the swimming pool close', 'is the pool open at night'],
    'gym_hours': ['gym hours', 'when is the gym open', 'is the fitness center open all day', 'can I work out early in the morning'],
    'pets': ['can I bring my pet', 'are dogs allowed', 'is the hotel pet friendly', 'can my cat stay', 'do you accept animals'],
    'smoking': ['can I smoke', 'is smoking allowed', 'where is the smoking area', 'are the rooms smoke free', 'can I vape in my room'],
    'accessibility': ['is the hotel wheelchair accessible', 'do you have accessible rooms', 'is there an elevator', 'facilities for disabled guests', 'step free access'],
    'housekeeping': ['how often are rooms cleaned', 'can I get fresh towels', 'is there daily housekeeping', 'can I have extra pillows'],
    'noise': ['is it noisy at night', 'are the rooms quiet', 'what are the quiet hours', 'are rooms soundproof'],
    'medical': ['is there a doctor', 'where is the nearest hospital', 'is there a pharmacy nearby', 'do you have first aid'],
    'feedback': ['how can I leave a review', 'I want to give feedback', 'where do I rate my stay', 'I have a suggestion'],
    'extra_beds': ['can I get an extra bed', 'do you have a crib', 'can you add a rollaway bed', 'extra mattress for my room', 'baby cot available'],
    'children': ['are children allowed', 'do kids stay free', 'child policy', 'is the hotel family friendly', 'do you charge for children'],
    'room_service': ['do you have room service', 'can I order food to my room', 'in room dining', 'can meals be delivered to the room'],
    'dietary': ['do you have vegetarian food', 'vegan options', 'is the food halal', 'I have a food allergy', 'gluten free meals'],
    'outside_food': ['can I bring my own food', 'outside food allowed', 'can I bring drinks to my room', 'can I eat takeout in my room'],
    'restaurant': ['is there a restaurant', 'where can I have dinner', 'do you serve lunch', 'what time does the restaurant open', 'place to eat inside the hotel'],
    'bar': ['is there a bar', 'where can I get a drink', 'do you have a lounge', 'cocktails at the hotel'],
    'spa': ['do you have a spa', 'can I book a massage', 'is there a sauna', 'wellness treatments'],
    'laundry': ['is there laundry service', 'can you do dry cleaning', 'can I wash my clothes', 'laundry room', 'ironing service'],
    'luggage': ['can I store my luggage', 'can I leave my bags after check out', 'luggage storage', 'hold my suitcase before check in'],
    'shuttle': ['do you have an airport shuttle', 'airport pickup', 'can you arrange airport transfer', 'transport from the airport'],
    'visitors': ['are visitors allowed', 'can friends visit my room', 'can I have guests over', 'visitor policy'],
    'checkin_documents': ['what documents do I need', 'do I need an id to check in', 'is a passport required', 'what should I bring for check in'],
    'length_of_stay': ['how many nights can I stay', 'what is the maximum length of stay', 'minimum number of nights', 'how long can I book for', 'long term stay'],
    'early_late': ['can I check in early', 'late check out', 'can I arrive in the morning', 'can I stay past noon', 'early arrival'],
    'special_requests': ['can I request a quiet room', 'how do I add a special request', 'can I ask for a high floor', 'add a note to my booking'],
    'in_room': ['what is in the room', 'do rooms have a tv', 'is there air conditioning', 'is there a minibar', 'do you provide towels', 'is there a hair dryer', 'does the room have a safe', 'room amenities'],
    'view_balcony': ['does the room have a balcony', 'room with a view', 'can I get a high floor', 'city view room'],
    'wifi': ['do you have wifi', 'is internet free', 'what is the wifi password', 'how fast is the wifi'],
    'parking': ['is there parking', 'is parking free', 'where can I park', 'do you have ev charging'],
    'business': ['do you have a business center', 'meeting rooms', 'can I print documents', 'conference room for a meeting'],
    'security': ['is the hotel safe', 'is there security', 'is there a curfew', 'is reception open 24 hours'],
    'contact': ['how can I contact you', 'what is your phone number', 'hotel email', 'how do I call reception'],
    'location': ['where is the hotel', 'what is your address', 'where are you located', 'how far is the airport', 'how do I get there'],
    'nearby': ['what is near the hotel', 'things to do nearby', 'tourist attractions near you', 'is there a mall nearby', 'is it near the beach'],
    'taxes_fees': ['are taxes included', 'are there extra fees', 'what currency do you use', 'is there a resort fee'],
    'discounts': ['do you have discounts', 'any promos', 'is there a promo code', 'special offers'],
    'about_hotel': ['tell me about the hotel', 'what is the hotel like', 'what is the name of the hotel', 'hotel overview'],
    'about_bot': ['what can you do', 'who are you', 'are you a bot', 'how can you help me', 'help'],
}

# FAQ topics that belong to a more specific intent than general hotel information.
TOPIC_INTENTS = {'location':'location_inquiry','nearby':'location_inquiry','early_late':'checkin_checkout_time','length_of_stay':'checkin_checkout_time'}
# Words that appear in almost every hotel question carry no topic evidence for retrieval
# ("can my cat stay" must not make "how long can I stay" look like a pet question).
DOMAIN_STOP_WORDS = ['stay', 'stays', 'staying', 'hotel', 'night', 'nights', 'day', 'days', 'want', 'like', 'need', 'please', 'get']


def matching_topics(text):
    return [key for key, (_, pattern) in TOPICS.items() if re.search(pattern, text, re.I)]


class FaqRetriever:
    """Nearest-neighbour retrieval over example questions with word + character TF-IDF."""

    def __init__(self):
        texts, self.keys = [], []
        for key, questions in EXAMPLES.items():
            for question in questions + [TOPICS[key][0]]:
                texts.append(question.lower())
                self.keys.append(key)
        self.vectorizer = FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), stop_words=sorted(ENGLISH_STOP_WORDS | set(DOMAIN_STOP_WORDS)), sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self.matrix = self.vectorizer.fit_transform(texts)
        self.word = self.vectorizer.transformer_list[0][1]
        self.word_matrix = self.word.transform(texts)

    def match(self, text):
        """Return (topic, cosine similarity) of the closest example sharing a content word.

        Character n-grams alone over-match function phrases ("do you accept ..."), so a candidate
        must share at least one non-stopword with the message.
        """
        vector = self.vectorizer.transform([text.lower()])
        # Each TF-IDF block is L2-normalised, so halving the dot product gives a cosine in [0, 1].
        scores = (self.matrix @ vector.T).toarray().ravel() / 2
        overlap = (self.word_matrix @ self.word.transform([text.lower()]).T).toarray().ravel() > 0
        scores = scores * overlap
        best = int(scores.argmax())
        return self.keys[best], float(scores[best])


def topic_answer(config, key):
    label = TOPICS[key][0]
    answer = config.get('hotel_faq', {}).get(key)
    if isinstance(answer, str) and answer.strip():
        return f'**{label[0].upper() + label[1:]}** — {answer.strip()}'
    contact = config.get('reception_contact')
    return (f'Information about {label} is not configured. Please contact reception'
            + (f' at {contact}' if contact else '') + ' for a verified answer.')


def configured_answer(config, text, topic=None):
    topics = matching_topics(text)[:2] or ([topic] if topic in TOPICS else [])
    if not topics:
        return None
    return '\n\n'.join(topic_answer(config, key) for key in topics)
