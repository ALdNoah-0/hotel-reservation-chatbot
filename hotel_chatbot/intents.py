"""Prioritized explicit requests, FAQ retrieval, and a local word/character NLP classifier."""
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import ROOT
from .hotel_info import TOPIC_INTENTS, FaqRetriever, matching_topics
from .models import Prediction

# Abstention settings chosen on out-of-fold training probabilities with utility
# +1 correct / 0 abstain / -1 wrong (see scripts/evaluate.py and EVALUATION.md).
THRESHOLD = .175
MARGIN = .02
# FAQ retrieval cosine similarity: strong matches override weak classifier guesses.
FAQ_STRONG = .55
FAQ_WEAK = .35
INFORMATIONAL = {'amenities_inquiry','location_inquiry','checkin_checkout_time','room_types_inquiry','payment_inquiry','cancellation_policy'}

CONTRACTIONS = [(r"\bcan't\b",'cannot'),(r"\bwon't\b",'will not'),(r"n't\b",' not'),(r"'re\b",' are'),(r"'m\b",' am'),
    (r"'ll\b",' will'),(r"'ve\b",' have'),(r"'d\b",' would'),(r"'s\b",' is'),(r'\bgonna\b','going to'),(r'\bwanna\b','want to'),
    (r'\bgotta\b','got to'),(r'\bu\b','you'),(r'\bpls\b|\bplz\b','please'),(r'\bthx\b|\bty\b','thanks')]
MONTHS = ('january|february|march|april|may|june|july|august|september|october|november|december|'
          'jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec')
# Entity masking: the classifier learns "book <room> for <date>" instead of memorising specific values.
MASKS = [
    (r'[\w.+-]+@[\w.-]+\.[a-z]{2,}', ' xemail '),
    (r'\bbr-[a-z0-9]{5,20}\b', ' xref '),
    (r'\b\d{4}-\d{2}-\d{2}\b', ' xdate '),
    (rf'\b(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b', ' xdate '),
    (rf'\b(?:the\s+)?\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{MONTHS})\b', ' xdate '),
    (r'\b(?:next|this)?\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', ' xdate '),
    (r'\b(?:today|tonight|tomorrow)\b', ' xdate '),
    (r'\bthe\s+\d{1,2}(?:st|nd|rd|th)\b', ' xdate '),
    (r'\b(?:standard|deluxe|suite|family room)s?\b', ' xroom '),
    (r'\b\d+\b|\b(?:two|three|four|five|six|seven|eight|nine|ten)\b', ' xnum '),
]


def normalize(text):
    t = text.lower().replace('’',"'")
    for source,target in {"don't":'do not',"can't":'cannot',"won't":'will not',"isn't":'is not',"what's":'what is',"i'd":'i would'}.items():
        t = t.replace(source,target)
    return re.sub(r'\s+',' ',t).strip()


def prepare(text):
    """Classifier preprocessing: normalise, expand contractions/slang, mask entities, strip punctuation."""
    t = normalize(text)
    for pattern,target in CONTRACTIONS:
        t = re.sub(pattern,target,t)
    for pattern,target in MASKS:
        t = re.sub(pattern,target,t)
    t = re.sub(r'[^\w\s]',' ',t)
    return re.sub(r'\s+',' ',t).strip()


class IntentRecognizer:
    def __init__(self,path=None):
        corpus = json.loads((path or ROOT / 'data' / 'intents.json').read_text(encoding='utf-8'))
        self.examples = corpus['examples']
        self.pipeline = Pipeline([
            ('features',FeatureUnion([
                ('word',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),
                ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,sublinear_tf=True)),
            ],transformer_weights={'word':1.0,'char':0.7})),
            ('classifier',LogisticRegression(C=10,max_iter=3000,class_weight='balanced',random_state=42)),
        ])
        self.pipeline.fit([prepare(e['text']) for e in self.examples],[e['intent'] for e in self.examples])
        self.faq = FaqRetriever()

    def rank(self,text,k=3):
        """Top-k classifier intents with probabilities, for diagnostics."""
        t = prepare(text)
        if not t:
            return []
        scores = self.pipeline.predict_proba([t])[0]
        return [(str(self.pipeline.classes_[i]),float(scores[i])) for i in scores.argsort()[::-1][:k]]

    def predict_ml(self,text):
        t = prepare(text)
        word = self.pipeline.named_steps['features'].transformer_list[0][1]
        if not t or word.transform([t]).nnz == 0:
            return Prediction('fallback',0.0,'uncertain')
        scores = self.pipeline.predict_proba([t])[0]
        ranked = scores.argsort()
        best,runner = int(ranked[-1]),int(ranked[-2])
        score = float(scores[best])
        if score < THRESHOLD or score-float(scores[runner]) < MARGIN:
            return Prediction('fallback',score,'uncertain')
        return Prediction(str(self.pipeline.classes_[best]),score,'classifier')

    def predict_faq_or_ml(self,text):
        """Classifier, with FAQ retrieval taking over for hotel-information questions."""
        ml = self.predict_ml(text)
        topic,similarity = self.faq.match(normalize(text))
        faq = Prediction(TOPIC_INTENTS.get(topic,'amenities_inquiry'),similarity,'faq retrieval',topic)
        if similarity >= FAQ_STRONG and (ml.intent in INFORMATIONAL | {'fallback','out_of_scope'} or ml.score < .5):
            return faq
        if similarity >= FAQ_WEAK and ml.intent == 'fallback':
            return faq
        return ml

    def predict(self,text):
        t = normalize(text)
        def has(pattern):
            return bool(re.search(pattern,t))
        def result(intent,topic=None):
            # Rule match strength; not a calibrated probability of correctness.
            return Prediction(intent,1.0,'explicit rule',topic)
        if has(r'\b(homework|assignment|weather|joke|poem|president|stock market|investing|football|basketball|capital of|programming language|bake|recipe|math|chemistry|guitar|birthday gift|translate.*sentence)\b|\b(?:router|wifi|internet)\b.*\b(?:at home|my house)\b'):
            return result('out_of_scope')
        if has(r'\b(human|real person|actual (?:person|staff)|live (?:agent|representative|person)|manager|staff member|customer (?:service|support))\b|\b(?:talk|speak|call|contact|connect|transfer|reach|get|put)\b.{0,25}\b(?:reception|front desk)\b|^(?:reception|front desk)[?.! ]*$|\bwant (?:a )?person\b|\b(?:bypass|around|instead of|do not want).{0,25}\b(?:automated|automation|bot)\b'):
            return result('request_human_agent')
        if not has(r'^(?:is|are|how)\b.{0,30}\b(?:noisy|noise|quiet|loud)\b') and has(r'\b(dirty|unclean|complaint|complain|unhappy|disappointed|overcharged|rude|noise|noisy|broken|not working|does not work|no hot water|bad smell|smells|bugs|cockroach\w*|unacceptable|terrible|housekeeping.*(?:never|missed)|nothing but trouble|not thrilled|not what i (?:expected|signed up for))\b|\b(?:not|has not|hasn.t).{0,15}\bgood experience\b'):
            return result('complaint')
        negative_booking = has(r'\b(?:do not|never|not ready to|not going to)\b.{0,30}\b(?:book|reserve|confirm|cancel|modify|change)\b')
        cancel = has(r'\b(cancel|void|withdraw|undo)\b|\b(?:call off|back out|pull out)\b|\b(?:cannot|will not).{0,20}\b(?:make|take).{0,15}\b(?:trip|stay)\b')
        booking = has(r'\b(book|reserve)\b|\bmake (?:a |another )?(?:booking|reservation)\b|\b(?:need|want|like).{0,20}\b(?:accommodation|lodging|a room|stay at)\b|\b(?:arrange|set .{0,10}up|secure|lock in|put me down|looking).{0,40}\b(?:stay|place|spot|trip|room)\b')
        modify = has(r'\b(modify|reschedule|extend|shorten|upgrade|alter|adjust|switch|swap|rework|rearrange|shuffle|tweak|edit)\b|\b(?:change|move|update|push).{0,35}\b(?:booking|reservation|stay|trip|date|check[ -]?in|room|guest)\b')
        topics = matching_topics(t)
        # "How many nights can I book?" asks about the stay limit; it does not start a booking.
        if 'length_of_stay' in topics and not cancel and not modify and not negative_booking:
            return result('checkin_checkout_time','length_of_stay')
        # "Can I book a massage?" books a hotel service, not a room.
        if booking and not cancel and not modify and topics and not has(r'\b(rooms?|suites?|stay|nights?|accommodation|lodging|reservation|booking|standard|deluxe|family)\b'):
            return result(TOPIC_INTENTS.get(topics[0],'amenities_inquiry'),topics[0])
        if negative_booking or sum((cancel,booking,modify))>1:
            return result('fallback')
        if not booking and not modify and has(r'\b(cancellation|cancelling|cancel|refunds?|refundable)\b.*\b(policy|policies|rules|fees?|charged?|penalt\w*|free|terms)\b|\b(policy|policies|rules|terms)\b.*\b(cancel|cancellation|refunds?)\b|\bwhat (?:happens|if).{0,20}\b(?:cancel|no[- ]show|do not show up)\b|\b(?:refunds?|refundable|no[- ]show)\b|\bhow late can i cancel\b|\bcancel for free\b|\bfree cancellation\b'):
            return result('cancellation_policy')
        if not booking and not modify and not cancel:
            if has(r'\b(?:check|show|view|find|look up|lookup|retrieve|pull up|see)\b.{0,35}\b(?:booking|reservation)\b|\b(?:booking|reservation)\b.{0,25}\b(?:status|still active|confirmed)\b|\bwhat did i book\b'):
                return result('view_booking')
            if topics:
                return result('checkin_checkout_time' if topics[0]=='early_late' else TOPIC_INTENTS.get(topics[0],'amenities_inquiry'),topics[0])
        if cancel:
            return result('cancel_booking')
        if modify:
            return result('modify_booking')
        if not booking and has(r'\bhow many (?:people|guests|persons|adults)\b.{0,40}\b(?:fit|stay|sleep|room|suite)\b|\b(?:room|rooms|space) for (?:\d+|two|three|four|five|six|seven|eight) (?:people|guests|persons|adults)\b|\b(?:max(?:imum)?|room) (?:capacity|occupancy)\b|\b(?:biggest|largest|smallest|best) room\b|\bwhich room\b|\broom (?:size|sizes)\b|\b(?:two|2|twin|double|separate|king|queen)[ -](?:size )?beds?\b|\bbed (?:type|size|configuration)s?\b|\bconnecting rooms?\b|\bcheapest to (?:nicest|most expensive|priciest)\b'):
            return result('room_types_inquiry')
        if has(r'\b(price|prices|pricing|rates?|cost|quote|tariff|expensive|pricey|cheap|cheapest|affordable|discount|how much|set me back|gonna run|total|pay for|budget|ballpark)\b'):
            return result('check_price')
        if has(r'\b(payment|pay|cash|gcash|paymaya|maya|paypal|bitcoin|crypto\w*|dollars?|usd|euros?|receipts?|invoices?|deposit|credit cards?|debit cards?|cards?|visa|mastercard|amex|e-?wallets?|bank transfer|settle.*bill|money up ?front|owe|square up)\b'):
            return result('payment_inquiry')
        if has(r'\b(?:what|when|early|late|earliest|latest|midnight|hours|cutoff).{0,50}\b(?:check[ -]?in\b|check[ -]?out\b|checkout\b|checkin\b|arriv\w*|leave\b|depart\w*|show up\b|get into\b|be out\b)|\bcheck[ -]?in.*check[ -]?out\b|^(?:check[ -]?in|check[ -]?out|checkout|checkin)(?: time| hours)?[?.! ]*$'):
            return result('checkin_checkout_time')
        if has(r'\b(?:what.{0,20}(?:included|comes with)|what comes with).{0,30}\b(?:stay|room|bed)\b|\b(?:bite|food|eat).{0,20}\b(?:property|hotel)\b'):
            return result('amenities_inquiry')
        if has(r'\b(?:what|which|list|describe|compare|difference|categories|types|options|catalog|kinds?).{0,35}\b(?:rooms?|suite|accommodation)\b|\broom (?:types|categories|options|catalog)\b|\b(?:on offer|fancier|basic option|bigger group|cheapest to nicest)\b'):
            return result('room_types_inquiry')
        if has(r'\b(available|availability|vacancy|vacancies|vacant|vacancies|fully booked|sold out|unoccupied|openings?)\b|\b(?:rooms?|suite|anything|space).{0,25}\b(?:free|open|left)\b|\b(?:any|got|have).{0,25}\b(?:open rooms?|space|openings)\b'):
            return result('check_availability')
        if booking:
            return result('book_room')
        if has(r'\b(address|located|location|directions|airport|nearby|neighbourhood|beach|downtown|landmark|part of town|around the area)\b|\bwhere.*hotel\b|\b(?:get there|get to you|close to anything)\b'):
            return result('location_inquiry')
        if has(r'\b(amenities|facilities|wifi|wi-fi|internet|pool|swimming|gym|fitness|breakfast|parking|spa|pets?|laundry|restaurant|minibar|room service|smoking|balcony|accessible)\b'):
            return result('amenities_inquiry')
        if has(r'\b(bye|goodbye|farewell|see (?:you|ya)|gotta (?:go|run)|got to go|have to go|done for now|until next time|aight.*out|head off|signing off|take care)\b'):
            return result('goodbye')
        if has(r'\b(thanks?|thank you|thank u|thx|ty|salamat|appreciat\w*|grateful|much obliged|cheers|exactly what i needed)\b'):
            return result('thanks')
        if has(r'^(?:hi|hii+|hello+|hey|heya|hiya|howdy|greetings|yo|sup|kumusta|good (?:morning|afternoon|evening|day)|morning|afternoon|what is up|hope you.re doing well)\b'):
            return result('greeting')
        return self.predict_faq_or_ml(text)
