"""Conservative slot parsing: ambiguity is a clarification, never a guess."""
import calendar
import re
from datetime import date, timedelta

from .models import ParsedMessage

WORDS = {'a':1,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
         'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,
         'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,
         'nineteen':19,'twenty':20,'thirty':30,'zero':0}
NUMBER = r'(?:-?\d+|' + '|'.join(WORDS) + r')'
EMAIL = re.compile(r'(?<![\w.+-])([\w.+-]+@[\w.-]+\.[a-zA-Z]{2,})(?![\w.-])')
BOOKING_ID = re.compile(r'\bBR-[A-Z0-9]{5,20}\b',re.I)
MONTH_MAP = {name.lower():i for i,name in enumerate(calendar.month_name) if name}
MONTH_MAP.update({name.lower():i for i,name in enumerate(calendar.month_abbr) if name})
MONTH_NAMES = '|'.join(sorted(MONTH_MAP,key=len,reverse=True))
MONTH_DATE = re.compile(rf'\b({MONTH_NAMES})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b',re.I)
DAY_MONTH = re.compile(rf'\b(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH_NAMES})\.?\b(?:,?\s+(\d{{4}}))?',re.I)
WEEKDAYS = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']


def _number(value):
    return WORDS[value.lower()] if value.lower() in WORDS else int(value)


def parse_message(text, today, config):
    result = ParsedMessage()
    t = text.lower().replace('’',"'")
    slots, errors = result.slots, result.errors
    room_count = re.search(rf'(?<!\w)({NUMBER})\s+rooms?\b',t)
    if room_count and _number(room_count.group(1)) != 1:
        errors.append('This demo supports one room per reservation. Please book each room separately.')
    unknown = re.search(r'\b(presidential suite|junior suite|executive room|twin room|single room|double room)\b',t)
    if unknown:
        errors.append('That room category is not configured. Choose ' + ', '.join(config['rooms']) + '.')
    rooms = [key for key in config['rooms'] if re.search(r'\b'+re.escape(key)+r'\b',t)]
    if len(rooms) > 1:
        errors.append('Please choose one room category for this reservation.')
    elif rooms and not unknown:
        slots['room_type'] = rooms[0]

    night_values = [_number(m.group(1)) for m in re.finditer(rf'(?<!\w)({NUMBER})\s+nights?\b',t)]
    # "a week", "2 weeks", "two weeks" -> 7 nights each; "next week"/"this week" stay a date-range error below.
    # "in two weeks" / "2 weeks from now" describe when, not how long, so they are not stay lengths.
    night_values += [7*_number(m.group(1)) for m in re.finditer(rf'(?<!\w)(?<!in )({NUMBER})\s+weeks?\b(?!\s+(?:from now|away|ago|later|time))',t)]
    if len(set(night_values)) > 1:
        errors.append('You gave different lengths of stay. How many nights do you want?')
    elif night_values:
        if not 1 <= night_values[0] <= 30:
            errors.append('A stay must be 1 to 30 nights.')
        else:
            slots['nights'] = night_values[0]

    counts = list(re.finditer(rf'(?<!\w)({NUMBER})\s+(guests?|people|persons?|adults?|children|kids?|child)\b',t))
    if counts:
        groups = [(m.group(2),_number(m.group(1))) for m in counts]
        if any(n < 0 for _,n in groups):
            errors.append('Guest counts cannot be negative.')
        else:
            generic = [n for label,n in groups if label in {'guest','guests','people','person','persons'}]
            specific = [n for label,n in groups if label not in {'guest','guests','people','person','persons'}]
            total = sum(specific) if specific else (generic[0] if generic else 0)
            if len(set(generic)) > 1 or (generic and specific and generic[0] != total):
                errors.append('Please give one total number of guests, including children.')
            elif total < 1:
                errors.append('At least one guest is required.')
            else:
                slots['guests'] = total
    else:
        family = re.search(rf'\bfamily of ({NUMBER})\b',t)
        if family:
            slots['guests'] = _number(family.group(1))
        elif re.search(r'\b(?:a couple|two of us)\b',t):
            slots['guests'] = 2
        elif re.search(r'\b(?:just myself|myself|just me|solo)\b',t):
            slots['guests'] = 1

    if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',t):
        errors.append('That numeric date is ambiguous. Use YYYY-MM-DD or a month name, for example March 4, 2027.')
    if re.search(r'\b(?:(?:next|this)\s+(?:week|month|weekend)|this coming\s+\w+)\b',t):
        errors.append('Please give an exact check-in date and number of nights for that date range.')

    found = []
    spans = []
    def add(start,end,value):
        if any(start < b and end > a for a,b in spans):
            return
        found.append((start,end,value))
        spans.append((start,end))

    for m in re.finditer(r'\b(\d{4})-(\d{2})-(\d{2})\b',t):
        try:
            add(m.start(),m.end(),date(*(int(v) for v in m.groups())))
        except ValueError:
            errors.append('That date does not exist. Please enter a valid date.')
    for pattern,day_first in ((MONTH_DATE,False),(DAY_MONTH,True)):
        for m in pattern.finditer(t):
            month = MONTH_MAP[m.group(2 if day_first else 1).lower()]
            day = int(m.group(1 if day_first else 2))
            year = int(m.group(3)) if m.group(3) else today.year
            try:
                value = date(year,month,day)
                if not m.group(3) and value < today:
                    value = date(year+1,month,day)
                add(m.start(),m.end(),value)
            except ValueError:
                errors.append('That calendar date is invalid. Please provide a valid date and year.')
    for m in re.finditer(r'\b(today|tonight|tomorrow)\b',t):
        add(m.start(),m.end(),today + timedelta(days=m.group(1)=='tomorrow'))
    for m in re.finditer(r'\b(?:(next|this)\s+)?('+'|'.join(WEEKDAYS)+r')\b',t):
        delta = (WEEKDAYS.index(m.group(2)) - today.weekday()) % 7
        # "next Friday" means the next occurrence after today, displayed before confirmation.
        if m.group(1) == 'next' and delta == 0:
            delta = 7
        add(m.start(),m.end(),today+timedelta(days=delta))
    found.sort()
    if len(found) > 2:
        errors.append('Please give exactly one check-in and one check-out date.')
    for index,(start,end,value) in enumerate(found[:2]):
        prefix = t[found[index-1][1] if index else max(0,start-60):start]
        label = re.search(r'\b(check[ -]?in|arrival|arriv(?:e|ing)|check[ -]?out|departure|depart|leav(?:e|ing))\s*(?:date\s*)?(?:(?:on|to|is|at)\s*)?$',prefix)
        if label:
            field = 'checkin' if re.fullmatch(r'check[ -]?in|arrival|arriv(?:e|ing)',label.group(1)) else 'checkout'
        elif re.search(r'\b(?:to|until)\s*$',prefix):
            field = 'checkout'
        else:
            field = 'checkin' if 'checkin' not in slots else 'checkout'
        if field in slots and slots[field]!=value:
            errors.append(f'You gave conflicting {field} dates. Please clarify.')
        else:
            slots[field] = value

    email = EMAIL.search(text)
    if email:
        slots['email'] = email.group(1).lower()
    ref = BOOKING_ID.search(text)
    if ref:
        slots['booking_id'] = ref.group(0).upper()
    name = re.search(r"\b(?:my name is|name\s*:|under the name)\s+([^,;\n]+)",text,re.I)
    if name:
        value = re.split(r'\s+(?:and\s+)?(?:my\s+)?email\b|\s+[\w.+-]+@[\w.-]+',name.group(1),maxsplit=1,flags=re.I)[0]
        slots['name'] = value.strip().rstrip('.')
    return result
