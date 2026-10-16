"""A monochrome Streamlit concierge. Run: python -m streamlit run app.py."""
import os
from datetime import date, timedelta
from html import escape

import streamlit as st

from hotel_chatbot.config import ROOT, load_config, money
from hotel_chatbot.conversation import HotelChatbot
from hotel_chatbot.hotel_info import TOPICS
from hotel_chatbot.intents import IntentRecognizer, prepare
from hotel_chatbot.reservations import ReservationService

st.set_page_config(page_title='Hotel Becerel | Reservations',page_icon='⬛',layout='wide',initial_sidebar_state='collapsed')
st.html(ROOT / 'styles' / 'monochrome.css')


@st.cache_resource(show_spinner='Preparing your reservation assistant…')
def get_recognizer(corpus_modified,implementation_modified,faq_modified):
    return IntentRecognizer()


@st.cache_data
def get_config(config_modified):
    return load_config()


config = get_config((ROOT / 'data' / 'hotel.json').stat().st_mtime_ns)
welcome = (f"Hi! I’m your {config['name']} virtual assistant. I can answer questions about rooms, rates, and the hotel, "
           'check availability, and make, change, or cancel a reservation. Ask me anything, or pick a question below.')
if 'bot' not in st.session_state:
    service = ReservationService(os.environ.get('HOTEL_DB_PATH',str(ROOT / 'var' / 'reservations.sqlite3')),config)
    st.session_state['bot'] = HotelChatbot(service,get_recognizer((ROOT / 'data' / 'intents.json').stat().st_mtime_ns,
        (ROOT / 'hotel_chatbot' / 'intents.py').stat().st_mtime_ns,(ROOT / 'hotel_chatbot' / 'hotel_info.py').stat().st_mtime_ns))
    st.session_state['messages'] = [{'role':'assistant','content':welcome}]

bot = st.session_state['bot']
# Apply a name change to existing browser sessions without resetting their draft or records.
bot.config['name'] = config['name']
bot.service.config['name'] = config['name']
bot.config['hotel_faq'] = config.get('hotel_faq',{})
welcome_version = (config['name'],'mvp-v3')
if st.session_state.get('assistant_welcome_version') != welcome_version:
    if st.session_state['messages'] and st.session_state['messages'][0]['role']=='assistant':
        st.session_state['messages'][0]['content'] = welcome
    st.session_state['assistant_welcome_version'] = welcome_version


def add_reply(reply,user=None):
    if user:
        st.session_state['messages'].append({'role':'user','content':user})
    st.session_state['messages'].append({'role':'assistant','content':reply})


def ask_question(question):
    add_reply(bot.respond(question),user=question)


def section_marker(label,detail=''):
    st.html(f'<div class="hotel-section-marker"><span>{escape(label)}</span><span>{escape(detail)}</span></div>')


def date_label(value):
    return f'{value.day} {value.strftime("%b %Y")}'


def show_quote():
    if not (bot.draft.room_type and bot.draft.checkin and (bot.draft.nights or bot.draft.checkout) and bot.draft.guests):
        st.html('<div class="hotel-empty"><h3>Your itinerary starts here.</h3><p>Share your plans in chat, or add the details below.</p></div>')
        return
    try:
        quote = bot.service.quote(bot.draft)
    except ValueError as exc:
        st.warning(str(exc))
        return
    room = config['rooms'][quote.room_type]['label']
    st.html(f'''<section class="hotel-quote" aria-label="Stay quote">
      <div class="hotel-quote-head"><div class="hotel-kicker">Your stay</div><h3>{escape(room)}</h3></div>
      <dl><div><dt>Check-in</dt><dd>{date_label(quote.checkin)}</dd></div>
          <div><dt>Check-out</dt><dd>{date_label(quote.checkout)}</dd></div>
          <div><dt>Nights / guests</dt><dd>{quote.nights} nights · {quote.guests} guests</dd></div>
          <div><dt>Nightly rate</dt><dd>{money(quote.nightly_rate)}</dd></div></dl>
      <div class="hotel-quote-total"><span>Total for your stay</span><strong>{money(quote.total)}</strong></div>
    </section>''')
    st.caption(config['pricing_note'])
    if getattr(bot.draft,'special_requests',''):
        st.caption('Special requests · recorded as preferences, not guaranteed')
        st.text(bot.draft.special_requests)


METHOD_LABELS = {'explicit rule':'Rule (high-precision pattern)','classifier':'ML classifier (TF-IDF + logistic regression)',
                 'faq retrieval':'FAQ retrieval (TF-IDF cosine similarity)','uncertain':'Abstained (low confidence)','welcome':'—'}


def show_nlp_insights():
    if not getattr(bot,'last_text',''):
        st.caption('Send a message to see its intent, confidence, extracted details, and the classifier’s top guesses.')
        return
    p = bot.last_prediction
    st.caption(f'MESSAGE · “{bot.last_text[:200]}”')
    a,b = st.columns(2,gap='small')
    a.metric('Intent',p.intent.replace('_',' '))
    b.metric('Confidence' if p.source!='explicit rule' else 'Rule match',f'{p.score:.0%}')
    st.write(f'**Decided by:** {METHOD_LABELS.get(p.source,p.source)}')
    if p.topic:
        st.write(f'**FAQ topic:** {TOPICS[p.topic][0]}')
    st.write('**Preprocessed for the classifier:** `' + (prepare(bot.last_text) or '∅') + '`')
    ranking = bot.recognizer.rank(bot.last_text)
    if ranking:
        st.write('**Classifier top 3**')
        for intent,score in ranking:
            st.progress(min(max(score,0.0),1.0),text=f'{intent.replace("_"," ")} · {score:.0%}')
    entities = {k:(v.isoformat() if hasattr(v,'isoformat') else v) for k,v in bot.last_slots.items()}
    st.write('**Extracted details (entities):** ' + (', '.join(f'{k} = {v}' for k,v in entities.items()) if entities else 'none'))
    st.caption('Rules handle explicit, high-stakes requests; the classifier and retriever handle free-form wording. Scores route messages and are not guarantees.')


def begin_management(action,row):
    """Callbacks run before rendering, so the native tab state can be updated safely."""
    label = 'Review cancellation' if action=='cancel' else 'Change reservation details'
    add_reply(bot.prepare_management(action,row['reference'],row['email']),user=label)
    st.session_state['navigation'] = 'Reservation assistant'


with st.sidebar:
    st.header(config['name'])
    st.caption('LOCAL DEMO · RESERVATION DESK')
    st.write('A considered stay, from your first question to the final confirmation.')
    st.divider()
    st.subheader('How it works')
    st.write('**01** Share your dates and room preference.\n\n**02** Review the itinerary and total.\n\n**03** Confirm your local reservation.')
    st.divider()
    st.caption(f"CHECK-IN {config['checkin_time']}\n\nCHECK-OUT {config['checkout_time']}")
    if st.button('Start a new conversation',key='reset_conversation',width='stretch'):
        bot.reset()
        st.session_state['messages'] = [{'role':'assistant','content':welcome}]
        st.session_state.pop('managed_booking',None)
        st.session_state['navigation'] = 'Reservation assistant'
        st.rerun()
    with st.expander('Developer diagnostics'):
        diagnostics = st.toggle('Show NLP diagnostics',value=False)
        if diagnostics:
            st.caption(f'Intent: {bot.last_prediction.intent}\n\nMethod: {bot.last_prediction.source}\n\nScore: {bot.last_prediction.score:.3f}')
            st.caption('These scores describe routing; they are not guarantees of correctness.')
    st.caption('Reservations are stored locally. This fictional hotel demo does not collect payments or contact hotel staff.')

st.html(f'''<a class="hotel-skip" href="#reservation-assistant">Skip to reservation assistant</a>
  <div class="hotel-masthead"><span class="hotel-wordmark">{escape(config['name'])}</span><span class="hotel-edition">Reservations<br>Local demo / {escape(config['currency'])}</span></div>
  <section class="hotel-hero"><h1>Ask.</h1><div><p>Questions about your stay?<br>Chat with our hotel assistant.</p><div class="hotel-kicker">Rooms · Prices · Amenities · Reservations</div></div></section>''')

chat_tab,info_tab,manage_tab,model_tab = st.tabs(['Reservation assistant','Rooms & details','Manage','How the NLP works'],key='navigation',on_change='rerun')

with chat_tab:
    with st.container(key='reservation_workspace'):
        chat_col,detail_col = st.columns([1.5,1],gap='large')
        with chat_col:
            with st.container(key='conversation'):
                st.html(f'''<div class="hotel-chat-intro" id="reservation-assistant">
                  <svg viewBox="0 0 24 24" width="32" height="32" aria-hidden="true" focusable="false"><path d="M4 4h16v12H9l-5 4V4z" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 8h8M8 12h5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>
                  <div><div class="hotel-kicker">{escape(config['name'])} · Virtual assistant</div><h2>How can I help?</h2><p>Ask a question about rooms, prices, amenities or your booking.</p></div></div>''')
                has_conversation = len(st.session_state['messages']) > 1
                if has_conversation:
                    with st.container(height=320,autoscroll=True,border=False,key='chat_history'):
                        for message in st.session_state['messages']:
                            with st.chat_message(message['role']):
                                st.caption('YOU' if message['role']=='user' else 'HOTEL ASSISTANT')
                                if message['role']=='user':
                                    st.write(message['content'])
                                else:
                                    st.markdown(message['content'])
                st.caption('YOUR MESSAGE · TYPE A QUESTION AND PRESS ENTER')
                if prompt := st.chat_input(f"Ask me a question about {config['name']}…",max_chars=2000,key='concierge_input'):
                    add_reply(bot.respond(prompt),user=prompt)
                    st.rerun()
                st.caption('TRY A QUESTION')
                prices_col,amenities_col = st.columns(2,gap='small')
                prices_col.button('Show room prices',key='ask_room_prices',width='stretch',on_click=ask_question,args=('What are your room rates?',))
                amenities_col.button('Hotel amenities',key='ask_amenities',width='stretch',on_click=ask_question,args=('What amenities do you offer?',))
                st.caption('For example: “Is breakfast included?” or “Book a deluxe room for 3 nights.”')
                with st.expander('More questions you can ask'):
                    questions = {
                        'What can you do?': 'What can you help me with?',
                        'Room types': 'What rooms do you have?',
                        'Room for a family': 'Which room is best for a family of 4?',
                        'Availability': 'Is a deluxe room available on December 24 for 2 nights for 2 guests?',
                        'Price quote': 'How much is a suite for 3 nights?',
                        'Make a booking': 'Book a deluxe room for 3 nights from December 24 for 2 guests',
                        'Check-in and check-out times': 'What time is check-in and check-out?',
                        'Length of stay': 'How many nights can I stay?',
                        'Early or late arrival': 'Can I get early check-in or late check-out?',
                        'Check-in documents': 'What documents do I need at check-in?',
                        'Payment methods': 'What payment methods do you accept?',
                        'Taxes and fees': 'Are taxes included in the price?',
                        'Discounts and promos': 'Do you have any discounts?',
                        'Cancellation policy': 'What is your cancellation policy?',
                        'Breakfast hours': 'What time does breakfast start?',
                        'Dining': 'Do you have a restaurant?',
                        'Dietary needs': 'Do you have vegetarian or halal options?',
                        'Room service': 'Do you offer room service?',
                        'Bar': 'Is there a bar?',
                        'Pool hours': 'What are the pool opening hours?',
                        'Gym hours': 'When does the gym open?',
                        'Spa': 'Do you have a spa?',
                        'Wi-Fi': 'Is the Wi-Fi free?',
                        'Parking': 'Is there free parking?',
                        'In-room amenities': 'What is in the room?',
                        'Views and balconies': 'Can I get a room with a balcony?',
                        'Pet policy': 'Can I bring my dog?',
                        'Children': 'What is your policy for children staying?',
                        'Extra beds and cribs': 'Can I get an extra bed?',
                        'Accessibility': 'Do you have wheelchair access?',
                        'Smoking': 'What is your smoking policy?',
                        'Visitors': 'Are visitors allowed?',
                        'Outside food': 'Can I bring my own food?',
                        'Laundry': 'Is laundry available?',
                        'Luggage storage': 'Can I store my luggage before check-in?',
                        'Airport transfer': 'Do you offer an airport shuttle?',
                        'Business and meetings': 'Do you have meeting rooms?',
                        'Safety and security': 'Is the hotel safe?',
                        'Location': 'Where is the hotel located?',
                        'Nearby places': 'What is near the hotel?',
                        'Contact reception': 'How can I contact you?',
                        'Special requests': 'Can I request a quiet room?',
                        'About the hotel': 'Tell me about the hotel',
                        'My reservation': 'Check my reservation status',
                    }
                    topic = st.selectbox('Choose a topic',list(questions),key='faq_topic')
                    st.caption(f'Sends: “{questions[topic]}”')
                    st.button('Ask this question',key='ask_more',width='stretch',on_click=ask_question,args=(questions[topic],))
                with st.expander('How I understood your last message (NLP)'):
                    show_nlp_insights()
                st.caption('Add a preference with “Special request: quiet room, please.” Requests are not guaranteed.')

        with detail_col:
            with st.container(key='stay_panel'):
                section_marker('02 / The details','Review before confirming')
                show_quote()
                if bot.pending:
                    labels = {'book':'booking','cancel':'cancellation','modify':'modification'}
                    label = labels[bot.pending]
                    st.info(f'Your {label} is ready for review. Confirm only when the details are correct.')
                    if st.button(f'Confirm {label}',key='confirm_action',type='primary',width='stretch'):
                        add_reply(bot.confirm(),user=f'Confirm {label}')
                        st.rerun()
                    if st.button('Keep editing',key='decline_action',width='stretch'):
                        add_reply(bot.respond('no'),user='Keep editing')
                        st.rerun()
                if bot.last_booking:
                    receipt = bot.last_booking
                    st.html(f'''<div class="hotel-receipt"><div class="hotel-kicker">{escape(receipt['status'])} reservation</div><div class="hotel-reference">{escape(receipt['reference'])}</div></div>''')
                    st.caption('Keep this reference and your matching email to manage the reservation.')
                    text = (f"Local demo reservation\nReference: {receipt['reference']}\nStatus: {receipt['status']}\n"
                            f"Room: {receipt['room_type']}\nDates: {receipt['checkin']} to {receipt['checkout']}\n"
                            f"Nights: {receipt['nights']}\nGuests: {receipt['guests']}\nTotal: {money(receipt['total'])}\n"
                            f"Special requests (not guaranteed): {receipt.get('special_requests') or 'None'}\n")
                    st.download_button('Download summary',text,file_name=f"{receipt['reference']}.txt",mime='text/plain',width='stretch')

                if st.session_state.get('_draft_revision') != bot.revision:
                    defaults = {'draft_room':bot.draft.room_type or '',
                                'draft_checkin':bot.draft.checkin or bot.service.today+timedelta(days=1),
                                'draft_nights':bot.draft.nights or ((bot.draft.checkout-bot.draft.checkin).days if bot.draft.checkout and bot.draft.checkin else 1),
                                'draft_guests':bot.draft.guests or 1,'draft_name':bot.draft.name or '',
                                'draft_email':bot.draft.email or '',
                                'draft_requests':getattr(bot.draft,'special_requests','')}
                    for key,value in defaults.items():
                        st.session_state[key] = value
                    st.session_state['_draft_revision'] = bot.revision
                st.subheader('Make it yours.')
                with st.form('reservation_details'):
                    room = st.selectbox('Room category',['']+list(config['rooms']),key='draft_room',
                                        format_func=lambda key: config['rooms'][key]['label'] if key else 'Choose your room')
                    checkin = st.date_input('Check-in date',min_value=date.min,max_value=date.max,key='draft_checkin')
                    nights_col,guests_col = st.columns(2,gap='small')
                    nights = nights_col.number_input('Nights',min_value=1,max_value=30,step=1,key='draft_nights')
                    guests = guests_col.number_input('Guests',min_value=1,max_value=10,step=1,key='draft_guests',help='Include everyone staying, including children.')
                    name = st.text_input('Guest name',max_chars=120,key='draft_name',disabled=bot.mode=='modify',placeholder='Name for the reservation')
                    email = st.text_input('Email address',max_chars=254,key='draft_email',disabled=bot.mode=='modify',placeholder='you@example.com')
                    requests = st.text_area('Special requests (optional)',max_chars=500,key='draft_requests',
                                           placeholder='For example: a quiet room or preferred floor',help='Preferences are saved with your booking. Fulfillment is not guaranteed; this demo cannot notify hotel staff.')
                    reviewed = st.form_submit_button('Review your stay',key='review_draft',width='stretch')
                if reviewed:
                    if bot.mode in {'cancel','view'}:
                        add_reply('Finish the current lookup or cancellation before reviewing a booking. Use Start a new conversation to abandon it.')
                    else:
                        if bot.mode!='modify':
                            bot.mode = 'book'
                        add_reply(bot.update_draft(room_type=room or None,checkin=checkin,nights=int(nights),
                            guests=int(guests),name=name.strip(),email=email.strip().lower(),special_requests=requests.strip()),user='Review reservation details')
                    st.rerun()
                st.caption('Check-out follows your selected number of nights. Reviewing details does not save a reservation.')

with info_tab:
    section_marker('The room collection','One room per reservation')
    st.header('Space for your stay.')
    st.write('Choose your room, then check the dates that work for you.')
    with st.container(key='room_catalog'):
        room_items = list(config['rooms'].items())
        for row_start in range(0,len(room_items),2):
            columns = st.columns(2,gap='medium')
            for offset,(key,room) in enumerate(room_items[row_start:row_start+2]):
                with columns[offset]:
                    index = row_start+offset
                    st.html(f'''<article class="hotel-room"><div class="hotel-kicker">{index+1:02} / The collection</div><h3>{escape(room['label'])}</h3>
                      <div class="hotel-room-rate">{money(room['nightly_rate'])} <small>/ night</small></div>
                      <p>{escape(room['description'])}</p><div class="hotel-room-meta">Up to {room['capacity']} guests · Check dates for availability</div></article>''')
    st.caption(config['pricing_note'])
    st.divider()
    st.header('The essentials.')
    for amenity in config['amenities'].values():
        st.write(f"**{amenity['label']}** — {amenity['description']}")
    with st.expander('Payment & cancellation'):
        st.write(config['payment_policy'])
        st.write(config['cancellation_policy'])
    with st.expander('More hotel information'):
        from hotel_chatbot.hotel_info import TOPICS
        for key,(label,_) in TOPICS.items():
            st.write(f"**{label.title()}**")
            st.write(config.get('hotel_faq',{}).get(key) or 'Not configured. Contact reception for a verified answer.')
    address = config.get('address')
    contact = config.get('reception_contact')
    st.caption(f"Address: {address or 'Not provided for this demo'} · Reception: {contact or 'Contact details not provided'}")

with manage_tab:
    with st.container(key='management'):
        section_marker('Your reservation','Reference + matching email')
        st.header('Plans can change.')
        st.write('Find your reservation to review a change or cancellation.')
        with st.form('lookup_form'):
            ref_col,email_col = st.columns(2,gap='medium')
            reference = ref_col.text_input('Booking reference',max_chars=30,key='lookup_ref',placeholder='BR-…')
            lookup_email = email_col.text_input('Booking email',max_chars=254,key='lookup_email',placeholder='Email used when booking')
            lookup = st.form_submit_button('Find reservation',key='lookup_submit',type='primary',width='stretch')
        if lookup:
            row = bot.service.lookup(reference,lookup_email)
            st.session_state['managed_booking'] = row
            if not row:
                st.warning('No matching reservation was found. Check the reference and email.')
        row = st.session_state.get('managed_booking')
        if row:
            row = bot.service.lookup(row['reference'],row['email'])
            st.session_state['managed_booking'] = row
            if row:
                st.divider()
                st.subheader(f"{row['room_type'].title()} · {row['status'].title()}")
                st.caption(row['reference'])
                st.write(f"{row['checkin']} → {row['checkout']} · {row['guests']} guests · **{money(row['total'])}**")
                if row.get('special_requests'):
                    st.caption('Special requests · not guaranteed')
                    st.text(row['special_requests'])
                if row['status']=='active':
                    modify_col,cancel_col = st.columns(2,gap='medium')
                    modify_col.button('Change stay details',key='prepare_modify',width='stretch',on_click=begin_management,args=('modify',row))
                    cancel_col.button('Review cancellation',key='prepare_cancel',width='stretch',on_click=begin_management,args=('cancel',row))
                    st.caption('You will return to the assistant to review and confirm. Nothing changes until you confirm.')

with model_tab:
    section_marker('The language model','Local, explainable, no API key')
    st.header('How the assistant understands you.')
    st.write('Every message passes through a hybrid pipeline. Each stage is shown live in '
             '**How I understood your last message (NLP)** under the chat.')
    st.markdown(
        '1. **Normalisation**: lowercase, expand contractions and slang (“gonna” → “going to”).\n'
        '2. **Entity masking**: dates, room names, numbers, emails, and booking references become placeholder tokens '
        '(`xdate`, `xroom`, `xnum`, …), so the model learns sentence patterns instead of memorising values.\n'
        '3. **High-precision rules** catch explicit, high-stakes requests (cancel, modify, book, complaints, human agent) '
        'and negations such as “do not cancel”.\n'
        '4. **ML intent classifier**: word 1–2-gram + character 3–5-gram TF-IDF features feeding a class-balanced '
        'logistic regression (C = 10). It abstains below a confidence threshold chosen by cross-validation.\n'
        f'5. **FAQ retrieval**: TF-IDF cosine similarity against example questions for {len(TOPICS)} hotel topics.\n'
        '6. **Slot extraction**: room, check-in/check-out dates, nights, guests, name, email, and booking reference.\n'
        '7. **Dialogue manager**: asks for missing details, quotes the exact price, and saves or changes reservations '
        'only after explicit confirmation.')
    results_path = ROOT / 'evaluation' / 'results.json'
    if results_path.exists():
        import json
        results = json.loads(results_path.read_text(encoding='utf-8'))
        st.subheader('Measured accuracy')
        ind = results.get('independent')
        if ind:
            c1,c2,c3 = st.columns(3,gap='small')
            c1.metric('Intent accuracy, held-out set',f"{ind['hybrid']['accuracy']:.1%}",help=f"{ind['hybrid']['correct']}/{ind['intent_count']} messages never used for training or tuning (full pipeline).")
            c2.metric('ML classifier alone, held-out set',f"{ind['classifier']['accuracy']:.1%}",help='Intent accuracy of the TF-IDF + logistic regression model with no rules.')
            c3.metric('FAQ topic accuracy',f"{ind['faq_hybrid_accuracy']:.1%}",help=f"{ind['faq_count']} held-out hotel questions routed to the correct answer.")
        c4,c5,c6 = st.columns(3,gap='small')
        c4.metric('Casual-language benchmark, full pipeline',f"{results['hybrid']['accuracy']:.1%}",help='Notebook regression benchmark; it was inspected while writing rules, so it is not independent evidence.')
        c5.metric('Casual-language benchmark, ML alone',f"{results['classifier']['accuracy']:.1%}",
                  delta=f"{(results['classifier']['accuracy']-results['historical_notebook']['original_holdout_accuracy'])*100:+.1f} pts vs original notebook")
        c6.metric('Training examples',f"{results['training_examples']:,}")
        if ind:
            st.caption(f"Run `python scripts/evaluate.py` to reproduce. The held-out set is small ({ind['intent_count']} intent + {ind['faq_count']} FAQ messages) and was "
                       'written by the same team as the training data, so treat it as an MVP indicator, not a production guarantee.')
    counts = {}
    for example in bot.recognizer.examples:
        counts[example['intent']] = counts.get(example['intent'],0)+1
    st.subheader('Training data by intent')
    st.bar_chart({'examples':counts},horizontal=True,height=420)

st.html(f'''<div class="hotel-facts" aria-label="Hotel details"><div><strong>{len(config['rooms'])} room categories</strong><span>A place for every stay</span></div>
  <div><strong>{escape(config['checkin_time'])}</strong><span>Check-in begins</span></div><div><strong>{escape(config['checkout_time'])}</strong><span>Check-out by</span></div></div>''')
st.caption('Fictional hotel demo · Reservations saved locally · No payments collected')
