# Hotel reservation chatbot — Streamlit + NLP

A runnable local hotel reservation demo based on your supplied NLP notebook. Chat about rooms and prices, collect validated reservation details, and explicitly confirm bookings, cancellations, or modifications. Reservations persist in SQLite, with individual room allocation and transaction-time inventory checks.

The original notebook in Downloads has not been changed. `Hotel_Reservation_Chatbot_Streamlit.ipynb` is a companion notebook for the implemented app.

## Run on Windows

Use Python **3.12** (the version tested). Open PowerShell in this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. After setup, `run.bat` launches the app using this virtual environment. If your Python installation uses `python` instead of `py`, substitute `python -m venv .venv` in the first command.

On macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

An internet connection is needed to install dependencies. The chatbot then runs locally without an API key, external model download, or NLTK resource download. It trains its small classifier locally at startup; Streamlit caches that immutable resource across sessions.

## Try a reservation

1. Say: `Book a deluxe room for 3 nights starting December 24 for 2 guests`.
2. Provide your name and email when requested, or fill in the form.
3. Ask `How much would that cost?` or ask about a hotel amenity.
4. Review the exact dates, year, guest count, and total. For the default Deluxe rate, three nights costs PHP 15,000.
5. Click **Confirm booking** or type `confirm booking`. A plain `yes` does not save anything.
6. Keep the generated reference. Use **Manage** with that reference and the matching email to review a cancellation or modify your stay. The app returns you to the assistant to confirm the resulting action.

Missing fields and unclear details trigger clarification. A form is always available for corrections. Changing details prepares a new quote and requires fresh confirmation. Reset clears the conversation and draft; it does not delete saved reservations.

## What is implemented

- Streamlit chat history, session-specific drafts, structured details form, room catalog, exact price summary, downloadable reservation summary, and management forms.
- A hotel-question picker with 43 example questions, configurable answers for 38 hotel topics (fictional demo facts in `data/hotel.json`), optional special-request notes, and identity-checked reservation status lookup in chat. See `QUESTIONS.md` for examples.
- Hybrid NLP: prioritized explicit-intent rules, TF-IDF FAQ retrieval, and a word/character TF-IDF + logistic regression intent classifier with entity masking. Low-confidence classifier guesses ask a clarifying question instead of acting; unknown items get an honest "no information" reply. A **How the NLP works** tab and a live per-message NLP panel explain each decision.
- Eighteen intent categories (the notebook's sixteen plus reservation lookup and cancellation policy), with 849 training examples in editable JSON. Measured results are in `EVALUATION.md`.
- Unambiguous dates, month names, relative days/weekdays, word/digit nights and guest counts, explicit name/email, and booking-reference extraction.
- Persistent SQLite records with overlapping-stay checks, concurrent last-room protection, idempotent creation tokens, identity checks, cancellation, and atomic modification.
- No database writes from classification alone. Confirmation names must match the pending action; contradictory or negated requests disarm the pending action.

## Configure the fictional hotel

Edit `data/hotel.json` and restart the app. The default name is Hotel Becerel. This file contains the hotel name, rooms, nightly rates in PHP, per-room guest limits, individual room inventory, check-in/out times, amenities, payment information, and optional address/reception contact. Room keys must be simple identifiers; rates and capacities are positive integers. Inventory can be zero.

Default catalog: Standard PHP 3,500, Deluxe PHP 5,000, Family PHP 7,000, Suite PHP 9,000 per night. These are **fictional demo rates**, not actual hotel offers. The total models one room and no additional taxes or fees. Stays must be 1–30 nights. Family/Suite rooms fit up to four guests; Standard/Deluxe up to two. Multi-room requests require separate reservations.

`hotel_faq` contains optional answers for pets, smoking, accessibility, extra beds, children, facility hours, dining, laundry, luggage, transport, visitors, check-in documents, and early/late arrangements. Supply verified hotel facts; leave unknown answers as `null`. A missing answer is explicitly unknown. The existing configuration remains compatible if this optional section is absent.

Availability uses specific room units, not a count of all overlapping reservation rows. Intervals include check-in and exclude check-out, so a room can accept a new guest on the prior guest's checkout date. Sold-out dates never generate a successful booking confirmation.

`var/reservations.sqlite3` is created on first run. To use another database, set `HOTEL_DB_PATH` before launching. Back up the database before changing catalog inventory. Reducing inventory retires room units for new bookings; historical reservations remain recorded. Keep database files private because they contain guest names and emails.

## Accuracy and evidence

Read `EVALUATION.md` for measured results and limitations. Run the reproducible evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py
```

This writes `evaluation/results.json` (every prediction, per-class metrics, abstentions) and `evaluation/errors.csv`. The original notebook's 96-message holdout was inspected and used to guide language coverage. It is therefore an **exposed regression benchmark**, not an independent estimate of performance on unseen users. Rule-match scores are diagnostics, not calibrated probabilities. The classifier-only and hybrid results are reported separately.

The automated tests verify booking correctness and conversational behavior. They do not substitute for human usability testing or establish a universal accuracy guarantee. Before real deployment, collect fresh, independently authored messages and test with actual users.

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

The suite includes real SQLite concurrent booking tests, parsing and conversation tests, evaluation CLI verification, and Streamlit's AppTest for rendering, chat, explicit booking confirmation, reruns, and cancellation. Tests use temporary databases, not the app's guest database.

The expansion also verifies read-only FAQs/status lookup, special-request validation/persistence/modification, old-database migration, and notes submitted through the Streamlit form. New notes are saved only on explicit booking/modification confirmation and never imply fulfillment.

## Project files

`app.py` is the Streamlit entrypoint. `hotel_chatbot/` holds reusable config, parsing, NLP, conversation, and reservation services. `data/` holds editable hotel/training data and the original exposed regression set. `tests/` contains verification. The companion notebook demonstrates the same tested implementation rather than maintaining a separate chatbot code path.

## Interface styling

The interface follows a monochrome editorial style: white backgrounds, black controls and rules, serif headings, monospace labels, square corners, and subtle line/grid textures. `styles/monochrome.css` centralizes the presentation tokens; `.streamlit/config.toml` supplies the matching native theme. Fonts use system fallbacks so no external font download is needed.

The “Ask.” heading and black “How can I help?” panel invite hotel questions. A large message box and visible **Show room prices** / **Hotel amenities** buttons make chat easy to start. Chat and stay details sit beside each other on desktop and stack on small screens. Long conversations use Streamlit's native scrolling and show the newest reply. The details form pairs nights/guests, keeps a separate review step, and shows the total in a distinct quote panel. **Rooms & details** presents the catalog; **Manage** finds existing reservations and returns reviews to **Reservation assistant**. Optional diagnostics are inside the collapsed sidebar.

To see a source update, refresh your browser. If theme settings remain stale, stop Streamlit with Ctrl+C and start it again using `run.bat` or the launch command above. Existing reservations stay in the same database.

For this backend expansion, restart Streamlit with Ctrl+C followed by `run.bat`, then refresh the browser. A running process can retain older imported Python modules even when the new form controls appear.

## Scope

This app is a local demonstration, with no connection to a real hotel property-management system, payment processing, email delivery, or staff-transfer service. Unconfigured facts are explicitly unknown. Reference/email checks offer limited privacy for local demo use; production needs verified identities, authentication, retention controls, and a real inventory integration. It supports English and conservative parsing, rather than every possible phrasing or language.

Streamlit integration follows its official [session-state documentation](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) and [AppTest documentation](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
