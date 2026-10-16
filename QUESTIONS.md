# Questions and reservation features

Open **More questions you can ask** in the assistant to choose a topic and send its example question. You can also type questions directly.

## Hotel questions

The assistant already answers configured room types, rates, capacity, availability, amenities, payment methods, check-in/out times, and cancellation policy. The expansion adds these specific topics:

| Example question | Key in `hotel_faq` |
|---|---|
| Can I bring my dog? | `pets` |
| What is your smoking policy? | `smoking` |
| Do you have wheelchair access? | `accessibility` |
| Can I get an extra bed or a crib? | `extra_beds` |
| What is your policy for children staying? | `children` |
| What time does breakfast start? | `breakfast_hours` |
| What are the pool opening hours? | `pool_hours` |
| When does the gym open? | `gym_hours` |
| Do you have a restaurant? | `restaurant` |
| Do you offer room service? | `room_service` |
| Is laundry available? | `laundry` |
| Can I store my luggage before check-in? | `luggage` |
| Do you offer an airport shuttle? | `shuttle` |
| Are visitors allowed? | `visitors` |
| What documents do I need at check-in? | `checkin_documents` |
| Can I get early check-in or late check-out? | `early_late` |

Answers live under `hotel_faq` in `data/hotel.json`. As of the MVP update (2026-10-16) every topic has a **fictional demo answer** (Hotel Becerel is not a real hotel), and further topics were added: spa, bar, in-room amenities, views and balconies, Wi-Fi, parking, business and meetings, dietary options, outside food, safety and security, contact, location, nearby places, taxes and fees, discounts, special requests, about the hotel, and what the assistant can do. The dropdown in the app lists 43 example questions. Replace any answer with verified hotel facts before real use. Setting an answer to `null` makes the chatbot say the information is not configured and point to reception.

For example, once the hotel confirms an answer, replace the relevant `null` with its text. Keep JSON quotes and commas valid. Restart Streamlit after editing configuration or Python modules.

## Special requests

Type `Special request: Quiet room, please.` or use **Special requests (optional)** in the details form. This replaces the current note. Notes are limited to 500 characters; `Clear special requests` removes the note from the draft.

The review includes the note before confirmation. Confirming saves it with the reservation and includes it in the downloadable summary. Changing notes on an existing reservation requires its reference and matching email, a modification review, and explicit confirmation. Existing reservations receive an empty note through an additive database migration.

Notes are preferences, not guaranteed room features. This local demo cannot notify hotel staff, fulfill requests, or change prices based on a note. Dates and room names inside an explicit note do not change the stay details.

## Check an existing reservation in chat

1. Ask `Check my reservation status` or `Show my booking`.
2. Provide the booking reference when requested.
3. Provide the matching booking email.
4. Read its active/cancelled status, room, dates, guests, recorded total, and any notes.

You can also supply both credentials in the lookup message. Lookup is read-only and shows cancelled reservations without reactivating them. **Manage** still provides the form-based lookup/change/cancellation workflow.

Each booking continues to cover one room and 1–30 nights. Multi-room bookings, payments, staff messaging, and real property inventory integration are outside this expansion.
