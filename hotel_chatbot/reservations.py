"""Transactional room allocation and reservation management."""
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from .config import hotel_today
from .models import Quote


class ReservationService:
    def __init__(self, db_path, config, today=None):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.config = config
        self._today = today
        with self._connection(write=True) as db:
            db.execute('CREATE TABLE IF NOT EXISTS rooms (id TEXT PRIMARY KEY, room_type TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1)')
            db.execute('''CREATE TABLE IF NOT EXISTS reservations (
                reference TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE,
                room_id TEXT NOT NULL REFERENCES rooms(id), room_type TEXT NOT NULL,
                checkin TEXT NOT NULL, checkout TEXT NOT NULL, nights INTEGER NOT NULL,
                guests INTEGER NOT NULL, name TEXT NOT NULL, email TEXT NOT NULL,
                nightly_rate INTEGER NOT NULL, total INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active','cancelled')),
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                CHECK(checkout > checkin), CHECK(nights BETWEEN 1 AND 30), CHECK(guests > 0))''')
            # Additive, idempotent migration: existing reservations retain every field.
            columns = {row['name'] for row in db.execute('PRAGMA table_info(reservations)')}
            if 'special_requests' not in columns:
                db.execute("ALTER TABLE reservations ADD COLUMN special_requests TEXT NOT NULL DEFAULT ''")
            db.execute('CREATE INDEX IF NOT EXISTS reservations_overlap ON reservations(room_id, status, checkin, checkout)')
            db.execute('UPDATE rooms SET active=0')
            for category, room in config['rooms'].items():
                for number in range(room['inventory']):
                    db.execute('INSERT INTO rooms(id,room_type,active) VALUES(?,?,1) ON CONFLICT(id) DO UPDATE SET active=1',
                               (f'{category}-{number+1:03}',category))

    @property
    def today(self):
        return self._today or hotel_today(self.config)

    @contextmanager
    def _connection(self,write=False):
        db = sqlite3.connect(self.path,timeout=30,isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            if write:
                db.execute('BEGIN IMMEDIATE')
            yield db
            if write:
                db.commit()
        except Exception:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    def quote(self,draft,require_identity=False):
        from datetime import timedelta
        room = self.config['rooms'].get(draft.room_type)
        if not isinstance(draft.special_requests, str) or len(draft.special_requests) > 500:
            raise ValueError('Special requests must be text of at most 500 characters.')
        if any(ord(char) < 32 and char not in '\n\r\t' for char in draft.special_requests):
            raise ValueError('Special requests contain unsupported control characters.')
        if not room:
            raise ValueError('Choose a configured room category.')
        if type(draft.checkin) is not date:
            raise ValueError('Provide a valid check-in date.')
        if draft.checkin < self.today:
            raise ValueError('Check-in cannot be in the past.')
        if draft.checkout is not None and type(draft.checkout) is not date:
            raise ValueError('Provide a valid check-out date.')
        nights = draft.nights
        if nights is None and draft.checkout is not None:
            nights = (draft.checkout-draft.checkin).days
        if type(nights) is not int or not 1 <= nights <= 30:
            raise ValueError('A stay must be 1 to 30 nights.')
        try:
            checkout = draft.checkin + timedelta(days=nights)
        except OverflowError as exc:
            raise ValueError('Check-out exceeds the supported calendar range. Choose an earlier check-in date.') from exc
        if draft.checkout is not None and draft.checkout != checkout:
            raise ValueError('Check-out and number of nights conflict. Please correct the dates or length of stay.')
        if type(draft.guests) is not int or not 1 <= draft.guests <= room['capacity']:
            raise ValueError(f"{room['label']} accommodates 1–{room['capacity']} guests. Choose a suitable room or adjust the guest count.")
        if require_identity:
            if not isinstance(draft.name,str) or not 2 <= len(draft.name.strip()) <= 120:
                raise ValueError('Enter a guest name of 2–120 characters.')
            if not isinstance(draft.email,str) or len(draft.email)>254 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+',draft.email.strip()):
                raise ValueError('Enter a valid email address.')
        return Quote(draft.room_type,draft.checkin,checkout,nights,draft.guests,room['nightly_rate'],room['nightly_rate']*nights)

    def _free_rooms(self,db,room_type,checkin,checkout,exclude_reference=None):
        return db.execute('''SELECT r.id FROM rooms r WHERE r.active=1 AND r.room_type=? AND NOT EXISTS (
            SELECT 1 FROM reservations b WHERE b.room_id=r.id AND b.status='active'
            AND b.checkin < ? AND b.checkout > ? AND (? IS NULL OR b.reference <> ?)) ORDER BY r.id''',
            (room_type,checkout.isoformat(),checkin.isoformat(),exclude_reference,exclude_reference)).fetchall()

    def available(self,room_type,checkin,checkout,exclude_reference=None):
        if room_type not in self.config['rooms'] or type(checkin) is not date or type(checkout) is not date or checkout<=checkin:
            raise ValueError('Provide a valid room category and date interval.')
        with self._connection() as db:
            return len(self._free_rooms(db,room_type,checkin,checkout,exclude_reference))

    def create(self,draft,request_id):
        quote = self.quote(draft,require_identity=True)
        if not isinstance(request_id,str) or not request_id:
            raise ValueError('A unique confirmation token is required.')
        with self._connection(write=True) as db:
            existing = db.execute('SELECT * FROM reservations WHERE request_id=?',(request_id,)).fetchone()
            if existing:
                return dict(existing)
            rooms = self._free_rooms(db,quote.room_type,quote.checkin,quote.checkout)
            if not rooms:
                raise ValueError('No room is available for those dates; the category is sold out. Try different dates or a different room.')
            ref = 'BR-'+uuid.uuid4().hex[:12].upper()
            now = datetime.now(timezone.utc).isoformat()
            db.execute('''INSERT INTO reservations (
                reference,request_id,room_id,room_type,checkin,checkout,nights,guests,
                name,email,nightly_rate,total,status,created_at,updated_at,special_requests)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (ref,request_id,rooms[0]['id'],quote.room_type,quote.checkin.isoformat(),quote.checkout.isoformat(),
                 quote.nights,quote.guests,draft.name.strip(),draft.email.strip().lower(),quote.nightly_rate,quote.total,'active',now,now,draft.special_requests.strip()))
            return dict(db.execute('SELECT * FROM reservations WHERE reference=?',(ref,)).fetchone())

    def lookup(self,reference,email):
        with self._connection() as db:
            row = db.execute('SELECT * FROM reservations WHERE reference=? AND email=?',
                (str(reference).strip().upper(),str(email).strip().lower())).fetchone()
            return dict(row) if row else None

    def _verified(self,db,reference,email):
        row = db.execute('SELECT * FROM reservations WHERE reference=? AND email=?',
            (str(reference).strip().upper(),str(email).strip().lower())).fetchone()
        if not row:
            raise ValueError('No matching reservation was found. Check the reference and email.')
        return row

    def cancel(self,reference,email):
        with self._connection(write=True) as db:
            row = self._verified(db,reference,email)
            db.execute("UPDATE reservations SET status='cancelled', updated_at=? WHERE reference=?",
                (datetime.now(timezone.utc).isoformat(),row['reference']))
            return dict(db.execute('SELECT * FROM reservations WHERE reference=?',(row['reference'],)).fetchone())

    def modify(self,reference,email,draft):
        quote = self.quote(draft,require_identity=True)
        with self._connection(write=True) as db:
            row = self._verified(db,reference,email)
            if row['status'] != 'active':
                raise ValueError('A cancelled reservation cannot be modified.')
            # Management changes stay details only; identity stays with the verified owner.
            if draft.email.strip().lower() != row['email'] or draft.name.strip() != row['name']:
                raise ValueError('Guest identity cannot be changed through reservation modification.')
            rooms = self._free_rooms(db,quote.room_type,quote.checkin,quote.checkout,row['reference'])
            if not rooms:
                raise ValueError('No room is available for the new dates. Your original reservation is unchanged.')
            db.execute('''UPDATE reservations SET room_id=?,room_type=?,checkin=?,checkout=?,nights=?,guests=?,
                nightly_rate=?,total=?,updated_at=?,special_requests=? WHERE reference=?''',
                (rooms[0]['id'],quote.room_type,quote.checkin.isoformat(),quote.checkout.isoformat(),quote.nights,
                 quote.guests,quote.nightly_rate,quote.total,datetime.now(timezone.utc).isoformat(),draft.special_requests.strip(),row['reference']))
            return dict(db.execute('SELECT * FROM reservations WHERE reference=?',(row['reference'],)).fetchone())

    def count(self):
        with self._connection() as db:
            return db.execute('SELECT COUNT(*) FROM reservations').fetchone()[0]
