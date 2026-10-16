from dataclasses import dataclass, field
from datetime import date


@dataclass
class Draft:
    room_type: str | None = None
    checkin: date | None = None
    checkout: date | None = None
    nights: int | None = None
    guests: int | None = None
    name: str | None = None
    email: str | None = None
    special_requests: str = ''


@dataclass(frozen=True)
class Quote:
    room_type: str
    checkin: date
    checkout: date
    nights: int
    guests: int
    nightly_rate: int
    total: int


@dataclass
class ParsedMessage:
    slots: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Prediction:
    intent: str
    score: float
    source: str
    topic: str | None = None
