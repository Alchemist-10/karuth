"""Mock Motor Vehicles Department (MVD) registry lookup.

A clearly synthetic stand-in for an external vehicle-registry API. It
implements the :class:`~karuth.planner.extensions.ExternalLookupTool` protocol
so a real registry client can later replace it without planner changes.

Everything returned here is fictional fixture data: demo plates map to
fictional owners, and unknown plates resolve to a deterministic but clearly
fake owner so the lookup never leaks or fabricates real personal data. The
lookup is approval-gated (``requires_approval = True``) and the results are
labeled ``MOCK_MVD_REGISTRY`` wherever they surface.

Safety notes surfaced on every result:
- registered ownership does not prove who operated the vehicle;
- registry data describes the registration record, not the investigation.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from karuth.planner.extensions import (
    ExternalLookupRequest,
    ExternalLookupResult,
    ExternalLookupTool,
)

# Fictional registered owners for the synthetic demo plates. Names and
# addresses are invented fixtures — never real individuals.
_MOCK_RC_OWNERS: dict[str, dict] = {
    "NM-JP-4242": {
        "registered_owner": "Dana R. Voss",
        "address": "412 Mockingbird Ln, Albuquerque, NM 87102",
        "registration_date": "2024-11-02",
        "make": "Volkswagen",
        "model": "Jetta",
        "color": "silver",
        "registration_status": "active",
        "insurance_status": "verified",
    },
    "KL07CX4242": {
        "registered_owner": "K. R. Menon",
        "address": "Synthetic House 7, Marine Drive, Kochi, KL 682031",
        "registration_date": "2023-05-19",
        "make": "Hyundai",
        "model": "i20",
        "color": "white",
        "registration_status": "active",
        "insurance_status": "verified",
    },
}

# Fictional owner pool for deterministic fallback on unknown plates.
_FALLBACK_FAMILIES = ("Fernandez", "Kowalski", "Okafor", "Lindqvist", "Moreau")
_FALLBACK_GIVEN = ("A. M.", "R. J.", "T. S.", "L. K.", "P. N.")


def _normalize_plate(value: str) -> str:
    """Uppercase, strip whitespace and punctuation (keeps hyphen form)."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def _fallback_owner(plate: str) -> dict:
    """Deterministic, clearly-fake owner for an unknown plate."""
    digest = hashlib.sha256(plate.encode("utf-8")).hexdigest()
    family = _FALLBACK_FAMILIES[int(digest[0:2], 16) % len(_FALLBACK_FAMILIES)]
    given = _FALLBACK_GIVEN[int(digest[2:4], 16) % len(_FALLBACK_GIVEN)]
    return {
        "registered_owner": f"{given} {family}",
        "address": f"Fictional Address {int(digest[4:8], 16) % 1000}, Synthetic City",
        "registration_date": "1999-01-01",
        "make": "Unknown",
        "model": "Unknown",
        "color": "Unknown",
        "registration_status": "fixture_only",
        "insurance_status": "not_checked",
    }


class MockMVDRegistry:
    """Mock vehicle-registry client (approval-gated, synthetic data)."""

    name = "MOCK_MVD_REGISTRY"
    requires_approval = True

    def lookup(self, request: ExternalLookupRequest) -> ExternalLookupResult:
        """Return the (fictional) RC record for the requested plate."""
        plate = _normalize_plate(request.key)
        if not plate:
            raise ValueError("MVD lookup requires a non-empty plate")
        owner = _MOCK_RC_OWNERS.get(plate) or _fallback_owner(plate)
        known_fixture = plate in _MOCK_RC_OWNERS
        findings = [
            {
                "field": "registered_owner",
                "value": owner["registered_owner"],
                "source": self.name,
                "confidence": 0.99 if known_fixture else 0.5,
                "note": (
                    "Registered owner from a synthetic fixture record. "
                    "Registered ownership does not prove who operated the vehicle."
                ),
            },
            {
                "field": "registered_address",
                "value": owner["address"],
                "source": self.name,
                "confidence": 0.99 if known_fixture else 0.5,
                "note": "Synthetic fixture address.",
            },
            {
                "field": "vehicle_details",
                "value": f"{owner['make']} {owner['model']} ({owner['color']})",
                "source": self.name,
                "confidence": 0.9 if known_fixture else 0.4,
                "note": "Synthetic fixture vehicle details.",
            },
            {
                "field": "registration_status",
                "value": owner["registration_status"],
                "source": self.name,
                "confidence": 0.95 if known_fixture else 0.5,
                "note": "Registry status; does not imply involvement in this case.",
            },
        ]
        return ExternalLookupResult(
            lookup_kind=request.lookup_kind or "mvd_vehicle_owner",
            key=plate,
            findings=findings,
            source_system=self.name,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            approved_by=request.requester_note or "investigator",
        )


__all__ = ["MockMVDRegistry"]
