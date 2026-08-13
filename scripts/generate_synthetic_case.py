"""Generate a fully synthetic case folder for Karuth demos and tests.

All generated people, numbers, accounts, locations, plates, files, and messages
are fictional. The output is intentionally local and deterministic.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


CASE_ID = "SYN-KARUTH-001"
BASE_TIME = datetime(2026, 1, 12, 9, 30, tzinfo=timezone.utc)


def reset_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)


def create_chat_db(output: Path, *, include_wow_data: bool = False) -> None:
    db_path = output / "chats" / "synthetic_whatsapp.sqlite"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE contacts (
                contact_id INTEGER PRIMARY KEY,
                display_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                username TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE messages (
                message_id INTEGER PRIMARY KEY,
                contact_id INTEGER NOT NULL,
                sent_at TEXT NOT NULL,
                body TEXT NOT NULL,
                FOREIGN KEY(contact_id) REFERENCES contacts(contact_id)
            )
            """
        )
        contacts = [
            (1, "Arun Nair", "+919900001001", "arun_demo"),
            (2, "Maya Pillai", "+919900001002", "maya_demo"),
            (3, "Ravi Menon", "+919900001003", "ravi_demo"),
        ]
        messages = [
            (1, 1, BASE_TIME.isoformat(), "Meet near Kaloor with vehicle KL07CX4242."),
            (2, 2, (BASE_TIME + timedelta(minutes=8)).isoformat(), "Payment memo says KL07CX4242."),
            (3, 3, (BASE_TIME + timedelta(hours=2)).isoformat(), "Camera serial CAM-SYN-009 appears in folder."),
            (4, 1, (BASE_TIME + timedelta(days=1)).isoformat(), "Use alias blue_lantern for the demo account."),
        ]
        if include_wow_data:
            # A second, contradictory sighting of the same plate in a distant
            # city near the same time — fuels the hypothesis-demo contradiction.
            messages.append(
                (5, 3, (BASE_TIME + timedelta(minutes=6)).isoformat(), "Spotted KL07CX4242 parked near Connaught Place, Delhi at 09:35.")
            )
        conn.executemany("INSERT INTO contacts VALUES (?, ?, ?, ?)", contacts)
        conn.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", messages)


def _synthetic_jpeg_bytes(label: str) -> bytes:
    # Minimal JPEG-like fixture with a synthetic APP1 block. Later EXIF parsing can
    # replace this with richer generated assets without using real personal data.
    exif_payload = f"Exif\x00\x00SyntheticKaruth;{label}".encode("ascii")
    app1_length = len(exif_payload) + 2
    return (
        b"\xff\xd8"
        + b"\xff\xe1"
        + app1_length.to_bytes(2, "big")
        + exif_payload
        + b"\xff\xdb\x00C"
        + bytes([0] * 67)
        + b"\xff\xd9"
    )


def create_images(output: Path) -> None:
    image_dir = output / "images"
    image_dir.mkdir(parents=True)
    identities = ["person_alpha", "person_beta", "person_gamma"]
    for index in range(1, 13):
        identity = identities[index % len(identities)]
        gps_lat = 9.9312 + index / 10000
        gps_lon = 76.2673 + index / 10000
        serial = "CAM-SYN-009" if index in {3, 7, 11} else f"CAM-SYN-{index:03d}"
        plate = "KL07CX4242" if index in {4, 8} else f"KL{index:02d}AB{1000 + index}"
        stem = f"IMG_{index:04d}"
        (image_dir / f"{stem}.jpg").write_bytes(
            _synthetic_jpeg_bytes(
                f"identity={identity};gps={gps_lat:.5f},{gps_lon:.5f};serial={serial};plate={plate}"
            )
        )
        (image_dir / f"{stem}.exif.json").write_text(
            json.dumps(
                {
                    "file": f"{stem}.jpg",
                    "captured_at": (BASE_TIME + timedelta(minutes=index * 4)).isoformat(),
                    "gps": {"lat": gps_lat, "lon": gps_lon},
                    "camera_serial": serial,
                    "synthetic_identity": identity,
                    "vehicle_plate": plate,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def create_bank_csv(output: Path) -> None:
    bank_dir = output / "bank"
    bank_dir.mkdir(parents=True)
    with (bank_dir / "transactions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["transaction_id", "timestamp", "amount", "account", "memo"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "transaction_id": "TXN-SYN-001",
                    "timestamp": (BASE_TIME + timedelta(minutes=10)).isoformat(),
                    "amount": "12500.00",
                    "account": "ACCT-SYN-4421",
                    "memo": "Fuel advance for KL07CX4242",
                },
                {
                    "transaction_id": "TXN-SYN-002",
                    "timestamp": (BASE_TIME + timedelta(hours=4)).isoformat(),
                    "amount": "3400.00",
                    "account": "ACCT-SYN-7781",
                    "memo": "Camera repair CAM-SYN-009",
                },
            ]
        )


def create_documents(output: Path) -> None:
    docs_dir = output / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "device_inventory.txt").write_text(
        "Synthetic device inventory\nPhone: +919900001001\nCamera serial: CAM-SYN-009\n",
        encoding="utf-8",
    )
    (docs_dir / "case_notes.md").write_text(
        "# Synthetic Case Notes\n\nVehicle KL07CX4242 appears in chat, image tags, and bank memo.\n",
        encoding="utf-8",
    )


def generate_case(output: Path, *, include_wow_data: bool = False) -> None:
    """Generate a synthetic case folder.

    ``include_wow_data`` adds a deliberate contradiction (a distant-city
    sighting of the demo vehicle) so the hypothesis-driven demo flow can
    discover it, challenge the lead, and lower confidence.
    """
    reset_output(output)
    create_chat_db(output, include_wow_data=include_wow_data)
    create_images(output)
    create_bank_csv(output)
    create_documents(output)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": CASE_ID,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "synthetic": True,
                "contents": ["chats", "images", "bank", "documents"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Jesse Pinkman synthetic demo preset (multimodal pitch demo)
# ---------------------------------------------------------------------------
# All content is fictional and clearly marked as a synthetic fixture. The demo
# connects a candidate person (synthetic_identity=jesse_pinkman), a vehicle
# (NM-JP-4242) and a location (Albuquerque NM) across three images with
# embedded EXIF + sidecar metadata. Two variants:
#
#   clean:       consistent evidence -> candidate association + human review
#   adversarial: planted conflicts (distant-city sighting, embedded-vs-sidecar
#                EXIF conflict, alternate plate reading, second identity on the
#                same plate/camera) -> contradiction -> skeptic -> review

JESSE_CASE_ID = "SYN-JESSE-001"
JESSE_BASE_TIME = datetime(2026, 1, 12, 9, 30, tzinfo=timezone.utc)
JESSE_GPS = {"lat": 35.0844, "lon": -106.6504}  # Albuquerque, NM
JESSE_PLATE = "NM-JP-4242"
JESSE_CAMERA = "CAM-JESSE-001"
JESSE_IDENTITY = "jesse_pinkman"
JESSE_GOAL = (
    "Find the person connected to vehicle NM-JP-4242 and determine whether "
    "the chat activity supports that connection."
)
JESSE_ACTIVITY = (
    "The image appears to show a person handling a small transparent object near a table."
)


def _dms_tuple(value: float) -> tuple:
    """Decimal degrees -> EXIF rational DMS tuple (three IFDRationals)."""
    from PIL.TiffImagePlugin import IFDRational

    value = abs(float(value))
    degrees = int(value)
    minutes_full = (value - degrees) * 60
    minutes = int(minutes_full)
    seconds = round((minutes_full - minutes) * 60, 6)
    return (
        IFDRational(degrees, 1),
        IFDRational(minutes, 1),
        IFDRational(int(round(seconds * 1000)), 1000),
    )


def _write_jpeg_with_exif(
    path: Path,
    rgb: tuple[int, int, int],
    *,
    captured_at: datetime | None = None,
    gps: dict | None = None,
    make: str = "KaruthSyntheticCam",
    model: str = "KaruthCam-1",
    orientation: int = 1,
) -> None:
    """Write a real JPEG with embedded EXIF (GPS + DateTimeOriginal + make/model)."""
    from PIL import ExifTags
    from PIL import Image

    image = Image.new("RGB", (320, 240), rgb)
    exif = image.getexif()
    exif[0x010F] = make
    exif[0x0110] = model
    exif[0x0112] = orientation
    if captured_at is not None:
        exif[0x9003] = captured_at.strftime("%Y:%m:%d %H:%M:%S")
    if gps:
        gps_ifd = {
            ExifTags.GPS.GPSLatitudeRef: "N" if gps["lat"] >= 0 else "S",
            ExifTags.GPS.GPSLatitude: _dms_tuple(gps["lat"]),
            ExifTags.GPS.GPSLongitudeRef: "E" if gps["lon"] >= 0 else "W",
            ExifTags.GPS.GPSLongitude: _dms_tuple(gps["lon"]),
        }
        exif[0x8825] = gps_ifd
    image.save(path, "JPEG", quality=90, exif=exif)


def _write_image(path: Path, rgb: tuple[int, int, int], **kwargs) -> None:
    """Write an image with embedded EXIF; degrade to fixture bytes without Pillow."""
    try:
        _write_jpeg_with_exif(path, rgb, **kwargs)
    except Exception:  # noqa: BLE001 - Pillow unavailable: sidecar-only demo still works
        path.write_bytes(_synthetic_jpeg_bytes("fallback"))


def _write_sidecar(image_path: Path, payload: dict) -> None:
    sidecar = image_path.with_suffix(".exif.json")
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _create_jesse_chat(output: Path, *, adversarial: bool) -> None:
    db_path = output / "chats" / "jesse_whatsapp.sqlite"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE contacts (contact_id INTEGER PRIMARY KEY, display_name TEXT NOT NULL, phone TEXT NOT NULL, username TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE messages (message_id INTEGER PRIMARY KEY, contact_id INTEGER NOT NULL, sent_at TEXT NOT NULL, body TEXT NOT NULL, FOREIGN KEY(contact_id) REFERENCES contacts(contact_id))"
        )
        contacts = [
            (1, "Jesse P.", "+915550000001", "jesse_demo"),
            (2, "Walt B.", "+915550000002", "walt_demo"),
            (3, "Saul G.", "+915550000003", "saul_demo"),
        ]
        messages = [
            (1, 1, (JESSE_BASE_TIME - timedelta(minutes=10)).isoformat(), "Meet at the Sandia Crest trailhead. Truck NM-JP-4242."),
            (2, 2, (JESSE_BASE_TIME - timedelta(minutes=5)).isoformat(), "Camera CAM-JESSE-001 stays with the vehicle."),
        ]
        if adversarial:
            messages.append(
                (3, 3, (JESSE_BASE_TIME + timedelta(minutes=5)).isoformat(), "Spotted NM-JP-4242 parked near Connaught Place, Delhi at 09:35.")
            )
        conn.executemany("INSERT INTO contacts VALUES (?, ?, ?, ?)", contacts)
        conn.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", messages)


def _create_jesse_images(output: Path, *, adversarial: bool) -> None:
    image_dir = output / "images"
    image_dir.mkdir(parents=True)

    # IMG_001 — suspicious activity. Sensitive content flagged; the vision
    # description is a canned LLM-style *interpretation* that requires review.
    t1 = JESSE_BASE_TIME
    _write_image(image_dir / "IMG_001_suspicious.jpg", (96, 70, 62), captured_at=t1)
    _write_sidecar(
        image_dir / "IMG_001_suspicious.jpg",
        {
            "file": "IMG_001_suspicious.jpg",
            "captured_at": t1.isoformat(),
            "synthetic_identity": JESSE_IDENTITY,
            "camera_serial": JESSE_CAMERA,
            "sensitive_content": True,
            "activity_interpretation": JESSE_ACTIVITY,
            "interpretation_confidence": 0.5,
        },
    )

    # IMG_002 — location bridge: same subject near a distinctive landmark.
    t2 = JESSE_BASE_TIME + timedelta(minutes=6)
    _write_image(image_dir / "IMG_002_landmark.jpg", (70, 96, 62), captured_at=t2)
    _write_sidecar(
        image_dir / "IMG_002_landmark.jpg",
        {
            "file": "IMG_002_landmark.jpg",
            "captured_at": t2.isoformat(),
            "synthetic_identity": JESSE_IDENTITY,
            "camera_serial": JESSE_CAMERA,
            "landmark": "Sandia Crest ridge, Albuquerque NM",
        },
    )

    # IMG_003 — person + vehicle + EXIF (GPS, timestamp, camera serial, plate).
    t3 = JESSE_BASE_TIME + timedelta(minutes=12)
    embedded_gps = JESSE_GPS
    embedded_time = t3
    if adversarial:
        # Embedded EXIF deliberately conflicts with the sidecar: capture
        # timestamp 09:48 and New York GPS vs sidecar 09:42 + Albuquerque.
        embedded_gps = {"lat": 40.7128, "lon": -74.0060}
        embedded_time = JESSE_BASE_TIME + timedelta(minutes=18)
    _write_image(
        image_dir / "IMG_003_vehicle.jpg",
        (62, 62, 96),
        captured_at=embedded_time,
        gps=embedded_gps,
    )
    sidecar3 = {
        "file": "IMG_003_vehicle.jpg",
        "captured_at": t3.isoformat(),
        "gps": JESSE_GPS,
        "camera_serial": JESSE_CAMERA,
        "synthetic_identity": JESSE_IDENTITY,
        "vehicle_plate": JESSE_PLATE,
        "visible_vehicle": "dark pickup truck",
    }
    if adversarial:
        sidecar3["plate_confidence"] = 0.78
        sidecar3["plate_alternatives"] = ["NM-JP-424Z", "NM-JP-4247"]
    else:
        sidecar3["plate_confidence"] = 0.97
        sidecar3["plate_alternatives"] = []
    _write_sidecar(image_dir / "IMG_003_vehicle.jpg", sidecar3)

    if adversarial:
        # A second synthetic identity appears with the SAME plate and camera
        # serial -> inconsistent identity contradictions.
        t4 = JESSE_BASE_TIME + timedelta(minutes=20)
        _write_image(
            image_dir / "IMG_004_other.jpg",
            (96, 62, 96),
            captured_at=t4,
            gps=JESSE_GPS,
        )
        _write_sidecar(
            image_dir / "IMG_004_other.jpg",
            {
                "file": "IMG_004_other.jpg",
                "captured_at": t4.isoformat(),
                "gps": JESSE_GPS,
                "camera_serial": JESSE_CAMERA,
                "synthetic_identity": "person_beta",
                "vehicle_plate": JESSE_PLATE,
            },
        )


def _create_jesse_bank(output: Path) -> None:
    bank_dir = output / "bank"
    bank_dir.mkdir(parents=True)
    with (bank_dir / "transactions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["transaction_id", "timestamp", "amount", "account", "memo"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "transaction_id": "TXN-JESSE-001",
                "timestamp": (JESSE_BASE_TIME + timedelta(minutes=15)).isoformat(),
                "amount": "85.00",
                "account": "ACCT-JESSE-11",
                "memo": "Fuel receipt NM-JP-4242",
            }
        )


def _create_jesse_documents(output: Path) -> None:
    docs_dir = output / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "case_notes.md").write_text(
        "# Synthetic Jesse Demo Case\n\n"
        "Vehicle NM-JP-4242 appears in images and chat. Camera CAM-JESSE-001. "
        "Location: Sandia Crest, Albuquerque NM. All data is a synthetic fixture.\n",
        encoding="utf-8",
    )


def generate_jesse_case(output: Path, *, adversarial: bool = False) -> None:
    """Generate the Jesse Pinkman synthetic demo case.

    ``adversarial=True`` plants the contradictions that power the pitch "wow
    moment": a distant-city sighting, embedded-vs-sidecar EXIF conflict,
    alternate plate OCR readings, and a second identity on the same
    plate/camera. All people and data are fictional fixtures.
    """
    reset_output(output)
    _create_jesse_chat(output, adversarial=adversarial)
    _create_jesse_images(output, adversarial=adversarial)
    _create_jesse_bank(output)
    _create_jesse_documents(output)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": JESSE_CASE_ID,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "synthetic": True,
                "demo": "jesse_pinkman",
                "variant": "adversarial" if adversarial else "clean",
                "contents": ["chats", "images", "bank", "documents"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic Karuth case.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_case(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
