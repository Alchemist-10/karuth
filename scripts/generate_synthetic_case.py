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


def create_chat_db(output: Path) -> None:
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


def generate_case(output: Path) -> None:
    reset_output(output)
    create_chat_db(output)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic Karuth case.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_case(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
