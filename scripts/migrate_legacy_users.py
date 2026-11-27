
import os
import sys

# Add the project root to Python's import path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "textutils.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import identify_hasher
from django.db import connection, transaction

User = get_user_model()

EXPECTED_USERS = {"Test", "Test1"}
EXCLUDED_USERS = {"praveen"}


def get_legacy_users():
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT userid, name, mailid, password
            FROM users
            WHERE userid IN ('Test', 'Test1')
            ORDER BY userid
            """
        )
        rows = cursor.fetchall()

    return rows


def validate_users(rows):
    found = {row[0] for row in rows}

    if found != EXPECTED_USERS:
        raise RuntimeError(
            f"Expected {sorted(EXPECTED_USERS)}, found {sorted(found)}"
        )

    for username, name, email, password_hash in rows:
        if User.objects.filter(username=username).exists():
            raise RuntimeError(
                f"Django username already exists: {username}"
            )

        if not password_hash:
            raise RuntimeError(
                f"Missing password hash for {username}"
            )

        try:
            identify_hasher(password_hash)
        except (ValueError, TypeError):
            raise RuntimeError(
                f"Unrecognized password hash for {username}"
            )

        if len(username) > 150:
            raise RuntimeError(f"Username too long: {username}")

    if not User.objects.filter(username="praveen").exists():
        raise RuntimeError("Expected Django admin 'praveen' was not found.")

    admin = User.objects.get(username="praveen")

    if not admin.is_staff or not admin.is_superuser:
        raise RuntimeError(
            "The existing praveen account is not a staff superuser."
        )


def preview(rows):
    print("\n=== MIGRATION DRY RUN ===")

    for username, name, email, password_hash in rows:
        hasher = identify_hasher(password_hash)

        print(f"\nUsername: {username}")
        print(f"First name: {name or ''}")
        print(f"Email: {email or ''}")
        print(f"Password algorithm: {hasher.algorithm}")
        print("Action: Would create Django user")

    print("\nExcluded legacy account: praveen")
    print("Existing Django administrator: preserved")
    print("\nDRY RUN ONLY — no database changes made.")


def apply_migration(rows):
    print("\nPreparing to create two Django users.")

    confirmation = input(
        "Type MIGRATE to confirm account creation: "
    )

    if confirmation != "MIGRATE":
        print("Cancelled. No changes made.")
        return

    with transaction.atomic():
        # Recheck immediately before creating accounts.
        validate_users(rows)

        for username, name, email, password_hash in rows:
            User.objects.create(
                username=username,
                first_name=name or "",
                email=email or "",
                password=password_hash,
                is_active=True,
                is_staff=False,
                is_superuser=False,
            )

    print("\nMigration transaction completed.")

    for username in sorted(EXPECTED_USERS):
        user = User.objects.get(username=username)
        print(
            f"Created: {user.username} | "
            f"Active: {user.is_active} | "
            f"Staff: {user.is_staff}"
        )


def main():
    apply = "--apply" in sys.argv

    if any(arg not in ("--apply",) for arg in sys.argv[1:]):
        print("Usage: python scripts/migrate_legacy_users.py [--apply]")
        sys.exit(1)

    try:
        rows = get_legacy_users()
        validate_users(rows)

        if apply:
            apply_migration(rows)
        else:
            preview(rows)

    except Exception as error:
        print(f"\nMIGRATION STOPPED: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
