
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "textutils.settings")
django.setup()

from django.db import connection
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import identify_hasher

User = get_user_model()

print("\n=== LEGACY USER MIGRATION PREVIEW ===")

with connection.cursor() as cursor:
    cursor.execute(
        "SELECT userid, name, mailid, mobileno, password FROM users ORDER BY userid"
    )
    legacy_users = cursor.fetchall()

existing_usernames = set(
    User.objects.values_list("username", flat=True)
)

for userid, name, email, mobile, password_hash in legacy_users:
    print(f"\nLegacy username: {userid}")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Mobile: {mobile}")

    if userid == "praveen":
        print("Action: EXCLUDE — existing Django admin account is preserved.")
        continue

    if userid in existing_usernames:
        print("Action: REVIEW — username already exists in Django.")
        continue

    try:
        hasher = identify_hasher(password_hash)
        print(f"Password hash: Recognized ({hasher.algorithm})")
    except (ValueError, TypeError):
        print("Password hash: NOT RECOGNIZED — manual review required.")
        continue

    print("Action: ELIGIBLE FOR MIGRATION (preview only)")

print("\n=== EXISTING DJANGO USERS ===")
for user in User.objects.all().order_by("username"):
    print(
        f"Username: {user.username} | "
        f"Staff: {user.is_staff} | "
        f"Superuser: {user.is_superuser}"
    )

print("\nPreview completed. No database records were changed.")
