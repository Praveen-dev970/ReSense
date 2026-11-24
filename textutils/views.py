# I have created this file - tusharpangare ©

import pandas as pd   
from django.shortcuts import render, redirect,HttpResponse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import csv
import MySQLdb
import os
import sys
import uuid
import subprocess
from pathlib import Path

from django.conf import settings
from django.db import connection, IntegrityError, DatabaseError
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ImproperlyConfigured, ValidationError
from functools import wraps
from django.views.decorators.http import require_POST


def user_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if request.session.get("resense_role") != "user":
            return redirect("index")
        return view_func(request, *args, **kwargs)
    return wrapper


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if request.session.get("resense_role") != "admin":
            return redirect("login1")
        return view_func(request, *args, **kwargs)
    return wrapper


def authenticated_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        role = request.session.get("resense_role")
        if role not in ("user", "admin"):
            return redirect("index")
        return view_func(request, *args, **kwargs)
    return wrapper

def index(request):
    return render(request, 'index.html')


def registration(request):
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        mailid = request.POST.get("mailid", "").strip()
        mobileno = request.POST.get("mobileno", "").strip()
        uid = request.POST.get("uid", "").strip()
        pwd = request.POST.get("pwd", "")

        if not all([name, mailid, mobileno, uid, pwd]):
            return render(request, "registration.html", {
                "msg": "Please fill in all fields."
            })

        try:
            validation_user = User(username=uid, first_name=name, email=mailid)
            validate_password(pwd, user=validation_user)
        except ValidationError as exc:
            return render(request, "registration.html", {
                "msg": " ".join(exc.messages)
            })

        hashed_password = make_password(pwd)

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users
                    (name, mailid, mobileno, userid, password)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    [name, mailid, mobileno, uid, hashed_password]
                )

            return render(request, "registration.html", {
                "msg": "User registered successfully."
            })

        except IntegrityError:
            return render(request, "registration.html", {
                "msg": "This username is already registered."
            })

    return render(request, "registration.html")


def login(request):
    if request.method == "POST":
        uid = request.POST.get("uid", "").strip()
        pwd = request.POST.get("pwd", "")

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT password
                FROM users
                WHERE userid = %s
                """,
                [uid]
            )
            user = cursor.fetchone()

            if user is not None:
                stored_password = user[0]
                is_valid = False
                if isinstance(stored_password, str):
                    try:
                        is_valid = check_password(pwd, stored_password)
                    except (ValueError, TypeError, AssertionError, AttributeError,
                            OverflowError, ImproperlyConfigured):
                        # Malformed or unsupported stored hashes fail closed.
                        is_valid = False

                if is_valid:
                    request.session.flush()

                    request.session["resense_user_id"] = uid
                    request.session["resense_role"] = "user"

                    request.session.set_expiry(8 * 60 * 60)

                    return redirect("analyze")

        return render(request, "index.html", {
            "msg": "Your login attempt was not successful. Please try again!"
        })

    return render(request, "index.html")


def login1(request):
    if request.method == "POST":
        uid = request.POST.get("uid", "").strip()
        pwd = request.POST.get("pwd", "")

        is_admin = False
        if uid and pwd and uid in settings.RESENSE_ADMIN_USERIDS:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT userid, password FROM users WHERE userid = %s",
                        [uid]
                    )
                    accounts = cursor.fetchall()

                if len(accounts) == 1:
                    account_userid, stored_password = accounts[0]
                    if account_userid == uid and isinstance(stored_password, str):
                        try:
                            is_admin = check_password(pwd, stored_password)
                        except (ValueError, TypeError, AttributeError):
                            # Unsupported or malformed hashes fail closed.
                            is_admin = False
            except DatabaseError:
                is_admin = False

        if is_admin:
            request.session.flush()

            request.session["resense_user_id"] = account_userid
            request.session["resense_role"] = "admin"
            request.session.set_expiry(8 * 60 * 60)

            return redirect("/analyze1/")

        return render(request, "login.html", {
            "msg": "Your login attempt was not successful. Please try again!"
        })

    return render(request, "login.html")

@user_required
def analyze(request):
    """
    Calculate the documented Q-CHAT-10 screening score.

    A1-A9: Sometimes, Rarely, Never = 1.
    A10: Always, Usually, Sometimes = 1.
    This is a questionnaire scoring rule, not a clinical diagnosis.
    """
    if request.method != "POST":
        return render(request, "autism.html", {})

    response_options = {"Always", "Usually", "Sometimes", "Rarely", "Never"}
    answers = []

    for question_number in range(1, 11):
        answer = request.POST.get("t{}".format(question_number), "")
        if answer not in response_options:
            return render(request, "autism.html", {
                "error": "Please answer all 10 questionnaire items using the available response options."
            })
        answers.append(answer)

    score = sum(
        1 if answer in {"Sometimes", "Rarely", "Never"} else 0
        for answer in answers[:9]
    )
    score += 1 if answers[9] in {"Always", "Usually", "Sometimes"} else 0

    screening_positive = score > 3
    result_text = (
        "The questionnaire score is above the screening threshold. "
        "Potential ASD traits are indicated by this screening rule."
        if screening_positive else
        "The questionnaire score is at or below the screening threshold. "
        "Potential ASD traits are not indicated by this screening rule."
    )

    return render(request, "autism.html", {
        "score": score,
        "screening_positive": screening_positive,
        "result_text": result_text,
    })

@user_required
def find(request):
   return render(request, "find_artism.html")

@admin_required
def analyze1(request):
   return render(request, "analyze1.html")

@authenticated_required
def analyze4(request):
    db = MySQLdb.connect(
        host="127.0.0.1",
        user="root",
        passwd="",
        db="autism",
        port=3307
    )
    c1 = db.cursor()
    c1.execute("SELECT * FROM vfiles")
    row = c1.fetchall()
    return render(request, 'analyze4.html', {'data': row})


@admin_required
def analyze2(request):
    f = request.FILES['fi1']
    des = request.POST["descr"]

    full_path = os.path.join(os.getcwd(), "static", "uploads") + os.sep

    with open(os.path.join(full_path, f.name), 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

    full_path1 = os.path.join(full_path, f.name)

    db = MySQLdb.connect(
        host="127.0.0.1",
        user="root",
        passwd="",
        db="autism",
        port=3307
    )
    c1 = db.cursor()
    c1.execute(
        "INSERT INTO vfiles(vpath, vdescr) VALUES (%s, %s)",
        [f.name, des]
    )
    db.commit()

    return render(request, "analyze2.html")

@admin_required
def analyze3(request):
   db = MySQLdb.connect(
        host="127.0.0.1",
        user="root",
        passwd="",
        db="autism",
        port=3307
    )
   c1 = db.cursor()
   c1.execute("select * from vfiles")
   row=c1.fetchall()
   return render(request, 'analyze3.html', {'data': row})


@user_required
def output(request):
    uploaded_file = request.FILES.get('fi')
    if uploaded_file is None:
        return render(request, 'find_artism.html', {
            'msg': 'Choose a video file before uploading.'
        })

    project_dir = Path(__file__).resolve().parent.parent
    upload_dir = project_dir / 'static' / 'uploads'
    safe_name = Path(uploaded_file.name).name
    suffix = Path(safe_name).suffix
    upload_path = upload_dir / f'{uuid.uuid4().hex}{suffix}'

    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        with upload_path.open('wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        result = subprocess.run(
            [sys.executable, str(project_dir / 'main.py'), str(upload_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_dir),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return render(request, 'find_artism.html', {
            'msg': 'Video processing timed out. Please try a shorter video.'
        })
    except (OSError, subprocess.SubprocessError) as exc:
        return render(request, 'find_artism.html', {
            'msg': f'Video processing could not start or complete: {exc}'
        })

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return render(request, 'find_artism.html', {
            'msg': 'Video processing failed. ' + (detail or 'Please check the video file and try again.')
        })

    return render(request, 'find_artism.html', {
        'msg': 'Video uploaded and frames extracted successfully. This step only processes video frames; it does not detect autism or predict behavior.'
    })

@require_POST
def logout(request):
    request.session.flush()
    return redirect("index")
