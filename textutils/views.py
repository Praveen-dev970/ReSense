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

from django.db import connection, IntegrityError
from django.contrib.auth.hashers import make_password, check_password, identify_hasher
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
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

                try:
                    identify_hasher(stored_password)
                    is_valid = check_password(pwd, stored_password)

                except ValueError:
                    # Existing account with a legacy plaintext password.
                    is_valid = (
                        stored_password == pwd
                        if stored_password is not None
                        else False
                    )

                    if is_valid:
                        new_password = make_password(pwd)

                        cursor.execute(
                            """
                            UPDATE users
                            SET password = %s
                            WHERE userid = %s
                            """,
                            [new_password, uid]
                        )

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
        uid = request.POST.get("uid", "")
        pwd = request.POST.get("pwd", "")

        if uid == "admin" and pwd == "admin":
            request.session.flush()

            request.session["resense_role"] = "admin"
            request.session.set_expiry(8 * 60 * 60)

            return redirect("/analyze1/")

        return render(request, "login.html", {
            "msg": "Your login attempt was not successful. Please try again!"
        })

    return render(request, "login.html")

@user_required
def analyze(request):
    if request.method=="POST":
        q1 = int(request.POST["t1"])
        q2 = int(request.POST["t2"])
        q3 = int(request.POST["t3"])
        q4 = int(request.POST["t4"])
        q5 = int(request.POST["t5"])
        q6 = int(request.POST["t6"])
        q7 = int(request.POST["t7"])
        q8 = int(request.POST["t8"])
        q9 = int(request.POST["t9"])
        q10 = int(request.POST["t10"])
        age = int(request.POST["t11"])
        gender = request.POST["t13"]
        jaundice = request.POST["t14"]
        asd_family = request.POST["t15"]

        emotion = pd.read_csv("dataset2.csv")
        #attr=["fft_749_b","label"]

        value="None"
        with open('dataset2.csv') as file_obj:
            reader_obj = csv.reader(file_obj)
            i=1
            for row in reader_obj:
                if i == 1:
                    i = i + 1
                    continue
                data1 = int(row[1])
                data2 = int(row[2])
                data3 = int(row[3])
                data4 = int(row[4])
                data5 = int(row[5])
                data6 = int(row[6])
                data7 = int(row[7])
                data8 = int(row[8])
                data9 = int(row[9])
                data10 = int(row[10])
                data_age = int(row[11])
                data_gender = row[13]
                data_jaundice = row[15]
                data_asd_family = row[16]

                #print(data)
                if data1 == q1 and data2 == q2 and data3 == q3 and data4 == q4 and data5 == q5 and data6 == q6 and data7 == q7 and data8 == q8 and data9 == q9 and data10 == q10 and data_age == age and data_gender == gender and data_jaundice == jaundice and data_asd_family == asd_family:
                    value = row[18]
                    break
        return render(request, 'autism.html', {"autism":value})
    return render(request, 'autism.html', {})

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
