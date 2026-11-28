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
from .questionnaire import calculate_qchat10_score, exceeds_screening_threshold

from django.contrib.auth import (
    authenticate,
    login as auth_login,
    logout as auth_logout,
)
from django.db import transaction
from django.shortcuts import render, redirect, get_object_or_404
from .models import AutismScreeningResult, VideoAssessmentResult
import json


def user_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if (
            not request.user.is_authenticated
            or request.session.get("resense_role") != "user"
        ):
            return redirect("index")

        return view_func(request, *args, **kwargs)

    return wrapper


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if (
            not request.user.is_authenticated
            or not request.user.is_staff
            or not request.user.is_superuser
            or request.session.get("resense_role") != "admin"
        ):
            return redirect("login1")

        return view_func(request, *args, **kwargs)

    return wrapper


def authenticated_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if (
            not request.user.is_authenticated
            or request.session.get("resense_role") not in ("user", "admin")
        ):
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

        if len(uid) > 150:
            return render(request, "registration.html", {
                "msg": "Username must not exceed 150 characters."
            })

        validation_user = User(
            username=uid,
            first_name=name,
            email=mailid
        )

        try:
            validate_password(pwd, user=validation_user)
        except ValidationError as exc:
            return render(request, "registration.html", {
                "msg": " ".join(exc.messages)
            })

        try:
            with transaction.atomic():

                if User.objects.filter(username=uid).exists():
                    return render(request, "registration.html", {
                        "msg": "This username is already registered."
                    })

                # Check the legacy table to prevent duplicate usernames.
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT userid FROM users WHERE userid = %s",
                        [uid]
                    )

                    if cursor.fetchone() is not None:
                        return render(request, "registration.html", {
                            "msg": "This username is already registered."
                        })

                    # Django is now the authentication source.
                    User.objects.create_user(
                        username=uid,
                        first_name=name,
                        email=mailid,
                        password=pwd
                    )

                    # Retain mobile information in the legacy table.
                    # Do not store another password hash here.
                    cursor.execute(
                        """
                        INSERT INTO users
                        (name, mailid, mobileno, userid, password)
                        VALUES (%s, %s, %s, %s, NULL)
                        """,
                        [name, mailid, mobileno, uid]
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

        user = authenticate(
            request,
            username=uid,
            password=pwd
        )

        if (
            user is not None
            and user.is_active
            and not user.is_staff
            and not user.is_superuser
        ):
            auth_login(request, user)

            request.session["resense_user_id"] = user.username
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

        user = None

        if uid and pwd and uid in settings.RESENSE_ADMIN_USERIDS:
            user = authenticate(
                request,
                username=uid,
                password=pwd
            )

        if (
            user is not None
            and user.is_active
            and user.is_staff
            and user.is_superuser
        ):
            auth_login(request, user)

            request.session["resense_user_id"] = user.username
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
    Validate, score and save the Q-CHAT-10 questionnaire.
    Prevent duplicate submissions using a unique submission token.
    Preliminary screening support only.
    """

    if request.method != "POST":
        return render(request, "autism.html", {
            "submission_token": uuid.uuid4()
        })

    # Validate the submitted token
    token_value = request.POST.get("submission_token", "").strip()

    try:
        submission_token = uuid.UUID(token_value)
    except (ValueError, TypeError, AttributeError):
        return render(request, "autism.html", {
            "error": "Invalid submission token. Please submit the questionnaire again.",
            "submission_token": uuid.uuid4()
        })

    # Check whether this token has already been used
    existing_record = AutismScreeningResult.objects.filter(
        submission_token=submission_token
    ).first()

    if existing_record:
        if existing_record.user_id == request.user.id:
            return redirect(
                "screening_result",
                result_id=existing_record.id
            )

        return render(request, "autism.html", {
            "error": "This submission token has already been used. Please try again.",
            "submission_token": uuid.uuid4()
        })

    # Helper to preserve the token when validation fails
    def form_error(message):
        return render(request, "autism.html", {
            "error": message,
            "submission_token": submission_token
        })

    # Collect demographic information
    age_value = request.POST.get("age", "").strip()
    age_months_value = request.POST.get("age_months", "").strip()
    gender = request.POST.get("gender", "").strip()
    jaundice = request.POST.get("jaundice", "").strip()
    asd_family = request.POST.get("asd_family", "").strip()

    # Validate age
    if not age_value.isdigit():
        return form_error(
            "Please enter age as a whole number of years."
        )

    if not age_months_value.isdigit():
        return form_error(
            "Please select additional months."
        )

    age = int(age_value)
    age_months = int(age_months_value)

    if not 0 <= age_months <= 11:
        return form_error(
            "Additional months must be between 0 and 11."
        )

    total_age_months = age * 12 + age_months

    if not 18 <= total_age_months <= 30:
        return form_error(
            "This questionnaire is configured for children aged 18 to 30 months."
        )

    # Validate demographic choices
    if gender not in ("Male", "Female", "Other"):
        return form_error(
            "Please select a valid gender."
        )

    if jaundice not in ("Yes", "No"):
        return form_error(
            "Please select a valid jaundice response."
        )

    if asd_family not in ("Yes", "No"):
        return form_error(
            "Please select a valid family history response."
        )

    # Collect questionnaire answers
    answers = [
        request.POST.get("t{}".format(number), "").strip()
        for number in range(1, 11)
    ]

    try:
        score = calculate_qchat10_score(answers)
    except ValueError:
        return form_error(
            "Please answer all 10 questions using the available response options."
        )

    # Convert responses into database storage codes
    response_codes = {
        "Always": 0,
        "Usually": 1,
        "Sometimes": 2,
        "Rarely": 3,
        "Never": 4,
    }

    encoded_answers = [
        response_codes[answer] for answer in answers
    ]

    screening_positive = exceeds_screening_threshold(score)

    result_label = (
        "Above threshold"
        if screening_positive
        else "At or below threshold"
    )

    # Save the record with its unique submission token
    try:
        with transaction.atomic():
            screening = AutismScreeningResult.objects.create(
                user=request.user,
                submission_token=submission_token,
                q1=encoded_answers[0],
                q2=encoded_answers[1],
                q3=encoded_answers[2],
                q4=encoded_answers[3],
                q5=encoded_answers[4],
                q6=encoded_answers[5],
                q7=encoded_answers[6],
                q8=encoded_answers[7],
                q9=encoded_answers[8],
                q10=encoded_answers[9],
                age=age,
                age_months=age_months,
                gender=gender,
                jaundice=jaundice,
                asd_family=asd_family,
                result=result_label,
            )

    except IntegrityError:
        # Handle two requests attempting to use the same token
        existing_record = AutismScreeningResult.objects.filter(
            submission_token=submission_token
        ).first()

        if existing_record and existing_record.user_id == request.user.id:
            return redirect(
                "screening_result",
                result_id=existing_record.id
            )

        if existing_record:
            return form_error(
                "This submission token has already been used. Please try again."
            )

        raise

    # Post/Redirect/Get
    return redirect(
        "screening_result",
        result_id=screening.id
    )

@user_required
def screening_result(request, result_id):
    """
    Display a previously saved screening result.
    Users can only access their own records.
    """

    screening = get_object_or_404(
        AutismScreeningResult,
        id=result_id,
        user=request.user
    )

    # Convert stored response codes back to original answers
    stored_answers = [
        screening.q1,
        screening.q2,
        screening.q3,
        screening.q4,
        screening.q5,
        screening.q6,
        screening.q7,
        screening.q8,
        screening.q9,
        screening.q10,
    ]

    code_to_response = {
        0: "Always",
        1: "Usually",
        2: "Sometimes",
        3: "Rarely",
        4: "Never",
    }

    answers = [
        code_to_response[value]
        for value in stored_answers
    ]

    # Recalculate the score using the existing scoring helper
    score = calculate_qchat10_score(answers)

    screening_positive = screening.result == "Above threshold"

    result_text = (
        "The questionnaire score is above the screening threshold. "
        "Potential ASD traits are indicated by this screening rule."
        if screening_positive else
        "The questionnaire score is at or below the screening threshold. "
        "Potential ASD traits are not indicated by this screening rule."
    )

    return render(request, "autism.html", {
        "score": score,
        "result_ready": True,
        "screening_positive": screening_positive,
        "result_text": result_text,

        # Previously saved questionnaire answers
        "answers": answers,

        # Previously saved demographic information
        "saved_age": screening.age,
        "saved_age_months": screening.age_months,
        "saved_gender": screening.gender,
        "saved_jaundice": screening.jaundice,
        "saved_asd_family": screening.asd_family,
    })

@user_required
def screening_history(request):
    """
    Display screening records belonging only to the logged-in user.
    """

    records = AutismScreeningResult.objects.filter(
        user=request.user
    ).order_by('-created_at')

    return render(request, "screening_history.html", {
        "records": records,
    })


@user_required
def user_dashboard(request):
    """
    Display the logged-in user's latest assessments
    and separate assessment histories.
    """

    # Questionnaire records belonging to this user
    questionnaire_records = AutismScreeningResult.objects.filter(
        user=request.user
    ).order_by('-created_at')

    # Video assessment records belonging to this user
    video_records = VideoAssessmentResult.objects.filter(
        user=request.user
    ).order_by('-created_at')

    # Latest results
    latest_questionnaire = questionnaire_records.first()
    latest_video = video_records.first()

    context = {
        'latest_questionnaire': latest_questionnaire,
        'latest_video': latest_video,

        # Complete histories, newest first
        'questionnaire_history': questionnaire_records,
        'video_history': video_records,

        # Completion indicators
        'questionnaire_completed': latest_questionnaire is not None,
        'video_completed': latest_video is not None,

        # Overall summary is completion status only
        'both_completed': (
            latest_questionnaire is not None
            and latest_video is not None
        ),
    }

    return render(request, 'user_dashboard.html', context)



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
    if request.method == "POST":
        f = request.FILES['fi1']
        des = request.POST["descr"]

        full_path = os.path.join(
            os.getcwd(), "static", "uploads"
        ) + os.sep

        with open(os.path.join(full_path, f.name), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)

        db = MySQLdb.connect(
            host="127.0.0.1",
            user="root",
            passwd="",
            db="autism",
            port=3307
        )

        try:
            c1 = db.cursor()

            c1.execute(
                """
                INSERT INTO vfiles (vpath, vdescr, uploaded_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP(6))
                """,
                [f.name, des]
            )

            db.commit()

        except Exception:
            db.rollback()
            raise

        finally:
            c1.close()
            db.close()

        return render(request, "analyze2.html")

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

    try:
        if request.method == "POST":
            video_id = request.POST.get("video_id")

            if not video_id or not video_id.isdigit():
                return redirect("/analyze3/")

            # Find the video filename using its database ID
            c1.execute(
                "SELECT vpath FROM vfiles WHERE id = %s",
                [video_id]
            )
            video = c1.fetchone()

            if video:
                filename = video[0]

                # Prevent directory traversal
                if filename != os.path.basename(filename):
                    raise ValueError("Invalid video filename")

                uploads_dir = os.path.realpath(
                    os.path.join(os.getcwd(), "static", "uploads")
                )

                video_path = os.path.realpath(
                    os.path.join(uploads_dir, filename)
                )

                if os.path.commonpath(
                    [uploads_dir, video_path]
                ) != uploads_dir:
                    raise ValueError("Invalid video path")

                # Delete the physical video file if it exists
                if os.path.isfile(video_path):
                    os.remove(video_path)

                # Delete the corresponding database record
                c1.execute(
                    "DELETE FROM vfiles WHERE id = %s",
                    [video_id]
                )

                db.commit()

            return redirect("/analyze3/")

        # Display all uploaded videos
        c1.execute("SELECT * FROM vfiles ORDER BY id DESC")
        row = c1.fetchall()

        return render(
            request,
            "analyze3.html",
            {"data": row}
        )

    except Exception:
        db.rollback()
        raise

    finally:
        c1.close()
        db.close()





@user_required
def output(request):
    uploaded_file = request.FILES.get('fi')

    if uploaded_file is None:
        return render(request, 'find_artism.html', {
            'msg': 'Choose a video file before uploading.'
        })

    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    max_upload_size = 100 * 1024 * 1024

    safe_name = Path(uploaded_file.name).name
    suffix = Path(safe_name).suffix.lower()

    if suffix not in allowed_extensions:
        return render(request, 'find_artism.html', {
            'msg': 'Unsupported video format. Upload MP4, AVI, MOV or MKV.'
        })

    if uploaded_file.size > max_upload_size:
        return render(request, 'find_artism.html', {
            'msg': 'Video exceeds the maximum upload size of 100 MB.'
        })

    project_dir = Path(__file__).resolve().parent.parent
    upload_dir = project_dir / 'static' / 'uploads'
    upload_path = upload_dir / f'{uuid.uuid4().hex}{suffix}'

    try:
        upload_dir.mkdir(parents=True, exist_ok=True)

        with upload_path.open('wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        result = subprocess.run(
            [
                sys.executable,
                str(project_dir / 'process_video.py'),
                str(upload_path)
            ],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(project_dir),
            check=False,
        )

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()

            return render(request, 'find_artism.html', {
                'msg': (
                    'Video processing failed. '
                    + (detail or 'Please try again.')
                )
            })

        # Read the JSON response from process_video.py
        try:
            data = json.loads(result.stdout)

            if data.get('status') != 'success':
                raise ValueError(
                    'Video processing did not return success.'
                )

            prediction = data['prediction']

            predicted_class = prediction['predicted_class']
            model_score = float(prediction['model_score'])
            frames_processed = int(prediction['frames_processed'])
            selected_frame = prediction['selected_frame']

            total_frames = int(data['total_frames'])
            frames_saved = int(data['frames_saved'])

        except (ValueError, KeyError, TypeError):
            return render(request, 'find_artism.html', {
                'msg': (
                    'Video processing completed, but the model '
                    'returned an invalid response.'
                )
            })

        # Validate the model output
        if predicted_class not in {'ASD', 'TD'}:
            return render(request, 'find_artism.html', {
                'msg': 'The video model returned an unknown class.'
            })

        if not 0.0 <= model_score <= 1.0:
            return render(request, 'find_artism.html', {
                'msg': 'The video model returned an invalid score.'
            })

        if min(total_frames, frames_saved, frames_processed) < 0:
            return render(request, 'find_artism.html', {
                'msg': 'The video model returned invalid frame information.'
            })

        if not isinstance(selected_frame, str) or not selected_frame:
            return render(request, 'find_artism.html', {
                'msg': 'The video model returned an invalid selected frame.'
            })

        # Save the successful video assessment in the database
        VideoAssessmentResult.objects.create(
            user=request.user,
            predicted_class=predicted_class,
            model_score=model_score,
            total_frames=total_frames,
            frames_saved=frames_saved,
            frames_processed=frames_processed,
            selected_frame=selected_frame,
        )

        # Display the result
        return render(request, 'find_artism.html', {
            'msg': (
                'Video frame processing and experimental model '
                'inference completed successfully.'
            ),
            'video_prediction': predicted_class,
            'video_model_score': f'{model_score:.4f}',
            'video_frames_processed': frames_processed,
            'video_disclaimer': (
                'This is an experimental AI-assisted screening output. '
                'It is not a medical diagnosis and must not replace '
                'professional clinical assessment.'
            ),
        })

    except subprocess.TimeoutExpired:
        return render(request, 'find_artism.html', {
            'msg': (
                'Video processing timed out. '
                'Please try a shorter video.'
            )
        })

    except (OSError, subprocess.SubprocessError) as exc:
        return render(request, 'find_artism.html', {
            'msg': f'Video processing could not start or complete: {exc}'
        })

    finally:
        if upload_path.exists():
            upload_path.unlink()

@require_POST
def logout(request):
    auth_logout(request)
    return redirect("index")

@admin_required
def admin_screenings(request):
    """
    Display all questionnaire screening records to authorized admins.
    """

    records = AutismScreeningResult.objects.select_related(
        "user"
    ).order_by("-created_at")

    return render(request, "admin_screenings.html", {
        "records": records,
    })