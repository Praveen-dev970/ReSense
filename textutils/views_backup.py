# I have created this file - tusharpangare ©

import pandas as pd   
from django.shortcuts import render, redirect,HttpResponse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import csv
import MySQLdb
import cv2
import os
from keras.models import load_model
import numpy as np
import subprocess

from django.db import connection, IntegrityError
from django.contrib.auth.hashers import make_password, check_password, identify_hasher

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
                    return redirect("analyze")

        return render(request, "index.html", {
            "msg": "Your login attempt was not successful. Please try again!"
        })

    return render(request, "index.html")


def login1(request):
    if request.method == "POST":
        uid = request.POST["uid"]
        pwd = request.POST["pwd"]
        print(uid,pwd)
        if uid=="admin" and pwd=="admin":
            return render(request, 'analyze1.html')
        return render(request, 'login.html', {"msg": "Your Login attempt was not successful. Please try again!!"})
    return render(request, 'login.html')

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

def find(request):
   return render(request, "find_artism.html")

def analyze1(request):
   return render(request, "analyze1.html")

def analyze4(request):
    db = MySQLdb.connect(
        host="127.0.0.1",
        user="root",
        passwd="",
        db="autism",
        port=3307
    )
    c1 = db.cursor()
    c1.execute("select * from vfiles")
    row = c1.fetchall()
    return render(request, 'analyze4.html', {'data': row})

def analyze2(request):
   f = request.FILES['fi1']
   des=request.POST["descr"]
   # Use dynamic path instead of hardcoded
   full_path = os.path.join(os.getcwd(), "static", "uploads") + os.sep
   print(full_path + "\n")
   with open(os.path.join(full_path, f.name), 'wb+') as destination:
       for chunk in f.chunks():
           destination.write(chunk)

   full_path1 = os.path.join(full_path, f.name)
   print(full_path1)
   db = MySQLdb.connect(
        host="127.0.0.1",
        user="root",
        passwd="",
        db="autism",
        port=3307
    )
   c1 = db.cursor()
   c1.execute("insert into vfiles(vpath,vdescr)values('%s','%s')" % (f.name, des))
   db.commit()
   return render(request, "analyze2.html")

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


def output(request):
    f=request.FILES['fi']
    # Save uploaded file first
    upload_path = os.path.join('static', 'uploads', f.name)
    with open(upload_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    
    # Path to video file
    print(f.name)
    # Run main.py with the uploaded file path
    subprocess.run(['python', 'main.py', upload_path])
    # video_path = f.name
    # output_folder = 'output_frames'
    #
    # # Create output directory if it doesn't exist
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    #
    # # Load the video
    # cap = cv2.VideoCapture(video_path)
    # frame_count = 0
    #
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     # Save each frame as an image file
    #     frame_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
    #     cv2.imwrite(frame_path, frame)
    #     frame_count += 1
    #
    # cap.release()
    # cv2.destroyAllWindows()
    # from keras.models import Sequential
    # from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    # from keras.preprocessing.image import ImageDataGenerator
    #
    # # Create a simple CNN model
    # model = Sequential()
    #
    # # First convolutional layer
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # Second convolutional layer
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # Flattening the layers
    # model.add(Flatten())
    #
    # # Full connection
    # model.add(Dense(units=128, activation='relu'))
    # model.add(Dense(units=1, activation='sigmoid'))
    #
    # # Compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    # # Preprocess training and validation data
    # train_datagen = ImageDataGenerator(rescale=1. / 255)
    # training_set = train_datagen.flow_from_directory('dataset/training_set',
    #                                                  target_size=(64, 64),
    #                                                  batch_size=32,
    #                                                  class_mode='binary')
    #
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_set = test_datagen.flow_from_directory('dataset/test_set',
    #                                             target_size=(64, 64),
    #                                             batch_size=32,
    #                                             class_mode='binary')
    #
    # # Train the model
    # model.fit(training_set,
    #           steps_per_epoch=8000 // 32,
    #           epochs=25,
    #           validation_data=test_set,
    #           validation_steps=2000 // 32)
    #
    #
    # # Load the trained model
    # model = load_model('autism_detection_model.h5')
    #
    # # Open the webcam
    # cap = cv2.VideoCapture(video_path)
    #
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     # Preprocess the frame
    #     frame_resized = cv2.resize(frame, (64, 64))
    #     frame_array = np.array(frame_resized)
    #     frame_array = frame_array / 255.0
    #     frame_array = np.expand_dims(frame_array, axis=0)
    #
    #     # Predict using the model
    #     prediction = model.predict(frame_array)
    #     print(prediction)
    #     # Interpret the prediction
    #     if prediction > 0.47:
    #         label = 'Autism Behavior Detected'
    #     else:
    #         label = 'Normal Behavior'
    #
    #     # Display the prediction
    #     cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #     cv2.imshow('Autism Detection', frame)
    #
    #     # Break loop on 'q' key press
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
    return redirect(find)
    #return render(request, 'find_artism1.html')