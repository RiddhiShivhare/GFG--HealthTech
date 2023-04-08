import cv2
import os
from flask import Flask,request,render_template,flash,redirect
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re


#### Defining Flask App
app = Flask(__name__)


app.secret_key = 'your secret key'
 
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'alapk26021973@'
app.config['MYSQL_DB'] = 'healthcare'
 
mysql = MySQL(app)


@app.route('/')
@app.route('/login', methods =['GET', 'POST'])
def login():
    msg = ''
    #print(request.method)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM userregister WHERE name = % s AND psw = % s', (username, password, ))
        account = cursor.fetchone()
        cursor.execute('SELECT * FROM hospitalregister WHERE name = % s AND psw = % s', (username, password,))
        account1=cursor.fetchone()
        cursor.execute('SELECT * FROM policeregister WHERE name = % s AND psw = % s', (username, password,))
        account2 = cursor.fetchone()
        if account:
            session['loggedinu'] = True
            session['id'] = account['Patient_id']
            session['username'] = account['name']
            session['phn']=account['phn']
            msg = 'Logged in successfully !'

            row_details = display_details(account['phn'])
            row_records = display_reports(account['Patient_id'])
            print(row_records)

            return render_template('index_user.html', msg = msg, rows =row_records , row1=row_details)
        elif account1:
            session['loggedinh'] = True
            session['hid'] = account1['hid']
            session['username'] = account1['name']
            msg = 'Logged in successfully !'
            row=hospital_details(session['hid'])
            return render_template('index_hospital.html', msg=msg,row=row)
        elif account2:
            session['loggedinp'] = True
            session['pid'] = account2['pid']
            session['username'] = account2['name']
            msg = 'Logged in successfully !'
            row = police_details(session['pid'])
            return render_template('index_police.html', msg=msg, row=row)

        else:
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg = msg)
 
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))
 
@app.route('/registerh', methods =['GET', 'POST'])
def registerh():
    msg = ''
    if request.method == 'POST' and 'name' in request.form and 'psw' in request.form and 'loc' in request.form and 'area' in request.form and 'pin' in request.form and 'state' in request.form and 'city' in request.form :
        name = request.form['name']
        psw = request.form['psw']
        loc = request.form['loc']
        area = request.form['area']
        pin = request.form['pin']
        state = request.form['state']
        city = request.form['city']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM hospitalregister WHERE name = % s', (name, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not name or not psw :
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO hospitalregister VALUES (NULL, % s, % s, % s, % s, % s, % s, % s)', (name, psw, loc,area,pin,state,city ,))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
            return render_template('login.html')
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('registerh.html', msg = msg)

@app.route('/registerp', methods =['GET', 'POST'])
def registerp():
    msg = ''
    if request.method == 'POST' and 'name' in request.form and 'psw' in request.form and 'loc' in request.form and 'area' in request.form and 'pin' in request.form and 'state' in request.form and 'city' in request.form :
        name = request.form['name']
        psw = request.form['psw']
        loc = request.form['loc']
        area = request.form['area']
        pin = request.form['pin']
        state = request.form['state']
        city = request.form['city']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM policeregister WHERE name = % s', (name, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not name or not psw :
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO policeregister VALUES (NULL, % s, % s, % s, % s, % s, % s, % s)', (name, psw, loc,area,pin,state,city ,))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
            return render_template('login.html')
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('registerp.html', msg = msg)

@app.route('/registeru', methods =['GET', 'POST'])
def registeru():
    msg = ''
    if request.method == 'POST' and 'name' in request.form and 'gender' in request.form and 'DOB' in request.form and 'phn' in request.form and 'addr' in request.form and 'bldgrp' in request.form and 'email' in request.form and 'psw' in request.form and 'ename' in request.form and 'relation' in request.form and 'phone' in request.form  :
        name = request.form['name']
        gender=request.form['gender']
        dob=request.form['DOB']
        phn=request.form['phn']
        addr=request.form['addr']
        bldgrp = request.form['bldgrp']
        email = request.form['email']
        psw=request.form['psw']
        ename=request.form['ename']
        relation =request.form['relation']
        phone=request.form['phone']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM userregister WHERE name = % s', (name, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not name or not psw or not email:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO userregister VALUES (NULL, % s, % s, % s, % s, % s, % s, % s, % s, % s, % s, % s)', (name,gender,dob,phn,addr,bldgrp,email,psw,ename,relation,phone, ))
            mysql.connection.commit()
            msg = 'You have successfully registered ! Now Register your Face'
            return render_template('register_face.html', msg=msg)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('registeru.html', msg = msg)

@app.route('/registerm', methods =['GET', 'POST'])
def registerm():
    msg=''
    if request.method == 'POST'and 'Dname' in request.form:
        Dname= request.form['Dname']
        sym= request.form['sym']
        date= request.form['date']
        Doc= request.form['doc']
        allergies= request.form['allergies']
        file = request.form['reports']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('INSERT INTO medical_records VALUES (% s, % s, % s, % s, % s, % s, %s)',
                       (session['id'],Dname, date, sym,Doc, allergies,file))
        mysql.connection.commit()
        msg = 'You have successfully uploaded the report ! '
        row_details = display_details(session['phn'])
        # print(account['Patient_id'])
        row_records = display_reports(session['id'])

        return render_template('index_user.html', msg=msg, rows=row_records, row1=row_details)
    return render_template('registerm.html')



@app.route('/index_hospital', methods =['GET', 'POST'])
def index_hospital():
    msg=''
    row_records = hospital_details(session['hid'])
    #print(row_records)
    return render_template('index_hospital.html', msg=msg, row=row_records)

@app.route('/index_police', methods =['GET', 'POST'])
def index_police():
    msg=''
    row_records = police_details(session['pid'])
    #print(row_records)
    return render_template('index_police.html', msg=msg, row=row_records)

@app.route('/index_user', methods =['GET', 'POST'])
def index_user():
    msg=''
    row_details = display_details(session['phn'])
    row_records = display_reports(session['id'])
    print(row_records)
    return render_template('index_user.html', msg=msg, rows=row_records, row1=row_details)

@app.route('/healthInsurance', methods =['GET', 'POST'])
def healthInsurance():
    return render_template('healthInsurance.html')

@app.route('/update', methods =['GET', 'POST'])
def update():
    msg = ''
    if request.method == 'POST':
      id = session['id']
      name = request.form['name']
      gender = request.form['gender']
      dob = request.form['DOB']
      phn = request.form['phn']
      addr = request.form['addr']
      bldgrp = request.form['bldgrp']
      email = request.form['email']
      psw = request.form['psw']
      ename = request.form['ename']
      relation = request.form['relation']
      phone = request.form['phone']
      cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
      cursor.execute("""
        UPDATE userregister 
        SET name=%s,gender=%s,dob=%s,phn=%s,addr=%s,bldgrp=%s,email=%s,psw=%s,ename=%s,relation=%s,phone=%s
         WHERE Patient_id=%s""",(name,gender,dob,phn,addr,bldgrp,email,psw,ename,relation,phone,id))
      mysql.connection.commit()
      msg = 'You have successfully Updated your Details !'
      row_details = display_details(session['phn'])
      row_records = display_reports(session['id'])
      return render_template('index_user.html', msg=msg, rows=row_records, row1=row_details)
    return render_template('update.html')

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')


#### extract the face from an image
def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')

def hospital_details(id):
   #fetch reports of phn no. mentioned
   cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
   cursor.execute("Select * from hospitalregister")
   myresult = cursor.fetchall()  # fetching all rows of the table
   for row in myresult:
       #print(row)
       if (id == row['hid']):  # if name is in row then return that row , row is a dictionary
           return (row)


def police_details(id):
   #fetch reports of phn no. mentioned
   cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
   cursor.execute("Select * from policeregister")
   myresult = cursor.fetchall()  # fetching all rows of the table
   for row in myresult:
       #print(row)
       if (id == row['pid']):  # if name is in row then return that row , row is a dictionary
           return (row)

def display_details(phn):
   #fetch reports of phn no. mentioned
   cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
   cursor.execute("Select * from userregister")
   myresult = cursor.fetchall()  # fetching all rows of the table
   for row in myresult:
       #print(row)
       if phn in row['phn']:  # if name is in row then return that row , row is a dictionary
           return (row)

def display_reports(id):
    r=[]
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("Select * from medical_records")
    myresult = cursor.fetchall()  # fetching all rows of the table
    for row in myresult:
        if (id == row['id']):# if name is in row then return that row
            if(row['reports']==''):
                row['reports']='https://drive.google.com/file/d/1oFkHtET8WNYl1S8wDsYBDqfjt_4u3_A-/view?usp=share_link'
            r.append(row)
    return (r)



#### This function will run when we click on Sacn Face Button
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('index.html',mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    count=0
    while ret:
        count+=1
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            print((identified_person))
            row_details=display_details(identified_person)
            row_records=display_reports(row_details['Patient_id'])
            return render_template('hospital_display.html', row=row_details,row1=row_records)

        elif count>5:
            return render_template('UnregisteredUser.html')


@app.route('/startp',methods=['GET'])
def startp():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('index.html',mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    count=0
    while ret:
        count+=1
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            row_details=display_details(identified_person)
            return render_template('police_display.html', row1=row_details)
        elif count>5:
            return render_template('UnregisteredUser.html')



#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    phn = request.form['phn']
    #newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+str(phn)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = phn+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    msg='Yor Face has been registered successfully! Now you can login'
    return render_template('login.html' ,msg=msg)


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
