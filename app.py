import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from flask import Flask, render_template, request, redirect, url_for, session
import re
import shutil
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy  import SQLAlchemy
from flask_login import  UserMixin


#### Defining Flask App
app = Flask(__name__)

db_path = os.path.join(os.path.dirname(__file__), 'healthcare.db')
db_uri = 'sqlite:///{}'.format(db_path)
app.config['SECRET_KEY'] = 'helloworld'
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
db=SQLAlchemy(app)



class userregister(UserMixin, db.Model):
    Patient_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(50))
    gender=db.Column(db.String(50))
    DOB=db.Column(db.String, nullable=False)
    phn=db.Column(db.String(50))
    addr=db.Column(db.String(80))
    bldgrp=db.Column(db.String(15))
    email = db.Column(db.String(50))
    psw = db.Column(db.String(80))
    ename=db.Column(db.String(50))
    relation=db.Column(db.String(15))
    phone=db.Column(db.String(50))

    def __init__(self, name,gender,dob,phn,addr,bldgrp,email,hashed_psw,ename,relation,phone):
        self.name=name
        self.gender=gender
        self.DOB=dob
        self.phn=phn
        self.addr=addr
        self.bldgrp=bldgrp
        self.email=email
        self.psw=hashed_psw
        self.ename=ename
        self.relation=relation
        self.phone=phone

class hospitalregister(db.Model):

        hid = db.Column(db.Integer, primary_key=True,autoincrement=True)
        name = db.Column(db.String(40))
        psw = db.Column(db.String(200))
        loc = db.Column(db.String(50))
        area= db.Column(db.String(50))
        pin = db.Column(db.String(50))
        state = db.Column(db.String(50))
        city = db.Column(db.String(50))

        def __init__(self, name, hashed_psw, loc,area,pin,state,city):
            self.name = name
            self.loc = loc
            self.area = area
            self.pin = pin
            self.state = state
            self.city = city
            self.psw = hashed_psw

class policeregister(db.Model):
    pid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(40))
    psw = db.Column(db.String(200))
    loc = db.Column(db.String(50))
    area = db.Column(db.String(50))
    pin = db.Column(db.String(50))
    state = db.Column(db.String(50))
    city = db.Column(db.String(50))

    def __init__(self, name, hashed_psw, loc, area, pin, state, city):
        self.name = name
        self.loc = loc
        self.area = area
        self.pin = pin
        self.state = state
        self.city = city
        self.psw = hashed_psw

class medical_reports(db.Model):
    mid=db.Column(db.Integer, primary_key=True, autoincrement=True)
    id = db.Column(db.Integer, db.ForeignKey('userregister.Patient_id'), nullable=False,primary_key=False)
    Dname = db.Column(db.String(40))
    date = db.Column(db.String, nullable=False)
    sym = db.Column(db.String(200))
    doc = db.Column(db.String(50))
    allergies = db.Column(db.String(50))
    reports = db.Column(db.String(250))

    def __init__(self, id,Dname, date, sym,Doc, allergies,file):
        self.id = id
        self.Dname = Dname
        self.date = date
        self.sym = sym
        self.doc = Doc
        self.allergies = allergies
        self.reports = file

@app.route('/')
@app.route('/login', methods =['GET', 'POST'])
def login():
    msg = ''

    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        # encoding user password


        user = userregister.query.filter_by(name=username).first()
        if user:
            result=check_password_hash(user.psw, password)
        hospital = hospitalregister.query.filter_by(name=username).first()
        print(hospital)
        if hospital:
            result=check_password_hash(hospital.psw, password)

        police = policeregister.query.filter_by(name=username).first()
        if police:
            result=check_password_hash(police.psw, password)

        if user and result :
            session['loggedinu'] = True
            session['id'] = user.Patient_id
            session['username'] = user.name
            session['phn']=user.phn
            msg = 'Logged in successfully !'

            row_details = display_details(user.phn)
            row_records = display_reports(user.Patient_id)
         

            return render_template('index_user.html', msg = msg, rows =row_records , row1=row_details)
        elif hospital and result:
            session['loggedinh'] = True
           
            session['hid'] = hospital.hid
            session['username'] = hospital.name
            msg = 'Logged in successfully !'
            row=hospital_details(session['hid'])
            return render_template('index_hospital.html', msg=msg,row=row)
        elif police and result:
            session['loggedinp'] = True
            session['pid'] = police.pid
            session['username'] = police.name
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

        hashed_psw = generate_password_hash(psw, method='sha256')


        account=hospitalregister.query.filter_by(name=name).all()
       
        if account:
            msg = 'Account already exists !'
        elif not name or not psw :
            msg = 'Please fill out the form !'
        else:
            new_user=hospitalregister(name, hashed_psw, loc,area,pin,state,city)
            db.session.add(new_user)
            db.session.commit()
            msg = 'You have successfully registered !'
            return render_template('login.html',msg=msg)
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
        hashed_psw = generate_password_hash(psw, method='sha256')

        account=policeregister.query.filter_by(name=name).all()
        if account:
            msg = 'Account already exists !'
        elif not name or not psw :
            msg = 'Please fill out the form !'
        else:
            new_user = policeregister(name, hashed_psw, loc, area, pin, state, city)
            db.session.add(new_user)
            db.session.commit()
            msg = 'You have successfully registered !'
            return render_template('login.html',msg=msg)
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
        hashed_psw = generate_password_hash(psw, method='sha256')
        account = userregister.query.filter_by(name=name).all()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not name or not psw or not email:
            msg = 'Please fill out the form !'
        else:
            new_user = userregister(name,gender,dob,phn,addr,bldgrp,email,hashed_psw,ename,relation,phone)
            db.session.add(new_user)
            db.session.commit()
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

        report = medical_reports(session['id'],Dname, date, sym,Doc, allergies,file)
        db.session.add(report)
        db.session.commit()

        msg = 'You have successfully uploaded the report ! '
        row_details = display_details(session['phn'])
        
        row_records = display_reports(session['id'])

        return render_template('index_user.html', msg=msg, rows=row_records, row1=row_details)
    return render_template('registerm.html')

@app.route('/update', methods =['GET', 'POST'])
def update():
    msg = ''
    if request.method == 'POST':
      id = session['id']
      user = userregister.query.filter_by(Patient_id=id).first()
 
      user.name = request.form['name']
      user.gender = request.form['gender']
      user.dob = request.form['DOB']
      user.phn = request.form['phn']
      user.addr = request.form['addr']
      user.bldgrp = request.form['bldgrp']
      user.email = request.form['email']
      user.psw = request.form['psw']
      user.ename = request.form['ename']
      user.relation = request.form['relation']
      user.phone = request.form['phone']
      psw = request.form['psw']

      hashed_psw = generate_password_hash(psw, method='sha256')
      user.psw=hashed_psw

      db.session.commit()


      msg = 'You have successfully Updated your Details !'
      row_details = display_details(user.phn)
      row_records = display_reports(user.Patient_id)
      return render_template('index_user.html', msg=msg, rows=row_records, row1=row_details)
    return render_template('update.html')


@app.route('/index_hospital', methods =['GET', 'POST'])
def index_hospital():
    msg=''
    row_records = hospital_details(session['hid'])

    return render_template('index_hospital.html', msg=msg, row=row_records)
@app.route('/xyz', methods =['GET', 'POST'])
def xyz():
    return render_template('xyz.html')

@app.route('/index_police', methods =['GET', 'POST'])
def index_police():
    msg=''
    row_records = police_details(session['pid'])

    return render_template('index_police.html', msg=msg, row=row_records)

@app.route('/index_user', methods =['GET', 'POST'])
def index_user():
    msg=''
    row_details = display_details(session['phn'])
    row_records = display_reports(session['id'])

    return render_template('index_user.html', msg=msg, rows=row_records, row1=row_details)

@app.route('/healthInsurance', methods =['GET', 'POST'])
def healthInsurance():
    return render_template('healthInsurance.html')

def hospital_details(id):
           row= hospitalregister.query.filter_by(hid=id).all()# if name is in row then return that row , row is a dictionary
           if(len(row)):
             return (row[0])


def police_details(id):
           row= policeregister.query.filter_by(pid=id).all()# if name is in row then return that row , row is a dictionary
           if(len(row)):
             return (row[0])

def display_details(phn):
           row = userregister.query.filter_by(phn=phn).all()
    
           if(len(row)):
             return (row[0])
           return render_template('UnregisteredUser.html')

def display_reports(id):
    r=[]

    myresult = medical_reports.query.filter_by(id=id).all() # fetching all rows of the table
    for row in myresult:
        if (id == row.id):# if name is in row then return that row
          
            l={'Dname':row.Dname,'date':row.date,'sym':row.sym,'doc':row.doc,'allergies':row.allergies,'reports':row.reports}
            r.append(l)
    return (r)




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
      if os.path.exists(f'static/faces/{user}'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')



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
           
            row_details=display_details(identified_person)
            row_records=display_reports(row_details.Patient_id)
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
    shutil.rmtree(userimagefolder)
    msg='Yor Face has been registered successfully! Now you can login'
    return render_template('login.html' ,msg=msg)


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
