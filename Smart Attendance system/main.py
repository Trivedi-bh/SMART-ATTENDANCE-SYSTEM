############################################# IMPORTS ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

############################################# CONFIG #################################################
BG = "#0f1724"
CARD = "#0b1220"
ACCENT = "#3b82f6"
ACCENT2 = "#10b981"
TEXT = "#e6eef8"
SUB = "#94a3b8"
ERROR = "#ef4444"

FONT_TITLE = ('Segoe UI', 18, 'bold')
FONT_SUB = ('Segoe UI', 11)
FONT_BOLD = ('Segoe UI', 12, 'bold')

CONF_THRESHOLD = 65

########################################### HELPERS #################################################
def assure_path_exists(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess._show(title='File Missing', message='haarcascade_frontalface_default.xml is missing. Add it and restart.')
        try:
            window.destroy()
        except Exception:
            pass

def valid_name(name):
    return bool(name) and all(c.isalpha() or c.isspace() for c in name)

def valid_usn(usn):
    return bool(usn) and usn.isalnum()

def ensure_training_label_dir():
    assure_path_exists("TrainingImageLabel")

def get_stored_password():
    ensure_training_label_dir()
    f = os.path.join("TrainingImageLabel", "psd.txt")
    if os.path.isfile(f):
        try:
            with open(f, "r") as fh:
                return fh.read()
        except Exception:
            return None
    return None

def set_stored_password(pwd):
    ensure_training_label_dir()
    f = os.path.join("TrainingImageLabel", "psd.txt")
    try:
        with open(f, "w") as fh:
            fh.write(pwd)
        return True
    except Exception:
        return False

########################################### DUPLICATE FACE #########################################
def is_face_duplicate(new_face):
    trainer_path = os.path.join("TrainingImageLabel", "Trainner.yml")
    if not os.path.isfile(trainer_path):
        return None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(trainer_path)
    except Exception:
        return None

    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    try:
        gray = cv2.cvtColor(new_face, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = new_face

    try:
        gray = cv2.resize(gray, (200, 200))
        gray = cv2.equalizeHist(gray)
    except Exception:
        pass

    faces = detector.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        face_crop = gray[y:y+h, x:x+w]
        try:
            serial, conf = recognizer.predict(face_crop)
            if conf < CONF_THRESHOLD:
                return serial
        except Exception:
            continue
    return None

########################################### TAKE IMAGES ################################################
def TakeImages():
    check_haarcascadefile()
    assure_path_exists("StudentDetails")
    assure_path_exists("TrainingImage")
    assure_path_exists("TrainingImageLabel")

    usn = txt.get().strip()
    name = txt2.get().strip()
    branch = branch_var.get()
    sem = sem_var.get()

    if not valid_usn(usn):
        mess._show("Input Error","USN must be alphanumeric and non-empty")
        return
    if not valid_name(name):
        mess._show("Input Error","Name must contain only letters and spaces")
        return
    if int(sem) < 1 or int(sem) > 8:
        mess._show("Input Error","Semester must be between 1 and 8")
        return

    students_csv = os.path.join("StudentDetails","StudentDetails.csv")

    if os.path.isfile(students_csv):
        try:
            df_check = pd.read_csv(students_csv)
            if 'USN' in df_check.columns and usn.lower() in df_check['USN'].astype(str).str.lower().values:
                mess._show(title='Duplicate USN', message='This USN is already registered!')
                return
            if 'NAME' in df_check.columns and name.lower() in df_check['NAME'].astype(str).str.lower().values:
                mess._show(title='Duplicate Name', message='This Name is already registered!')
                return
        except Exception:
            pass

    # determine serial number
    if os.path.isfile(students_csv):
        try:
            dfsd = pd.read_csv(students_csv)
            if 'SERIAL NO.' in dfsd.columns and len(dfsd) > 0:
                serial = int(dfsd['SERIAL NO.'].max()) + 1
            else:
                serial = 1
        except Exception:
            serial = 1
    else:
        serial = 1

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name=='nt' else cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    if not cam.isOpened():
        mess._show("Camera Error","Could not access camera")
        return

    ret, first_frame = cam.read()
    if not ret:
        mess._show("Camera Error","Cannot read from camera")
        cam.release()
        return

    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    try:
        gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray_first = first_frame

    faces_first = detector.detectMultiScale(gray_first,1.1,5)
    if len(faces_first)>0:
        (fx,fy,fw,fh)=faces_first[0]
        cropped_face=first_frame[fy:fy+fh,fx:fx+fw]
        dup_serial = is_face_duplicate(cropped_face)
    else:
        dup_serial = None

    if dup_serial is not None:
        if os.path.isfile(students_csv):
            try:
                df = pd.read_csv(students_csv)
                student = df.loc[df['SERIAL NO.']==dup_serial]
                if not student.empty:
                    reg_name = student['NAME'].values[0]
                    reg_usn = student['USN'].values[0]
                    mess._show(title="Duplicate Face", message=f"Face already registered:\n{reg_name}-{reg_usn}")
                    cam.release()
                    return
            except Exception:
                mess._show(title='Duplicate Face',message='Face already registered')
                cam.release()
                return

    sampleNum = 0
    last_face_time = time.time()

    window_name = "Registration - Press ESC or Q to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try: cv2.setWindowProperty(window_name,cv2.WND_PROP_TOPMOST,1)
    except: pass

    assure_path_exists("TrainingImage")

    while True:
        ret,img = cam.read()
        if not ret:
            mess._show("Camera Error","Camera disconnected")
            break

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray,1.1,5)
        if len(faces)>0:
            last_face_time = time.time()

        for (x,y,w,h) in faces:
            face = gray[y:y+h,x:x+w]
            try:
                face=cv2.resize(face,(200,200))
                face=cv2.equalizeHist(face)
            except:
                pass
            sampleNum+=1
            filename=f"{name}.{serial}.{usn}.{sampleNum}.jpg"
            filepath=os.path.join("TrainingImage",filename)
            try:
                cv2.imwrite(filepath,face)
            except Exception as e:
                print("Write failed:",e)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,200,120),2)
            cv2.putText(img,f"Img {sampleNum}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.putText(img,f"{sampleNum}/10 images",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)
        cv2.imshow(window_name,img)
        k=cv2.waitKey(1)&0xFF
        if k==27 or k==ord('q'):
            break
        if time.time()-last_face_time>6:
            break
        if sampleNum>=10:
            break

    cam.release()
    cv2.destroyAllWindows()

    if sampleNum>0:
        if os.path.isfile(students_csv):
            try:
                df=pd.read_csv(students_csv)
            except Exception:
                df=pd.DataFrame(columns=['SERIAL NO.','USN','NAME','BRANCH','SEM'])
        else:
            df=pd.DataFrame(columns=['SERIAL NO.','USN','NAME','BRANCH','SEM'])
        df.loc[len(df)]=[serial,usn,name,branch,sem]
        df.to_csv(students_csv,index=False)
        message1.configure(text=f"Images taken for USN: {usn} (Serial: {serial})")
        mess._show("Success",f"Images taken for {name}-{usn}")
    else:
        mess._show("Cancelled","No images captured")

########################################### TRAINING #################################################
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        try:
            pilImage=Image.open(imagePath).convert('L')
            imageNp=np.array(pilImage,'uint8')
            try:
                imageNp=cv2.resize(imageNp,(200,200))
                imageNp=cv2.equalizeHist(imageNp)
            except:
                pass
            filename=os.path.split(imagePath)[-1]
            parts=filename.split(".")
            if len(parts)>=4:
                try: ID=int(parts[1])
                except: continue
                if imageNp.shape[0]<50 or imageNp.shape[1]<50: continue
                faces.append(imageNp)
                Ids.append(ID)
        except Exception as e:
            print("Skipping file:",imagePath,e)
            continue
    return faces,Ids

def psw():
    ensure_training_label_dir()
    psd_file=os.path.join("TrainingImageLabel","psd.txt")
    stored=None
    if os.path.isfile(psd_file):
        try:
            with open(psd_file,'r') as f:
                stored=f.read()
        except:
            stored=None
    passwd=tsd.askstring('Password','Enter admin password',show='*')
    if passwd is None: return
    if stored is None:
        set_it=tsd.askstring('Set Password','No password found. Enter new:',show='*')
        if set_it:
            set_stored_password(set_it)
            mess._show('Password','Password saved. Click Train again.')
        return
    if passwd==stored:
        TrainImages()
    else:
        mess._show('Wrong Password','Incorrect password. Training aborted.')

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel")
    try:
        recognizer=cv2.face.LBPHFaceRecognizer_create()
    except:
        mess._show("Error","OpenCV 'face' module not found. Install opencv-contrib-python.")
        return

    dataPath="TrainingImage"
    if not os.path.exists(dataPath) or len([f for f in os.listdir(dataPath) if f.lower().endswith(('.jpg','.jpeg','.png'))])==0:
        mess._show("Error","No registered students found! Register at least one student.")
        return

    faces,Ids=getImagesAndLabels(dataPath)
    if len(faces)==0:
        mess._show("Error","Please register someone first!!!")
        return
    try:
        recognizer.train(faces,np.array(Ids))
        trainer_file=os.path.join("TrainingImageLabel","Trainner.yml")
        recognizer.save(trainer_file)
    except Exception as e:
        mess._show("Training Error",str(e))
        return

    res="Profile Saved Successfully"
    message1.configure(text=res)
    mess._show("Training","Training completed successfully!")

########################################### ATTENDANCE ################################################
def mark_attendance_once(usn,name,branch,sem):
    assure_path_exists("Attendance")
    filename="Attendance/Attendance.csv"
    if not os.path.isfile(filename):
        with open(filename,"w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["USN","NAME","BRANCH","SEM","DATE","TIME"])
    try:
        df=pd.read_csv(filename)
    except Exception:
        df=pd.DataFrame(columns=["USN","NAME","BRANCH","SEM","DATE","TIME"])
    today=datetime.datetime.now().strftime("%d-%m-%Y")
    if ((df['USN'].astype(str)==str(usn)) & (df['DATE']==today)).any():
        return
    now=datetime.datetime.now()
    df.loc[len(df)]=[usn,name,branch,sem,now.strftime("%d-%m-%Y"),now.strftime("%H:%M:%S")]
    df.to_csv(filename,index=False)

def TrackImages():
    trainer_path=os.path.join("TrainingImageLabel","Trainner.yml")
    if not os.path.isfile(trainer_path):
        mess._show("Error","Trainer not found. Please Train first.")
        return
    try:
        recognizer=cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(trainer_path)
    except:
        mess._show("Error","Failed to load trainer. Re-train model.")
        return
    faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    students_csv=os.path.join("StudentDetails","StudentDetails.csv")
    if not os.path.isfile(students_csv):
        mess._show("Error","StudentDetails missing. Register students first.")
        return
    df=pd.read_csv(students_csv)

    for k in tv.get_children():
        tv.delete(k)

    cam=cv2.VideoCapture(0,cv2.CAP_DSHOW) if os.name=='nt' else cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    if not cam.isOpened():
        mess._show("Camera Error","Unable to access camera")
        return

    window_name="Attendance - Press ESC to Exit"
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    try: cv2.setWindowProperty(window_name,cv2.WND_PROP_TOPMOST,1)
    except: pass

    font=cv2.FONT_HERSHEY_SIMPLEX
    last_seen_time=time.time()
    seen_usns=set()
    last_marked_time={}
    COOLDOWN_SECONDS=4

    while True:
        ret,img=cam.read()
        if not ret:
            mess._show("Camera Error","Unable to read from camera")
            break
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray,1.1,5)
        if len(faces)>0: last_seen_time=time.time()

        for (x,y,w,h) in faces:
            face=gray[y:y+h,x:x+w]
            try:
                face=cv2.resize(face,(200,200))
                face=cv2.equalizeHist(face)
            except: pass
            try:
                serial,conf=recognizer.predict(face)
            except: serial,conf=None,999
            display_text="Unknown"; color=(0,0,255)
            conf_text=f"Conf:{int(conf) if conf!=999 else '---'}"
            if serial is not None and conf<CONF_THRESHOLD:
                profile=df[df['SERIAL NO.']==serial]
                if not profile.empty:
                    usn=str(profile.iloc[0]['USN']).strip()
                    name=str(profile.iloc[0]['NAME']).strip()
                    branch=str(profile.iloc[0]['BRANCH']).strip()
                    sem=str(profile.iloc[0]['SEM']).strip()
                    now_ts=time.time()
                    last_t=last_marked_time.get(usn,0)
                    if now_ts-last_t>=COOLDOWN_SECONDS:
                        mark_attendance_once(usn,name,branch,sem)
                        last_marked_time[usn]=now_ts
                    display_text=f"{name} ({usn})"; color=(0,255,0)
                    if usn not in seen_usns:
                        seen_usns.add(usn)
                        ts=time.time()
                        date_str=time.strftime('%d-%m-%Y',time.localtime(ts))
                        time_str=time.strftime('%H:%M:%S',time.localtime(ts))
                        tv.insert('','end',text=usn,values=(name,branch,sem,date_str,time_str))
                else: display_text="Unknown"; color=(0,0,255)
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,display_text,(x,y-25),font,0.7,color,2)
            cv2.putText(img,conf_text,(x,y-5),font,0.6,color,1)

        cv2.imshow(window_name,img)
        k=cv2.waitKey(10)&0xFF
        if k==27 or k==ord('q'): break
        if time.time()-last_seen_time>15: break

    cam.release()
    cv2.destroyAllWindows()

########################################### GUI #################################################
window=tk.Tk()
window.geometry("1280x720")
window.resizable(True,False)
window.title("Attendance System")
window.configure(background=BG)

# Header
header=tk.Frame(window,bg=CARD,bd=0)
header.place(relx=0.02,rely=0.02,relwidth=0.96,relheight=0.12)
title=tk.Label(header,text="RYMEC Face Recognition Based Attendance System",bg=CARD,fg=ACCENT,font=FONT_TITLE)
title.pack(anchor='center')

# Left form
form=tk.Frame(window,bg=CARD)
form.place(relx=0.02,rely=0.16,relwidth=0.30,relheight=0.82)

tk.Label(form,text="USN",bg=CARD,fg=TEXT,font=FONT_SUB).pack(pady=5)
txt=tk.Entry(form,font=FONT_SUB)
txt.pack(pady=5)

tk.Label(form,text="Name",bg=CARD,fg=TEXT,font=FONT_SUB).pack(pady=5)
txt2=tk.Entry(form,font=FONT_SUB)
txt2.pack(pady=5)

tk.Label(form,text="Branch",bg=CARD,fg=TEXT,font=FONT_SUB).pack(pady=5)
branch_var=tk.StringVar()
branch_dropdown=ttk.Combobox(form,textvariable=branch_var,state='readonly',font=FONT_SUB)
branch_dropdown['values']=('CSE','ISE','ECE','EEE','MECH','CIVIL','AI&DS','M.Tech')
branch_dropdown.pack(pady=5)
branch_dropdown.current(0)

tk.Label(form,text="Semester",bg=CARD,fg=TEXT,font=FONT_SUB).pack(pady=5)
sem_var=tk.StringVar()
sem_dropdown=ttk.Combobox(form,textvariable=sem_var,state='readonly',font=FONT_SUB)
sem_dropdown['values']=tuple([str(i) for i in range(1,9)])
sem_dropdown.pack(pady=5)
sem_dropdown.current(0)

tk.Button(form,text="Take Images",command=TakeImages,bg=ACCENT,fg=TEXT,font=FONT_BOLD).pack(pady=15)
tk.Button(form,text="Train Images",command=psw,bg=ACCENT2,fg=TEXT,font=FONT_BOLD).pack(pady=15)
tk.Button(form,text="Track Attendance",command=TrackImages,bg=ACCENT,fg=TEXT,font=FONT_BOLD).pack(pady=15)

message1=tk.Label(form,text="",bg=CARD,fg=ACCENT,font=FONT_SUB)
message1.pack(pady=15)

# Right Table
tv_frame=tk.Frame(window,bg=CARD)
tv_frame.place(relx=0.34,rely=0.16,relwidth=0.64,relheight=0.82)

cols=("NAME","BRANCH","SEM","DATE","TIME")
tv=ttk.Treeview(tv_frame,columns=cols,show='headings')
for c in cols: tv.heading(c,text=c)
tv.pack(expand=True,fill='both')

window.mainloop()
