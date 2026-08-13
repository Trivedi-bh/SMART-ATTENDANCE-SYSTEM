# Face Recognition Based Attendance System

A simple **Face Recognition Attendance System** developed using **Python, Tkinter, OpenCV, and LBPH Face Recognition**. The system captures student face images, trains a face recognition model, and automatically marks attendance when a registered student is recognized through the webcam.

## Features

* Student registration with USN, Name, Branch, and Semester
* Captures face images using a webcam
* Detects faces using OpenCV Haar Cascade
* Trains faces using LBPH Face Recognizer
* Recognizes registered students automatically
* Automatically records attendance with date and time
* Prevents duplicate attendance on the same day
* Tkinter-based graphical user interface
* Stores student and attendance details in CSV files
* Admin password protection for model training

## Technologies Used

* Python
* Tkinter
* OpenCV
* LBPH Face Recognition
* NumPy
* Pandas
* Pillow
* CSV
* Haar Cascade Classifier

## How It Works

1. Enter the student's **USN, Name, Branch, and Semester**.
2. Click **Take Images** to capture face samples using the webcam.
3. Click **Train Images** to train the LBPH face recognition model.
4. Click **Track Attendance** to start face recognition.
5. When a registered student is recognized, the system automatically records their attendance.
6. Attendance is saved with **USN, Name, Branch, Semester, Date, and Time**.

## Project Structure

```text
Face-Recognition-Attendance/
│
├── Attendance/
│   └── Attendance.csv
│
├── StudentDetails/
│   └── StudentDetails.csv
│
├── TrainingImage/
│   └── Face Images
│
├── TrainingImageLabel/
│   ├── Trainner.yml
│   └── psd.txt
│
├── haarcascade_frontalface_default.xml
└── main.py
```

## Installation

Install the required Python libraries:

```bash
pip install opencv-contrib-python numpy pandas pillow
```

Tkinter is normally included with Python on Windows.

## Run the Project

```bash
python main.py
```

Then use the application to register students, train the model, and track attendance.

## Output

The system displays recognized students in the application and stores attendance in:

```text
Attendance/Attendance.csv
```

Each attendance record contains:

* USN
* Name
* Branch
* Semester
* Date
* Time

## Future Improvements

* Improve recognition accuracy using advanced deep learning models
* Add a database such as MySQL or MongoDB
* Add an attendance report/export feature
* Add an admin login system
* Develop a web or mobile-based interface

## Author

**TRIVEDI BH**
