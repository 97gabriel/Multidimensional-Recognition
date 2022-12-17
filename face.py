import os
import face_recognition as fr
import numpy, pickle
import cv2
import dlib
from deepface import DeepFace

TOLERANCE = 0.8
data = []

db = "DB/"
DETECTOR = "dlib" #['opencv', 'ssd', 'dlib', 'mtcnn']
MODEL = "VGG-Face" #["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"] 

G = "\33[92m";R = "\33[91m"
Y = "\33[93m";A = "\33[90m"
C = "\33[0m"



def encode(data_path):
  global data
  data = []
  print(G+"Loading known faces from DB \33[0m \n"+C)
  try:
    for name in os.listdir(db):
      for filename in os.listdir(f"{db}/{name}"):
        if filename.lower().endswith(('.png','.jpg','.jpeag')):
          known_face = []
          known_name = []
          image = fr.load_image_file(f"{db}/{name}/{filename}")
          encoding = fr.face_encodings(image)[0]
          known_face.append(encoding)
          known_name.append(f"{name}/{filename}")
          print(f"Face encoded: {Y+name+C} IMG: {A+filename+C}")
          data.append(known_name + numpy.array(known_face).tolist())
  except NotADirectoryError:
    pass
#   appendData()
  
def appendData():
#   with open("DB/representations_vgg_face_MANUAL.pkl", "wb") as f:
#     pickle.dump(data, f)
  try:
    DeepFace.find(img_path="", db_path=db, model_name=MODEL, distance_metric=DISTANCE,
                          detector_backend=DETECTOR, enforce_detection="True")
  except ValueError:
    pass


def detector(img_path):
  img = cv2.imread(str(img_path))
  detector = dlib.cnn_face_detection_model_v1("detector/face_detector.dat")
  # apply face detection (cnn)
  faces_cnn = detector(img, 1)
  i=0
  # loop over detected faces
  for face in faces_cnn:
      x = face.rect.left()
      y = face.rect.top()
      w = face.rect.right() - x
      h = face.rect.bottom() - y
      i = i+1
      
      global obj, f_name
      # draw box over face
      frame = cv2.rectangle(img,(x,y), (x+w,y+h), (0,0,255), 2)
      # frame = img[y:y+h, x:x+w]
      obj = DeepFace.find( img_path=frame, db_path=db, model_name=MODEL, distance_metric="euclidean",
                          detector_backend=DETECTOR, enforce_detection="True")
      # print(obj)
      f_name = str(obj[["identity", "VGG-Face_euclidean"]].head().max()).split('/')[-2:][0]
      cv2.putText(frame, f_name, (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
      cv2.imwrite("tmp/face.jpg", frame)    # Save Image Frame to local directory
  

  O = "\33[92m"
  C = "\33[0m"
  print(f"{O} Detector: {C+DETECTOR+O} && Model: {C + MODEL}")
  print(f"{O} Face found {C + str(i)}")
  print(f'Name/s: {C + str(obj[["identity", "VGG-Face_euclidean"]].head().max())+ O}\n')
  # print(obj.head().max())
  
  # cv2.imshow(f_name,img)
  cv2.waitKey(1)
  cv2.destroyAllWindows()


def nameFound():
  return f_name



# def find(img_path):
#   inp_image = fr.load_image_file(img_path)
#   location = fr.face_locations(inp_image, model=MODEL)
#   encoding = fr.face_encodings(inp_image, location)
#   inp_image = cv2.cvtColor(inp_image, cv2.COLOR_RGB2BGR)
#   count = 0
#   for face_encoding, face_location in zip(encoding, location):
#     count = count+1
#     with open(os.fspath("encoded/"), "rb")as faces:
#       for f in faces:
#         face_encoding = numpy.load(f)
#         result = fr.compare_faces(known_face, face_encoding, tolerance=TOLERANCE)
#         face_distance = fr.face_distance(known_face, face_encoding)
#         match = None
#         if True in result:
#           match = known_name[result.index(True)]
#           print(f"Person found: {match} [Face_{count}]")


# encode(db)
# print(f"\n{G}Face found in DB: {R+str(len(known_face))+C}")
# data = detector("0.jpg")

# find("0.jpg")
