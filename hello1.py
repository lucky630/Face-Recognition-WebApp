## python -m http.server
## from the output folder to open http on 8000 port

from flask import Flask, render_template, request,Response
import os
from werkzeug import secure_filename
import recog
from client4 import Vidcamera
from own_pc import Vidcamera1
import video_recog1

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = r'C:\Users\gurvinder1.singh\Downloads\Facial-Similarity-with-Siamese-Networks-in-Pytorch-master\data\input_fold'
app.config['OUTPUT_FOLDER'] = r'C:\Users\gurvinder1.singh\Downloads\Facial-Similarity-with-Siamese-Networks-in-Pytorch-master\data\output_fold'

### front page 
@app.route('/')
def front_page():
   return render_template('index1.html')

### image processing
@app.route('/main1')
def main_page():
   return render_template('image_load1.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'inp.jpg'))
      #name = os.system("python recog.py")
      name = recog.identify_face()
      print(name)
      if len(name)==0:
          name="Not Able to Recognize!!"
      else:
          name='Hi '+name[0].split(':')[0]+'!'
      return render_template('image_load.html',ident=name)
      #return 'file uploaded successfully'

### Remote camera processing

@app.route('/video')
def index():
    return render_template('index.html')

def gen(camera):
    print('gen camera method')
    while True:
        frame = camera.framing()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    print('end of gen camera method')

@app.route('/video_feed')
def video_feed():
    print('video_feed method')
    aa=gen(Vidcamera())
    print('video_feed method 2')
    return Response(aa,mimetype='multipart/x-mixed-replace; boundary=frame')


## for own computer camera processing
@app.route('/video_1')
def index_1():
    return render_template('index_1.html')

def gen_1(camera):
    print('gen camera method')
    while True:
        frame = camera.framing()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    print('end of gen camera method')

@app.route('/video_feed_1')
def video_feed_1():
    print('video_feed method')
    aa=gen_1(Vidcamera1())
    print('video_feed method 2')
    return Response(aa,mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
   app.run(host= '0.0.0.0', debug = True)
