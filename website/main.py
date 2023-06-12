# -*- coding: utf-8 -*-
import cv2
import utils
import base64
import numpy as np
import tensorflow as tf

from collections import defaultdict
from flask import Flask, render_template, request


SIAMESE_PAIRS_MODEL_FNAME = 'siamese_network_pairs.h5'
SIAMESE_TRIPLES_MODEL_FNAME = 'siamese_network_triples.h5'
INPUT_SIZE = 150

TRIPLET_LOSS_MARGIN = 0.2

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')


siamese_network_pairs = tf.keras.models.load_model(SIAMESE_PAIRS_MODEL_FNAME, custom_objects={'contrastive_loss': utils.loss(1)})
siamese_network_triples = tf.keras.models.load_model(SIAMESE_TRIPLES_MODEL_FNAME, custom_objects={'identity_loss': utils.identity_loss})


query_labels = {}
image_list, train_y_list = utils.load_images('.', 'dataset', (INPUT_SIZE,INPUT_SIZE), query_labels)
query_images = defaultdict(list)


for img, y in zip(image_list, train_y_list):
    query_images[y].append(img)

 

def upload_image(arg):
    b64_img = ''
    f = request.files[arg].stream
    f.seek(0)
    img_bytes = f.read()
    nparr = np.fromstring(img_bytes, np.uint8)

    if len(nparr):
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR);
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC)
        image_reader = tf.convert_to_tensor(img)
        b, g, r = tf.unstack (image_reader, axis=-1)
        image_reader = tf.stack([r/255, g/255, b/255], axis=-1)
        _, buffer = cv2.imencode('.jpg', img)
        b64_img = base64.b64encode(buffer).decode('utf-8')

    return image_reader, b64_img

    

app = Flask(__name__, template_folder='.')

@app.get('/')
def index():
    return render_template('templates/index.html')

@app.get('/predict_pair')
def predict_pair_get():
    return render_template('templates/predict_pair.html')


@app.post('/predict_pair')
def predict_pair_post():
    cls = ''
    chance = ''
    
    image_reader, b64_img = upload_image('img')


    if b64_img:
        c = []
        for k, vs in query_images.items():
            pred_values = siamese_network_pairs.predict([np.array([image_reader]*len(vs)), np.array(vs)]).squeeze()

            avg = sum(pred_values) / len(pred_values)
            c.append((k, avg))


        cls, p = max(c, key=lambda x: x[1])
            

        chance = str(p * 100)[0:5]

    return render_template('templates/predict2_pair.html', img=b64_img, cls=query_labels[cls], chance=chance)



@app.get('/predict_triple')
def predict_triple_get():
    return render_template('templates/predict_triple.html')


@app.post('/predict_triple')
def predict_triple_post():
    cls = ''
    chance = ''
    
    image_reader, b64_img = upload_image('img')


    if b64_img:
        c = []
        for k, vs in query_images.items():
            c2 = []
            for k2, vs2 in query_images.items():
                if (k == k2):
                    continue
                
                loss_values = siamese_network_triples.predict([np.array([image_reader]*len(vs)), np.array(vs), np.array(vs2)]).squeeze()

                avg = sum(loss_values) / len(loss_values)
                c2.append(avg)

            avg = sum(c2) / len(c2)
            c.append((k, avg))

            


        cls, loss = min(c, key=lambda x: x[1])

        p = 100 - loss * 100 / TRIPLET_LOSS_MARGIN
            

        chance = str(p)[0:5]

    return render_template('templates/predict2_triple.html', img=b64_img, cls=query_labels[cls], chance=chance)





 
@app.get('/pair')
def pair_get():
    return render_template('templates/pair.html')



@app.post('/pair')
def pair_post():
    pred = ''
    chance = ''
    
    image_reader1, b64_img1 = upload_image('img1')
    image_reader2, b64_img2 = upload_image('img2')


    if b64_img1 and b64_img2:
        pred_value = siamese_network_pairs.predict([np.array([image_reader1]), np.array([image_reader2])]).squeeze()

        if pred_value > 0.5:
            pred = 'True'
            chance = str(pred_value * 100)[0:5]
        else:
            pred = 'False'
            chance = str(100 - pred_value * 100)[0:5]
    


    return render_template('templates/pair2.html', img1=b64_img1, img2=b64_img2, pred=pred, chance=chance)



@app.get('/triple')
def triple_get():
    return render_template('templates/triple.html')



@app.post('/triple')
def triple_post():
    pred = ''
    chance = ''
    
    image_reader1, b64_img1 = upload_image('img1')
    image_reader2, b64_img2 = upload_image('img2')
    image_reader3, b64_img3 = upload_image('img3')


    if b64_img1 and b64_img2:
        loss_value = siamese_network_triples.predict([np.array([image_reader1]), np.array([image_reader2]), np.array([image_reader3])]).squeeze()

        if loss_value <= TRIPLET_LOSS_MARGIN:
            pred = 'True'
        else:
            pred = 'False'
    


    return render_template('templates/triple2.html', img1=b64_img1, img2=b64_img2, img3=b64_img3, pred=pred)



if __name__=='__main__':
    app.run(debug = True)
