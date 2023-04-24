from flask import *
import numpy
from tensorflow.keras.preprocessing import image
from keras.models import load_model
model = load_model('model_v1_inceptionV3.h5')

classes = {
    0:'Burger is a fast food item that consists of a patty which is squished between a pair of buns. The patty can be either vegetarian or non-vegetarian. Veg patties are usually made of potatoes and/or indian cottage cheese (paneer) and non-veg patties consists of either chicken, mutton or beef, especially in the western demography. Additionally, there are some toppings that are added on top of the patty, like cheese, vegetables like onions and tomatoes, lettuce and some kind of sauce, be it ketchup and mayo, or with some restaurants a "special" sauce. Burgers are extremely common in various fast food places as it is very affordable and quite filling as well.',
    2:'Chai translates to milk tea. It is the indian version of tea that is the most common and staple version that is available. It is usually much sweeter and encompasses much more flavour than the more traditional tea. Chai is the most popular beverage of choice in many families as their morning drink and tea stalls across the country see a vast amount of customers who visit just for a cup of chai to start their morning',
    13:"Masala Dosa is originated from the sourthern part of india, and is a staple breakfast item in states such as Kerala and Tamil Nadu. Masala Dosa consists of a fermented rice batter, which is spread on a hot 'tawa' and is filled with some kind of 'masala' or filling. The 'masala' that is used to fill the dosa is usually made of spiced and mashed potato and is served with a combination of coconut chutney and sambhar.",
}

def classify(file_path):
    global label_packed
    # image = Image.open(file_path)
    # image = image.resize((299,299))
    # image = numpy.expand_dims(image, axis=0)
    # image = numpy.array(image)
    # pred = model.predict_classes([image])[0]
    # pred = numpy.argmax(model.predict([image])[0], axis=-1)
    # img_ = image.load_img(file_path, target_size=(299, 299))
    img_array = image.img_to_array(file_path)
    img_processed = numpy.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)

    index = numpy.argmax(prediction)
    sign = classes[index]
    return sign


def load_image(image):
    text = classify(image)
    return text

app = Flask(__name__)


@app.route('/')
def index():
    return "welcome to app"

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['img']
        # file_path = secure_filename(f.filename)
        f.save("img.jpg")
        # Make prediction
        result = load_image("img.jpg")
        # result = result.title()
        # d = {"Ice Cream":"🍨",'Fried Rice':"🍚","Pizza":"🍕","Sandwich":"🥪","Samosa":"🌭"}
        # result = result+d[result]
        print(result)
        # os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run()