from UNet.model import *
from UNet.data import *
import datetime
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def perform_unet_detection():
    model = load_model('UNet/unet_vines.hdf5')
    testGene = test_generator('UNet/data/vines/test')
    t1 = datetime.datetime.now()
    results = model.predict_generator(testGene,20,verbose=0)
    t2 = datetime.datetime.now()
    time = t2-t1
    print("Unet detection time, seconds: ", time.seconds, " microseconds: ", time.microseconds)
    save_result('UNet/data/vines/test',results)
    return time
