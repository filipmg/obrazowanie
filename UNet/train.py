from model import *
from data import *
from keras.models import load_model


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = train_generator(2,'data/vines/train','image','label',data_gen_args,save_to_dir = None)

model = load_model('unet_vines.hdf5')
model_checkpoint = ModelCheckpoint('unet_vines.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=5,callbacks=[model_checkpoint])

testGene = test_generator("data/vines/test")
results = model.predict_generator(testGene,20,verbose=1)
save_result("data/vines/test",results)
