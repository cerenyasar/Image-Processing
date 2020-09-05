import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf

from time import time
start_time = time()


"""import time
start_time = time.time()"""



def createTrain(folder):
    train=[]
    number_of = len(os.listdir(folder)) #number_of sinif sayisidir.
    index=0
    for c_name in tqdm(os.listdir(folder)): # c_name sinif adlarini ifade eder.
        path = os.path.join(folder,c_name)
        label=np.zeros((number_of - 1,), dtype=np.int) #bir sinifi tanimlayan label array olusturuluyor.
        label=np.insert(label, index, 1) # Sinifa uygun yere 0 lardan olusan label arrayine 1 sayisi ekleniyor.
        index += 1
        for image in tqdm(os.listdir(path)):
            in_path=os.path.join(path,image)
            img=cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img, (boyut, boyut))
            train.append([np.array(img), label, image, c_name])
    return train,number_of

#hangi dosyayi okuyalim? Buradan ayarliyoruz.
ghim='GHIM20'
tumor_teshis='tumorTeshis'
boyut=50
train,number_of=createTrain(tumor_teshis)

shuffle(train)
print(train)

test_img=train[:int(len(train)/100)]
test_image_number=int((len(train)-int(len(train)/100))/5)
trains = train[int(len(train)/100):-1*test_image_number]
tests = train[-1*test_image_number:]

train_matrix=np.array([i[0] for i in trains]).reshape(-1, boyut, boyut, 1)
train_target=[i[1] for i in trains]
test_matrix=np.array([i[0] for i in tests]).reshape(-1, boyut, boyut, 1)
test_target=[i[1] for i in tests]


tf.reset_default_graph()

net = input_data(shape=[None, boyut, boyut, 1], name='input')

net = conv_2d(net, 512, 5, activation='relu')
net = max_pool_2d(net, 5)

net = conv_2d(net, 256, 5, activation='relu')
net = max_pool_2d(net, 5)

net = conv_2d(net, 192, 5, activation='relu')
net = max_pool_2d(net, 5)

net = conv_2d(net, 128, 5, activation='relu')
net = max_pool_2d(net, 5)

net = conv_2d(net, 64, 5, activation='relu')
net = max_pool_2d(net, 5)

net = conv_2d(net, 32, 5, activation='relu')
net = max_pool_2d(net, 5)

net = fully_connected(net, 1024, activation='relu')

net = dropout(net, 0.8)

net = fully_connected(net, number_of, activation='softmax')


net = regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                     name='targets')
#learning rate ne kadar hızlı öğreneceğimizi anlamaya yarayan bir sistem.

model = tflearn.DNN(net, tensorboard_dir='log', tensorboard_verbose=0)

model.fit({'input': train_matrix}, {'targets': train_target}, n_epoch=10,validation_set=({'input': test_matrix}, {'targets': test_target}),snapshot_step=500, show_metric=True, run_id="Resim tanima")

#saver = tf.train.Saver()
#saver.save(, 'my_test_model',global_step=1000)

success=0
for i in test_img:
    mat=i[0].reshape(boyut, boyut, 1)
    output=model.predict([mat])

    #output labelini bulma
    k=0
    while (train[k][1][np.argmax(output[0])]!=1):
        k+=1
    label=train[k][3]


    print("bulunan label: "+ label)
    print("gercek label: "+ i[3])
    print("out:"+str(output))
    print("gercek:"+str(i[1]))
    print("ad:" + i[2])
    print("---------------------------")
    if i[1][np.argmax(output[0])]==1:
        success+=1
print("Basari: %" + str((success/len(test_img))*100))
#print("--- %s seconds ---" % (time.time() - start_time))
end_time = time()
time_taken = end_time - start_time # time_taken is in seconds
hours, rest = divmod(time_taken,3600)
minutes, seconds = divmod(rest, 60)
print ( str(hours)+" hour "+str(minutes)+ " minute " +str(seconds)+" seconds ")