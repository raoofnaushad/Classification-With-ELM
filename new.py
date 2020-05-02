import keras
from keras.preprocessing.image  import ImageDataGenerator
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import os
from keras.utils.np_utils import to_categorical
from elm import ELM, load_model


def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        # if ftype == ".png":
        img = read(PATH)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
           
        img = cv2.resize(img, (RESIZE,RESIZE))
           
        IMG.append(np.array(img))
    return IMG

benign_train = np.array(Dataset_loader('data/train/1',28))
malign_train = np.array(Dataset_loader('data/train/1',28))
benign_test = np.array(Dataset_loader('data/test/1',28))
malign_test = np.array(Dataset_loader('data/test/2',28))


benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

X_train = np.concatenate((benign_train, malign_train), axis = 0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis = 0)
X_test = np.concatenate((benign_test, malign_test), axis = 0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis = 0)


s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]



x_train = X_train.astype(np.float32) / 255.
x_train = X_train.reshape(-1, 28**2)
x_test = X_test.astype(np.float32) / 255.
x_test = x_test.reshape(-1, 28**2)
t_train = to_categorical(Y_train, 2).astype(np.float32)
t_test = to_categorical(Y_test, 2).astype(np.float32)




def softmax(x):
    c = np.max(x, axis=-1)
    upper = np.exp(x - c)
    lower = np.sum(upper, axis=-1)
    return upper / lower

model = ELM(
    n_input_nodes=28**2,
    n_hidden_nodes=1024,
    n_output_nodes=2,
    loss='mean_squared_error',
    activation='sigmoid',
    name='elm',
)

# ===============================
# Training
# ===============================
model.fit(x_train, t_train)
train_loss, train_acc, train_uar = model.evaluate(x_train, t_train, metrics=['loss', 'accuracy', 'uar'])
# print('train_loss: %f' % train_loss) # loss value
# print('train_acc: %f' % train_acc) # accuracy
# print('train_uar: %f' % train_uar) # uar (unweighted average recall)

# ===============================
# Validation
# ===============================


val_loss, val_acc, val_uar = model.evaluate(x_test, t_test, metrics=['loss', 'accuracy', 'uar'])
# print('val_loss: %f' % val_loss)
# print('val_acc: %f' % val_acc)
# print('val_uar: %f' % val_uar)

# ===============================
# Prediction
# ===============================
x = x_test[1]

y = softmax(model.predict(x))

print(y)
# ===============================
# Save model
# ===============================
print('saving model...')
model.save('model.h5')
del model

# ===============================
# Load model
# ===============================
print('loading model...')
model = load_model('model.h5')

