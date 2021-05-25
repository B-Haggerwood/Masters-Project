# This code is used to train our models over a defined number of epochs
# for a specific input and image resolution.

import tensorflow as tf
from tensorflow import keras
import numpy as np

import os
import time

from matplotlib import pyplot as plt
from IPython import display


# Trying to fix issue (seems to work?)
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# %% Loading in our PNG data:


BUFFER_SIZE = 192  
BATCH_SIZE = 1
IMG_WIDTH =  256
IMG_HEIGHT =  256


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image, channels=0, dtype=tf.uint16)

    w = tf.shape(image)[1]
    w = w // 2
    # input_image = image[:w, w:2*w] # Selecting the 64 grid part
    # input_image = image[:w, :w] # Selecting the 32 grid part
    real_image = image[w:2*w, w:2*w] # Selecting the 256 grid part
    input_image = image[w:2*w, :w] # Selecting the 128 grid part

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

inp, re = load('../Four_Grids_192_PNG/train_set_0.png')
# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(inp/255.0)
plt.figure()
plt.imshow(re/255.0)

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image


def normalize(input_image, real_image):
  
  input_image = 2*((input_image / 65535)**0.5)-1
  real_image = 2*((real_image / 65535)**0.5)-1

  return input_image, real_image


def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image



#######
# %% Input Pipeline for our data (Attempt 2)

train_dataset = tf.data.Dataset.list_files('../Four_Grids_192_PNG/train_set_*.png')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE) #num_parallel_calls=None)#,
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# %% Build the Generator

OUTPUT_CHANNELS = 1 

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02) 

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

down_model = downsample(1, 4) 
down_result = down_model(tf.expand_dims(inp, 0)) 

print('Shape of down_result = '+str(down_result.shape))

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

up_model = upsample(1, 4) 
up_result = up_model(down_result)
print('Shape of up_result = '+str(up_result.shape))

def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 1])    

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  
    downsample(128, 4),  
    downsample(256, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4, apply_dropout=True),  
    upsample(256, 4),  
    upsample(128, 4),  
    upsample(64, 4),  
    upsample(64, 4),  
  ]

  initializer = tf.random_normal_initializer(0., 0.02) 
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                          strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          activation='tanh')  
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)
  
  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

# %% Generator loss

LAMBDA = 100 

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

# %% Build the Discriminator:
    
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image') 
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image') 

  x = tf.keras.layers.concatenate([inp, tar])  

  down1 = downsample(64, 4, False)(x)  
  down2 = downsample(128, 4)(down1)  
  down3 = downsample(256, 4)(down2)  

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) 
  conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, 
                                use_bias=False)(zero_pad1)  

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  

  last = tf.keras.layers.Conv2D(1, 5, strides=1,
                                kernel_initializer=initializer,
                                activation='tanh')(zero_pad2)  

  return tf.keras.Model(inputs=[inp, tar], outputs=last) 

discriminator = Discriminator()

# %% Discriminator loss

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

# %% Optimizers and saving checkpoints:
    
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5) 
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5) 

checkpoint_dir = './training_checkpoints_trial'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

# %% Plotting images during training:
# May need to comment this out for now as we have no test dataset, only training

def generate_images(model, train_input, tar, epoch):
   prediction = model(train_input, training=True)
   plt.figure(figsize=(15, 15))

   display_list = [train_input[0], tar[0], prediction[0]]
   title = ['Input Image', 'Ground Truth', 'Predicted Image']

   for i in range(3):
     plt.subplot(1, 3, i+1)
     plt.title(title[i])
     # getting the pixel values between [0, 1] to plot it.
     plt.imshow(display_list[i] * 0.5 + 0.5)
     plt.axis('off')
   plt.show()
   Path3 = 'Epoch_%d.png' % (epoch + 1)
   plt.savefig(Path3, dpi=1000)
  
#%% Early Stopping

def early_stopping(l1_loss_memory):
    
    first_five = (l1_loss_memory[0] + l1_loss_memory[1] + l1_loss_memory[2] +
        l1_loss_memory[3] + l1_loss_memory[4]) / 5
    last_five = (l1_loss_memory[49] + l1_loss_memory[48] + l1_loss_memory[47] +
        l1_loss_memory[46] + l1_loss_memory[45]) / 5
    
    diff = last_five - first_five
    
    if diff < 0:
        stop = True
    else:
        stop = False
    
    return stop

# %% Training

EPOCHS = 1000

@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    
  return gen_l1_loss, gen_gan_loss, disc_loss


def fit(train_ds, epochs):
  L1_loss_array = []
  gen_gan_loss_array = []
  disc_loss_array = []
  
  l1_loss_memory = np.linspace(0,49,50)
  
  for epoch in range(epochs):
    start = time.time()
    
    mean_L1_loss = []
    mean_disc_loss = []
    mean_gen_gan_loss = []

    display.clear_output(wait=True)

    if((epoch+1) % 50 == 0 or (epoch+1) == 1):
        for example_input, example_target in train_ds.take(1):
            generate_images(generator, example_input, example_target, epoch)
            
    
    print("Epoch: ", epoch)

    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      L1_loss_single, gen_gan_loss_single, disc_loss_single = train_step(input_image, target, epoch)
      
      mean_L1_loss.append(L1_loss_single)
      mean_gen_gan_loss.append(gen_gan_loss_single)
      mean_disc_loss.append(disc_loss_single)
      
    mean_L1_loss = np.mean(mean_L1_loss)
    mean_gen_gan_loss = np.mean(mean_gen_gan_loss)
    mean_disc_loss = np.mean(mean_disc_loss)
    L1_loss_array.append(mean_L1_loss)
    gen_gan_loss_array.append(mean_gen_gan_loss)
    disc_loss_array.append(mean_disc_loss)
    
    if early_stopping(l1_loss_memory) == True:
      break
    else:
      pass
  
    for i in range(49,-1,-1):
        if i == 0:
            l1_loss_memory[i] = mean_L1_loss
        else:
            l1_loss_memory[i] = l1_loss_memory[i-1]
    
    print()
    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  
  checkpoint.save(file_prefix=checkpoint_prefix)
  for example_input, example_target in train_ds.take(1):
            generate_images(generator, example_input, example_target, epoch)
    
  return L1_loss_array, gen_gan_loss_array, disc_loss_array

# %% Running the training loop:
    
L1_loss_array, gen_gan_loss_array, disc_loss_array = fit(train_dataset, EPOCHS)
np.save('./L1_loss_array', L1_loss_array)
np.save('./gen_gan_loss_array', gen_gan_loss_array)
np.save('./disc_loss_array', disc_loss_array)

#%% Plotting Losses

x = np.linspace(1, EPOCHS, EPOCHS)

fig1 = plt.figure()
ax1 = fig1.add_subplot(311)

plt.title('lambda = 100, Initializer = 0.02')

ax1.plot(x, disc_loss_array)
ax1.axes.xaxis.set_ticklabels([])
plt.ylabel('disc_loss')

ax2 = fig1.add_subplot(312)
ax2.plot(x, L1_loss_array)
ax2.axes.xaxis.set_ticklabels([])
plt.ylabel('L1_loss')
plt.axis([None, None, 0, 0.1])

ax3 = fig1.add_subplot(313)
ax3.plot(x, gen_gan_loss_array)
plt.xlabel('Epoch')
plt.ylabel('gen_loss')

print(np.min(L1_loss_array))

plt.tight_layout()
plt.savefig("final_losses_L1.png", dpi=1000, pad_inches=0.5)

#%%    
session.close()

