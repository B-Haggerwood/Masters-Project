# This code is used to import the weights of the trained model and initiate
# the training process

import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt

from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# %% Loading in our PNG data:
BUFFER_SIZE = 192  # 400
BATCH_SIZE = 1
IMG_WIDTH =  256
IMG_HEIGHT =  256


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image, channels=0, dtype=tf.uint16)

    w = tf.shape(image)[1]
    w = w // 2
    # real_image = image[:w, w:2*w] # Selecting the 64 grid part
    # input_image = image[:w, :w] # Selecting the 32 grid part
    real_image = image[w:2*w, w:2*w] # Selecting the 256 grid part
    input_image = image[w:2*w, :w] # Selecting the 128 grid part

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

inp, re = load('../Four_Grids_192_PNG/train_set_0.png')

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

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

# %% Build the Generator

OUTPUT_CHANNELS = 1 # 3 Change this to 1?

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02) # 0.02

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

down_model = downsample(1, 4) 
down_result = down_model(tf.expand_dims(inp, 0)) # This line adds a batch dimension at the start

print('Shape of down_result = '+str(down_result.shape))

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02) # 0.02)

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

#%% Discriminator loss

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

# %% Optimizers and saving checkpoints:
    
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.6) 
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.6) 

checkpoint_dir = './training_checkpoints_trial'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

# %% Restore last checkpoint and test:
    
# Test Dataset
test_dataset = tf.data.Dataset.list_files('../Test_Set_10_PNG/test_set_*.png')
test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

#%%

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)) 

# Creating function to test reloaded gen
def generate_images_reload(model, test_input, tar, num):
  prediction = model(test_input, training=False)
  np.save('./Test_Comparison_New/Test_Prediction_%d' % (num), prediction.numpy())
  np.save('./Test_Comparison_New/Test_Truth_%d' % (num), tar.numpy())
  np.save('./Test_Comparison_New/Test_Input_%d' % (num), test_input.numpy())
  
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  Path3 = './Test_Image_%d.png' % (num)
  plt.savefig(Path3, dpi=1000)
  plt.show()
  
# Plot training image to check that reloading has been done correctly
num = 0
for example_input, example_target in test_dataset:
    generate_images_reload(generator, example_input, example_target, num)

    num += 1
    
#%%
session.close()
