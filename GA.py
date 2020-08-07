#importing necessary packages
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import tflearn
import tensorflow as tf
from PIL import Image
%matplotlib inline
#for writing text files
import glob
import os     
import random 
#reading images from a text file
from tflearn.data_utils import image_preloader
import math
#-------------------------------------------------------------------------------------------------------------------------
def load_images():
    
    IMAGE_FOLDER = '/Users/tells/Documents/Apple/train'
    TRAIN_DATA = '/Users/tells/Documents/Apple/training_data.txt'
    TEST_DATA = '/Users/tells/Documents/Apple/test_data.txt'
    VALIDATION_DATA = '/Users/tells/Documents/Apple/validation_data.txt'



    #read the image directories
    filenames_image = os.listdir(IMAGE_FOLDER)
    #shuffling the data is important otherwise the model will be fed with a single class data for a long time and 
    #network will not learn properly
    random.shuffle(filenames_image)
    
    #total number of images
    total=len(filenames_image)
    ##  *****training data******** 
    fr = open(TRAIN_DATA, 'w')
    train_files=filenames_image[0: int(train_proportion*total)]
    for filename in train_files:
        if filename[0:9] == 'FREC_Scab':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 0\n')
        elif filename[0:5] == 'FREC_C_Rust':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 1\n')
        elif filename[0:7] == 'JR_FrgE':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 2\n')
        elif filename[0:5] == 'RS_HL':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 3\n')
    fr.close()

    ##  *****testing data******** 
    fr = open(TEST_DATA, 'w')
    test_files=filenames_image[int(math.ceil(train_proportion*total)):int(math.ceil((train_proportion+test_proportion)*total))]
    for filename in test_files:
        if filename[0:9] == 'FREC_Scab':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 0\n')
        elif filename[0:5] == 'FREC_C_Rust':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 1\n')
        elif filename[0:7] == 'JR_FrgE':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 2\n')
        elif filename[0:5] == 'RS_HL':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 3\n')
    fr.close()
    
    ##  *****validation data******** 
    fr = open(VALIDATION_DATA, 'w')
    valid_files=filenames_image[int(math.ceil((train_proportion+test_proportion)*total)):total]
    for filename in valid_files:
        if filename[0:9] == 'FREC_Scab':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 0\n')
        elif filename[0:5] == 'FREC_C_Rust':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 1\n')
        elif filename[0:7] == 'JR_FrgE':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 2\n')
        elif filename[0:5] == 'RS_HL':
            fr.write(IMAGE_FOLDER + '/'+ filename + ' 3\n')
    fr.close()

#-------------------------------------------------------------------------------------------------------------------------

def create_train_test(train_test_prop, detector_value, pixel, concolution_node):
    
    IMAGE_FOLDER = '/Users/tells/Documents/Apple/train'
    TRAIN_DATA = '/Users/tells/Documents/Apple/training_data.txt'
    TEST_DATA = '/Users/tells/Documents/Apple/test_data.txt'
    VALIDATION_DATA = '/Users/tells/Documents/Apple/validation_data.txt'
    train_proportion=1-train_test_prop/10.0
    test_proportion=train_test_prop/10.0
    
    #Importing data
    X_train, Y_train = image_preloader(TRAIN_DATA, image_shape=(pixel,pixel),mode='file', categorical_labels=True,normalize=True)
    X_test, Y_test = image_preloader(TEST_DATA, image_shape=(pixel,pixel),mode='file', categorical_labels=True,normalize=True)    
    
    
    #input image
    x=tf.placeholder(tf.float32,shape=[None,pixel,pixel,3] , name='input_image') 
    #input class
    y_=tf.placeholder(tf.float32,shape=[None, 4] , name='input_class')
    
    
    
    #Input Layer
    
    #reshaping input for convolutional operation in tensorflow
    # '-1' states that there is no fixed batch dimension, 28x28(=784) is reshaped from 784 pixels and '1' for a single
    #channel, i.e a gray scale image
    
    x_input=x
    #first convolutional layer with 32 output filters, filter size 5x5, stride of 2,same padding, and RELU activation.
    #I am not adding bias, but one could add bias.Optionally you can add max pooling layer as well 
    
    conv_layer1=tflearn.layers.conv.conv_2d(x_input, nb_filter=detector_value, filter_size=5, strides=[1,1,1,1],
                                            padding='same', activation='relu', regularizer="L2", name='conv_layer_1')
    
    #2x2 max pooling layer
    
    out_layer1=tflearn.layers.conv.max_pool_2d(conv_layer1, 2)
    
    #second convolutional layer 
    conv_layer2=tflearn.layers.conv.conv_2d(out_layer1, nb_filter=detector_value, filter_size=5, strides=[1,1,1,1],
                                            padding='same', activation='relu',  regularizer="L2", name='conv_layer_2')
    out_layer2=tflearn.layers.conv.max_pool_2d(conv_layer2, 2)
    #fully connected layer
    
    fcl= tflearn.layers.core.fully_connected(out_layer2, concolution_node, activation='relu')
    
    fcl_dropout = tflearn.layers.core.dropout(fcl, 0.8)
    y_predicted = tflearn.layers.core.fully_connected(fcl_dropout, 4, activation='softmax', name='output')
    
    
    
    #loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_predicted+np.exp(-10)), reduction_indices=[1]))
    #optimiser -
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #calculating accuracy of our model 
    correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    
    # session parameters
    sess = tf.InteractiveSession()
    #initialising variables
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    save_path="/Users/tells/Documents/data/mark3full.ckpt"


    # grabbing the default graph
    g = tf.get_default_graph()
    
    # every operations in our graph
    [op.name for op in g.get_operations()]
    
    
    
    epoch=10 # run for more iterations according your hardware's power
    #change batch size according to your hardware's power. For GPU's use batch size in powers of 2 like 2,4,8,16...
    batch_size=24 
    no_itr_per_epoch=len(X_train)//batch_size
    
    
    
    no_itr_per_epoch
    n_test=len(X_test) #number of test samples
    
    
    
    for iteration in range(epoch):
        #print("Iteration no: {} ".format(iteration))
        
        previous_batch=0
        # Do our mini batches:
        for i in range(no_itr_per_epoch):
            current_batch=previous_batch+batch_size
            x_input=X_train[previous_batch:current_batch]
            x_images=np.reshape(x_input,[batch_size,pixel,pixel,3])
            
            y_input=Y_train[previous_batch:current_batch]
            y_label=np.reshape(y_input,[batch_size,4])
            previous_batch=previous_batch+batch_size
            
            _,loss=sess.run([train_step, cross_entropy], feed_dict={x: x_images,y_: y_label})
            #if i % 100==0 :
                #print ("Training loss : {}" .format(loss))
                
       
            
        x_test_images=np.reshape(X_test[0:n_test],[n_test,pixel,pixel,3])
        y_test_labels=np.reshape(Y_test[0:n_test],[n_test,4])
        Accuracy_test=sess.run(accuracy,
                               feed_dict={
                            x: x_test_images ,
                            y_: y_test_labels
                          })
        Accuracy_test=round(Accuracy_test*100,3)
        
        #print("Accuracy ::  Test_set {} % " .format(Accuracy_test))
    return Accuracy_test

#-------------------------------------------------------------------------------------------------------------------------

def cal_pop_fitness(new_population):
     # Calculating the fitness value of each solution in the current population.
     # The fitness function calculates the sum of products between each input and its corresponding weight.
     #load_images()
     fitness = []
     for i in new_population:
         #print((i[0],i[1],i[2],i[3]))
         fitness.append(create_train_test(i[0],i[1],i[2],i[3]))
     return np.array(fitness)

#------------------------------------------------------------------------------------------------------------------------- 
'''
def select_mating_pool(pop, fitness, num_parents):
        parents[parent_num, :] = pop[max_fitness_idx, :]
'''
def select_mating_pool(pop, fitness, num_parents):

    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    #print(pop)
    parents = np.empty((num_parents, len(pop[0])))
    
    for parent_num in range(num_parents):

        max_fitness_idx = np.where(fitness == np.max(fitness))

        max_fitness_idx = max_fitness_idx[0][0]
        
        parents[parent_num, :] = pop[max_fitness_idx, :]
        #parents.append(pop[max_fitness_idx].copy())

        fitness[max_fitness_idx] = -99
    
    return parents 

#-------------------------------------------------------------------------------------------------------------------------
  
def crossover(parents, offspring_size):
     offspring = np.empty(offspring_size)
     # The point at which crossover takes place between two parents. Usually, it is at the center.
     crossover_point = np.uint8(offspring_size[1]/2)
 
     for k in range(offspring_size[0]):
         # Index of the first parent to mate.
         parent1_idx = k%parents.shape[0]
         # Index of the second parent to mate.
         parent2_idx = (k+1)%parents.shape[0]
         # The new offspring wi ll have its first half of its genes taken from the first parent.
         offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
         # The new offspring will have its second half of its genes taken from the second parent.
         offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
     return offspring
 
#-------------------------------------------------------------------------------------------------------------------------
def mutation(offspring_crossover):

    # Mutation changes a single gene in each offspring randomly.
    
    index = random.randint(1,3)

    for idx in range(offspring_crossover.shape[0]):

        # The random value to be added to the gene.

        random_value = random.randint(-1.0, 1.0)
                        
        offspring_crossover[idx, index] = offspring_crossover[idx, index]*(2**random_value)
        #It will multyply with 2 or devide by 2

    return offspring_crossover
#----------------------------------------------------------------------------------------------------------------------------    
