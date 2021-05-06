


import sys, os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dropout, concatenate, BatchNormalization,Activation,
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv3DTranspose
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras import optimizers
import keras.backend as K
from keras.losses import binary_crossentropy as bce
from keras.utils import multi_gpu_model



vgg_basepath = '/array/ssd/umapathy/Code/OASIS3/Pretrained/'

if vgg_basepath not in sys.path:
    sys.path.insert(0,vgg_basepath)


def setTF_environment(visible_gpu='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    K.set_session(tf.Session(config=config))
    return

def unet_block_contract(block_input,numFt1,numFt2,conv_kernel,upsample_flag=True):
    # Defining a UNET block in the feature downsampling path
    if(upsample_flag):
        block_input = MaxPooling3D(pool_size=(2,2,1))(block_input)
        
    down = Conv3D(numFt1,conv_kernel,padding='same')(block_input)
    down = BatchNormalization()(down)
    down = Activation('relu')(down)
    down = Conv3D(numFt2,conv_kernel,padding='same')(down)
    down = BatchNormalization()(down)
    down = Activation('relu')(down)
    return down
 
def unet_block_expand(block_input,numFts,concat_block,conv_kernel):
    # Defining a UNET block in the feature upsampling path
    up = Conv3DTranspose(numFts,(3,3,3), strides=(2,2,1), padding='same')(block_input)
    up = concatenate([up, concat_block], axis=4)
    up = Conv3D(numFts,conv_kernel,padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    up = Conv3D(numFts,conv_kernel,padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    return up

def UNET3D_Segmentation(ipShape,blkSize=5,nChannels=1,num_classes=1,conv_kernel=(3,3,3),final_Activation='softmax'):  
    inputs = Input((ipShape[0],ipShape[1],blkSize,nChannels))
    # Contracting path
    down_blk1 = unet_block_contract(inputs,16,32,conv_kernel,upsample_flag=False)
    down_blk2 = unet_block_contract(down_blk1,32,64,conv_kernel)
    center_blk4 = unet_block_contract(down_blk2,64,128,conv_kernel)
    center_blk4 = Dropout(0.4)(center_blk4)
    # Expanding path
    up_blk3 = unet_block_expand(center_blk4,64,down_blk2,conv_kernel)
    up_blk2 = unet_block_expand(up_blk3,32,down_blk1,conv_kernel)
    # Thalamus prediction
    thalamus_output = Conv3D(1, (1,1,1), activation='sigmoid', name='thalamus_label')(up_blk2)
    blk4 = concatenate([up_blk2, thalamus_output], axis=-1)
    # Nuclei prediction
    nuclei_output = Conv3D(num_classes, (1,1,1), activation=final_Activation, name='nuclei_label')(blk4) 
    model = Model(inputs=[inputs], outputs=[nuclei_output,thalamus_output], name='thalamic_nuclei_segmentation_model')
    return model

  
# Base Model for MPRAGE to WMn-MPRAGE Contrast synthesis  
def UNET3D_ContrastSynthesis(ipShape,name):
    input_orig = Input(ipShape)
    conv_kernel = (3,3,3)
    # Contracting path
    down_blk1 = unet_block_contract(input_orig,16,32,conv_kernel,upsample_flag=False)
    down_blk2 = unet_block_contract(down_blk1,32,64,conv_kernel)
    center_blk4 = unet_block_contract(down_blk2,64,128,conv_kernel)
    center_blk4 = Dropout(0.4)(center_blk4)
    # Expanding path
    up_blk3 = unet_block_expand(center_blk4,64,down_blk2,conv_kernel)
    up_blk2 = unet_block_expand(up_blk3,32,down_blk1,conv_kernel)
    synthesized_output = Conv3D(1, (1,1,1) , padding='same', name='contrast_synthesis_model')(up_blk2)
    model = Model(inputs=input_orig, outputs=synthesized_output,name=name)
    return model 

# Base Model for MPRAGE to WMn-MPRAGE Contrast Synthesis with perceptual loss
def UNET3D_ContrastSynthesis_perceptual(ipShape,optimizer,visible_gpu):
    from vgg16_modified import VGG16
    vgg16_weights = vgg_basepath + 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg16 = VGG16(include_top=False, weights='imagenet', weights_path=vgg16_weights,
              input_tensor=None)
    xDim,yDim,zDim,cDim = ipShape
 
    # instantiate feature Extractor model
    CS_model = UNET3D_ContrastSynthesis(ipShape,name='CS_model')
    
    # Create a new VGG model to compare feature map output from relu3_3
    vgg16_model = Model(inputs=vgg16.inputs, 
                    outputs=[vgg16.get_layer('relu3_3').output])
    image_style = Input(shape=(xDim,yDim,3))
    output_style = vgg16_model(image_style)
    model_vgg = Model(inputs=image_style, outputs=output_style, name= 'vgg_style_transfer')
    # Set trainable to false to avoid updating weights during the training process
    model_vgg.trainable = False 
    for idx in range(len(model_vgg.layers[1].layers)):
        model_vgg.layers[1].layers[idx].trainable = False
        
    csfn_ip = Input(shape=ipShape, name='csfn_input')
    wmn_op = CS_model(csfn_ip)
    final_model = Model(inputs=csfn_ip, outputs=wmn_op,name='csfn_to_wmn')
    if len(visible_gpu)>1: 
        final_model = multi_gpu_model(final_model,gpus=len(visible_gpu))
    final_model.compile(optimizer = optimizer, loss =custom_perceptualLoss(model_vgg) )
    return final_model 

# Custom reconstruction loss for contrast synthesis
def custom_perceptualLoss(model_architecture):
    def Recon_perceptualLoss(y_true, y_pred):   
        # Extract a 5D tensor to 4D for VGG compatibility - extract center slice
        # The input blocks are NxHxWxSxC
        temp_y_true = y_true[:,:,:,2,:]
        temp_y_pred = y_pred[:,:,:,2,:]

        # RGB channels for vgg compatibility
        y_true_vgg = concatenate([temp_y_true,temp_y_true,temp_y_true],axis=-1)      
        y_pred_vgg = concatenate([temp_y_pred,temp_y_pred,temp_y_pred],axis=-1)
        
        # Get the feature maps from VGG16
        style_fts_true = model_architecture(y_true_vgg)     
        style_fts_pred = model_architecture(y_pred_vgg)

        _,H,W,CH = style_fts_true.get_shape().as_list()
        # Feature Normalization
        if H is None or W is None or CH is None:
            scalar_mult = 1.0
        else:
            scalar_mult = 1/ ( H * W * CH)
        style_loss = scalar_mult * tf.reduce_mean(tf.squared_difference(style_fts_true, style_fts_pred))
        perceptualLoss_weight = 0.1  # scalar factor for importance of mae vs perceptual loss
        mae_loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return mae_loss + (perceptualLoss_weight * style_loss) 
    return Recon_perceptualLoss  

def initializeSegmentationCNN(model_architecture, params, pretrained_weights = None):
    ipShape = params['ipShape']
    num_classes = params['num_classes']  
    model = model_architecture(ipShape[0],ipShape[1],ipShape[2],nChannels=ipShape[3],num_classes=num_classes,conv_kernel=params['conv_kernel'],final_Activation='sigmoid')
    if len(params['visible_gpu'])>1: 
        model = multi_gpu_model(model,gpus=len(params['visible_gpu']))
    Adamopt = params['optimizer']
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    model.compile(loss={'nuclei_label':segmentation_loss(num_classes), 'thalamus_label': segmentation_loss(1)}, 
                  optimizer=Adamopt,
                  loss_weights={'nuclei_label':1., 'thalamus_label':0.5})
    return model

def segmentation_loss(numClasses):
    # Multi-class loss based on numClasses
    def loss(y_true, y_pred):
        loss = [dice_bce_coeff(y_true[...,d],y_pred[...,d]) for d in range(numClasses)]
        return K.sum(loss)/numClasses
    return loss

def dice_bce_coeff(y_true, y_pred):
    # Compute Dice loss
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred)
    intersection_fg = K.sum(y_true_f * y_pred_f) +  K.epsilon()
    union_fg = K.sum(y_true_f) + K.sum(y_pred_f)
    coeff = (2 * intersection_fg) / (union_fg +  K.epsilon())
    dice_loss = 1 - coeff
    bce_loss = bce(y_true_f,y_pred_f)
    return dice_loss + bce_loss

