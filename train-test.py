
def getArg():
  import argparse, os
  
  # Parse arguments
  parser = argparse.ArgumentParser(
    description='Tensorflow 2.x Resnet Training')
    
  # Tensorflow verbose info
  parser.add_argument(
    '--enable-tenorflow-verbose',
    default=False,
    action='store_true')
    
  # pre-trained model
  parser.add_argument(
    '--use-whole-network-model',
    default=False,
    action='store_true')
    
  parser.add_argument(
    '--pre-train-model-path-dir',
    type=str,
    default="")
    
  # test image
  parser.add_argument(
    '--test-image',
    type=str,
    default=os.path.join(
      'dataset',
      'flower_photos',
      '592px-Red_sunflower.jpg')
  )
  
  # Datasets
  parser.add_argument(
    '--train-dataset-path',
    default='path to train dataset',
    type=str,
    required=True)
    
  parser.add_argument(
    '--test-dataset-path',
    default='path to test dataset',
    type=str,
    required=True)
    
  parser.add_argument(
    '--resnet-version',
    default=1,
    type=int,
    metavar='N',
    help='resnet version(default: 1, can be set 1 and 2)')
    
      
  parser.add_argument(
    '--resnet-layer-number',
    default=50,
    type=int,
    metavar='N',
    help='resnet layer number(default: 50, can be set 18/50/101/152)')
    
    
  parser.add_argument(
    '--classes',
    default=5,
    type=int,
    metavar='N',
    help='images classes(default: 5, flower dataset)')
    
  parser.add_argument(
    '--image-height',
    default=180,
    type=int,
    metavar='N',
    help='image height(default: 180, flower dataset)')
    
  parser.add_argument(
    '--image-width',
    default=180,
    type=int,
    metavar='N',
    help='image width(default: 180, flower dataset)')
  
  # Optimization options
  parser.add_argument(
    '--opt-type',
    default='SGD',
    type=str,
    help='Optimizator type(default: RMSprop)[options: Adam, SGD, Adadelta, Adamax, Ftrl, Nadam, RMSprop]')
    
  # for SGD
  parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    help='initial SGD momentum(default: 0.9)')
    
  parser.add_argument(
    '--lr',
    default=0.001,
    type=float,
    metavar='LR',
    help='initial learning rate(default: 0.001)')
  
  parser.add_argument(
    '--epochs',
    default=20,
    type=int,
    metavar='N',
    help='number of total epochs to run(default: 20)')
  
  parser.add_argument(
    '--batchsize',
    default=8,
    type=int,
    metavar='N',
    help='batchsize (default: 8)')
    
  #Device options
  parser.add_argument(
    '--gpu-id',
    default='0',
    type=int,
    help='id(s) for CUDA_VISIBLE_DEVICES')
  
  args = parser.parse_args()
  
  if not args.enable_tenorflow_verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  
  import tensorflow as tf
  gpu_available_num = len(tf.config.experimental.list_physical_devices('GPU'))
  print("Num GPUs Available: ",gpu_available_num)
  if gpu_available_num > args.gpu_id:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
  else:
    print("No GPU Available! Would use CPU.")
    
  print("Tensorflow Version: ", tf.__version__)
  
  return args
  
if __name__ == '__main__':
  import os
  
  args = getArg()
  print(args)
  
  from model import ResNetWork
  resNetWork = ResNetWork(args)
  resNetWork.train()
  resNetWork.test(args)
  
  resNetWork.saveModel('resnet.h5')
  resNetWork.saveModel('resnet-whole', mode='whole_network')
  
  
  
  