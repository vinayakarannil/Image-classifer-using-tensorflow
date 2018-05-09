# Sushi_Sandwich_Classifier

This project is an image classifier. Given a photo, the model will classify it into one of two categories: sushi or sandwich. .

## Getting Started

Here i have used tensorflow-slim library to transfer learn a Resnet-V1 model for our custom dataset of sushi/sandwich images. 

### Prerequisites

Please make sure you have installed all the dependencies. You can install the dependencies using theses commands.(If you have already installed some of these, you dont need to install again.Make sure to omit already installed libraries from the commands while running)
```
sudo apt-get install python-pip python-dev python-tk python-opencv

sudo pip install --upgrade pip tensorflow matplotlib 

```
### Running

You can either run the Dockerfile which will install all the dependencies and download the dataset. If you are doing everything manually. Please follow the below steps.
```

1. Download the dataset, extract and keep the dataset of images inside /dataset_casestudy folder. Rename the extracted folder to "data" The folder structure should be as below.

  ---dataset_casestudy
     ----data
          ------sushi
          ------sandwich
          
   url of dataset : http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip
   
2. Download pre-trained Resnet-V1 model from the below url, extract and keep the model in /ckpt directory(create ckpt directory if not available)

    http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
3. Run the download_and_convert_data.py script which will convert the datset into tfrecord format. After running the script, you will see tfrecord file created inside dataset_casestudy directory and a file label.txt containing label mappings.
4. Run the training script train_image_classifier.py. Keep monitoring the /Result folder. 
5. After graphs get saved in Result folder you can run the eval_image_classifier.py to analyse the accuracy and recall.

```


## Model Analysis

The model was trained on a 12GB titanX GPU for almost 15K epoch. Here are the evaluation metrics

Training Accuracy(F1 score): 99%
Training Recall : 99%
Training Precision:99%


Validation Accuracy(F1 score):85%
Validation Recall: 85%
Validation Precision:85%


## How to use the model

I have added scripts to use the model for other images. You may please run test_model.py by editing image path for any test image.
This script will inturn call predict method of predict_img.py which loads the saved model from /Result directory and get the prediction.




