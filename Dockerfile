FROM ubuntu:latest

#install dependencies
RUN apt-get update
RUN apt-get install -y python python-pip wget
RUN pip install --upgrade pip tensorflow matplotlib

#Download, extract and place the data inside dataset directory
RUN mkdir dataset_casestudy && cd dataset_casestudy && wget http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip
RUN unzip sushi_or_sandwich_photos.zip && mv sushi_or_sandwich data

#Download pre-trained Resnet-V1 model from the below url, extract and keep the model in /ckpt directory
RUN cd ..
RUN mkdir ckpt && wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
RUN tar xf resnet_v1_152_2016_08_28.tar.gz && mv resnet_v1_152.ckpt ckpt/


#Run the download_and_convert_data.py script which will convert the datset into tfrecord format. After running the script, you will see #tfrecord file created inside dataset_casestudy directory and a file label.txt containing label mappings.
RUN python download_and_convert_data.py

#Run the training script train_image_classifier.py. Keep monitoring the /Result folder. 
RUN python train_image_classifier.py

#After graphs get saved in Result folder you can run the eval_image_classifier.py to analyse the accuracy and recall.
#RUN python eval_image_classifier.py



