
from predict_img import predict

test_image = "provide the path of the image"

label_file = open('dataset_casestudy/labels.txt')
labels = {}
for line in label_file:
	key = line.split(':')[0]
        value = line.split(':')[1]
        labels[key] = value.replace("\n", "")


index = predict(image_path)
label = labels[str(index)]
print("predicted label is "+label)

    
