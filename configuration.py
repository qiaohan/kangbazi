import os

data_folder = 'C:\\Juyue_Personal\\Avito_data'
train_csv_file = os.path.join(data_folder, 'train.csv','train.csv')
image_folder = os.path.join(data_folder, 'train_jpg_0')

## feature extraction
batch_size = 10

## different model has different model size. configure this later on.
model = 'xception'
img_size = 299

feature_folder = os.path.join(data_folder, 'img_feature', model)
log_folder = os.path.join(data_folder, 'img_feature_log', model)

# model = 'inception_v3'
# model = 'VGG19'
# model = 'Resnet50'
# img_size = 244