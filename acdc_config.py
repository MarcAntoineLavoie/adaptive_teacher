import sys
import yaml

# dataset = 'rain'
dataset = sys.argv[1]

file_train = '("ACDC_train_{}",)'.format(dataset)
file_test = '("ACDC_val_{}",)'.format(dataset)

with open("adaptive_teacher/configs/faster_rcnn_VGG_cross_city_prob.yaml") as f:
    y = yaml.safe_load(f)
    y['DATASETS']['TRAIN_UNLABEL'] = file_train
    y['DATASETS']['TEST'] = file_test
    print

with open("adaptive_teacher/configs/faster_rcnn_VGG_cross_city_acdc.yaml", 'w') as f:
    yaml.dump(y, f, default_flow_style=False, sort_keys=False)

