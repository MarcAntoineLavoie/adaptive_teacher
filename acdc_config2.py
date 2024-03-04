import sys
import yaml

dataset_source = sys.argv[1]
dataset_target = sys.argv[2]
if len(sys.argv) > 3:
    run_type = sys.argv[3]
else:
    run_type = 'faster_rcnn_VGG_cross_city_prob.yaml'

config_file = "configs/" + run_type
output_file = config_file.rsplit('/',1)[0] + "/faster_rcnn_VGG_cross_city_acdc.yaml"

file_source = '("ACDC_train_{}",)'.format(dataset_source)
file_train = '("ACDC_train_{}",)'.format(dataset_target)
file_test = '("ACDC_val_{}",)'.format(dataset_target)

with open(config_file) as f:
    y = yaml.safe_load(f)
    y['DATASETS']['TRAIN_LABEL'] = file_source
    y['DATASETS']['TRAIN_UNLABEL'] = file_train
    y['DATASETS']['TEST'] = file_test
    print

with open(output_file, 'w') as f:
    yaml.dump(y, f, default_flow_style=False, sort_keys=False)

