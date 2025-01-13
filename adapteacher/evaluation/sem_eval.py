import numpy as np
from detectron2.evaluation.sem_seg_evaluation import SemSegEvaluator, load_image_into_numpy_array
from cityscapesscripts.helpers.labels import id2label

class newSemsSegEvaluator(SemSegEvaluator):
    def __init__(self, dataset_name, distributed=True, output_dir=None, *, sem_seg_loading_fn=load_image_into_numpy_array, num_classes=None, ignore_label=None, seg_instances_only=False):
        super().__init__(dataset_name, distributed, output_dir, sem_seg_loading_fn=sem_seg_loading_fn, num_classes=num_classes, ignore_label=ignore_label)
        self._seg_instances_only = seg_instances_only
        if self._seg_instances_only:
            self._num_classes = 8
            self._class_names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
            self.reset()


    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int, copy=True)
            # gt = gt.copy()
            labelIds = np.unique(gt)
            if self._seg_instances_only:
                class2id = {'person':0,'rider':1,'car':2,'truck':3,'bus':4,'train':5,'motorcycle':6,'bicycle':7}
                for labelId in labelIds:
                    class_ = id2label[labelId].name
                    trainId = class2id.get(class_, 255)
                    mask = gt == labelId
                    gt[mask] = trainId
                # self._num_classes = 8
            else:
                for labelId in labelIds:
                    trainId = id2label[labelId].trainId
                    mask = gt == labelId
                    gt[mask] = trainId

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))