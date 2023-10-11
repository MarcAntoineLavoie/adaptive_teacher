# Modified from pycocotools/cocoeval.py

from pycocotools.cocoeval import COCOeval
import numpy as np
import datetime
import time
from collections import defaultdict
import pycocotools.mask as maskUtils
import copy
import torch
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures.boxes import Boxes
from detectron2.structures import pairwise_iou


class prob_COCOeval(COCOeval):
    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self.bbox_dif = Box2BoxTransform((10.0, 10.0, 5.0, 5.0))

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def evalprobsImg(self, gt_bbox, dt_bbox, bbox_cov):
        gt_bbox = torch.tensor(gt_bbox)
        dt_bbox = torch.tensor(dt_bbox)
        gt_bbox[2:] = gt_bbox[:2] + gt_bbox[2:]
        dt_bbox[2:] = dt_bbox[:2] + dt_bbox[2:]
        bbox_cov = torch.tensor(bbox_cov)

        deltas = self.bbox_dif.get_deltas(dt_bbox.unsqueeze(0), gt_bbox.unsqueeze(0)).squeeze(0)

        if (deltas > 100).any():
            a=1

        #lower_tri = torch.eye(len(bbox_cov)) * (bbox_cov**0.5 + 1e-6) 
        #bbox_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), scale_tril=lower_tri)
        bbox_dist = torch.distributions.normal.Normal(torch.zeros(4), bbox_cov**0.5)

        cdf = bbox_dist.cdf(deltas)

        # Compute negative log probability
        negative_log_prob = -bbox_dist.log_prob(deltas).mean()

        # Energy Score.
        sample_set = bbox_dist.sample((1001,))
        sample_set_1 = sample_set[:-1]
        sample_set_2 = sample_set[1:]

        energy_score = torch.norm((sample_set_1),dim=1).mean(0) - 0.5*torch.norm((sample_set_1 - sample_set_2), dim=1).mean(0)


        # GT overlap / Self Overlap
        prob_bboxes = self.bbox_dif.apply_deltas(sample_set_1, dt_bbox.unsqueeze(0))
        mean_IoU_GT = pairwise_iou(Boxes(prob_bboxes), Boxes(gt_bbox.unsqueeze(0))).mean()
        mean_IoU = pairwise_iou(Boxes(prob_bboxes), Boxes(dt_bbox.unsqueeze(0))).mean()

        return negative_log_prob, energy_score, mean_IoU_GT, mean_IoU, deltas, cdf

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        box_NLL = np.zeros((T,D))
        box_ES = np.zeros((T,D))
        box_gtIoU = np.zeros((T,D))
        box_dtIoU = np.zeros((T,D))
        box_deltas = np.zeros((T,D,4))
        box_cdf = np.zeros((T,D,4))

        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']

        dtm_check = dtm*(dtIg-1)
        matches = {}
        for lv1 in range(T):
            for lv2 in range(D):
                if dtm_check[lv1, lv2]:
                    ids = (self._dts[imgId, catId][lv2]['id'], dtm[lv1, lv2])
                    if ids in matches.keys():
                        box_NLL[lv1,lv2], box_ES[lv1,lv2], box_gtIoU[lv1,lv2], box_dtIoU[lv1,lv2], box_deltas[lv1,lv2,:], box_cdf[lv1,lv2,:] = matches[ids]
                    else:
                        bbox_gt = [x['bbox'] for x in self._gts[imgId, catId] if x['id']==dtm[lv1,lv2]][0]
                        bbox_dt = self._dts[imgId, catId][lv2]['bbox']
                        covs_dt = self._dts[imgId, catId][lv2]['bbox_covs']
                        box_NLL[lv1,lv2], box_ES[lv1,lv2], box_gtIoU[lv1,lv2], box_dtIoU[lv1,lv2], box_deltas[lv1,lv2,:], box_cdf[lv1,lv2,:] = self.evalprobsImg(bbox_gt, bbox_dt, covs_dt)
                        matches[ids] = (box_NLL[lv1,lv2], box_ES[lv1,lv2], box_gtIoU[lv1,lv2], box_dtIoU[lv1,lv2], box_deltas[lv1,lv2,:], box_cdf[lv1,lv2,:])

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
                'dtBoxNLL':     box_NLL,
                'dtBoxES':      box_ES,
                'gtIoUs':       box_gtIoU,
                'dtIoUs':       box_dtIoU,
                'dtDeltas':     box_deltas,
                'dtCDF':        box_cdf,
            }
    
    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        dtIoUs = []
        gtIoUs = []
        cdfs = []
        deltas = []

        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)

                    for e in E:
                        dtIoUs = dtIoUs + e['dtIoUs'][0,:].tolist()
                        gtIoUs = gtIoUs + e['gtIoUs'][0,:].tolist()
                        cdfs = cdfs + e['dtCDF'][0,:,:].tolist()
                        deltas = deltas + e['dtDeltas'][0,:,:].tolist()
                        if (e['dtDeltas'][0,:,:] > 10).any():
                            a=1

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))


# import matplotlib.pyplot as plt
# dtIoUs_2 = [x for x in dtIoUs if x > 0]
# gtIoUs_2 = [x for x in gtIoUs if x > 0]

# bins=np.arange(0,1.05,0.05)
# fig = plt.figure()
# axs = fig.subplots(2, 2, sharex=True, sharey=True)
# axs[0,0].hist(cdfs_2[:,0], bins=bins, cumulative=True, density=True)
# axs[0,0].plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1),':')
# axs[0,0].set_xlabel('x error cdf')
# axs[0,1].hist(cdfs_2[:,1], bins=bins, cumulative=True, density=True)
# axs[0,1].plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1),':')
# axs[0,1].set_xlabel('y error cdf')
# axs[1,0].hist(cdfs_2[:,2], bins=bins, cumulative=True, density=True)
# axs[1,0].plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1),':')
# axs[1,0].set_xlabel('width error cdf')
# axs[1,1].hist(cdfs_2[:,3], bins=bins, cumulative=True, density=True)
# axs[1,1].plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1),':')
# axs[1,1].set_xlabel('height error cdf')
# fig.suptitle('Calibration Plots')
# plt.tight_layout()


# plt.figure
# plt.hist(dtIoUs_2, alpha=1.0)
# plt.xlabel('Predicted Overlap')

# cdfs_2 = np.array([x for x in cdfs if any(x) > 0])
# deltas_2 = np.array([x for x in deltas if any(abs(y) for y in x) > 0])

# plt.figure();plt.hist(cdfs_2[:,0], bins=bins, cumulative=True, density=True);plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1),':')

# xs = np.arange(-4.0,4.1,0.1)
# ys = np.e**(-0.5*xs**2)/(2*np.pi)**.5
# bins_ = xs
# plt.figure();plt.hist(deltas_2[:,0], density=True, bins=bins_);plt.plot(xs,ys)
# plt.figure();plt.hist(deltas_2[:,1], density=True, bins=bins_);plt.plot(xs,ys)
# plt.show()

# import json
# with open('/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/datasets/cityscapes_foggy_val_coco_format.json','r') as f_in:
#     data = json.load(f_in)

# vals = [x for x in data['annotations'] if x['image_id'] == 'frankfurt_000000_011461_leftImg8bit_foggy_beta_0.005.png']
# from detectron2.structures.boxes import BoxMode
# for val in vals:
#     val['bbox_mode'] = BoxMode.XYWH_ABS
# vals = {'annotations':vals}

# img = 255*plt.imread('/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/datasets/cityscapes_foggy/leftImg8bit/val/frankfurt/frankfurt_000000_011461_leftImg8bit_foggy_beta_0.005.png')
# img2 = 255*plt.imread('/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/datasets/cityscapes_foggy/gtFine/val/frankfurt/frankfurt_000000_011461_gtFine_instanceIds.png')
# img3 = 255*plt.imread('/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/datasets/cityscapes_foggy/gtFine/val/frankfurt/frankfurt_000000_011461_gtFine_labelIds.png')
