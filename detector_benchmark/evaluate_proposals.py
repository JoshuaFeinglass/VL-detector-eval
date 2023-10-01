import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from modified_coco_eval_summarize import summarize
import types
import sys
def F1(a,b):
    a = (abs(a) + a) / 2
    b = (abs(b) + b) / 2
    if a>0 or b>0:
        return 2*a*b/(a+b)
    else:
        return 0

proposal_dir = 'baseline_proposals/'
proposal_file = proposal_dir+sys.argv[1]
gt_dir = 'thresholded_annotations/'
annFile = gt_dir+sys.argv[2] #'vg_gt_thres30.json'
vg_gt = COCO(annFile)
detector_props = np.fromfile(proposal_file).reshape(-1,7)
cocoEval = COCOeval(vg_gt, vg_gt.loadRes(detector_props), 'bbox')
cocoEval.summarize = types.MethodType(summarize, cocoEval)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
f1_score = F1(cocoEval.stats[1],cocoEval.stats[6])
print('Precision: '+str(cocoEval.stats[1]))
print('Recall: '+str(cocoEval.stats[6]))
print('F1 Score: '+str(f1_score))