from pycocotools.coco import COCO
import json
from core_algorithm.compute_LiM import LiM
from core_algorithm.LiM_components import get_polys

letters = ['A','B','C','D','E']
top_n = 5
directory = 'generated_survey_data/'

dataDir = './COCO_annotations/'
dataType = 'val2017'
annFile = '{}/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)
annFile_cap = '{}/captions_{}.json'.format(dataDir, dataType)
annFile_det = '{}/instances_{}.json'.format(dataDir, dataType)
coco_caps = COCO(annFile_cap)
coco_dets = COCO(annFile_det)

fh = open('survey_data/survey_coco_ids.txt','r')
imgIds = [int(coco_id) for coco_id in fh.readlines()]
fh.close()

all_ann = json.load(open(annFile, 'r'))
categories = {cat['id']:cat['name'] for cat in all_ann['categories']}
imgs = coco.loadImgs(imgIds)
LiM_alg = LiM(categories)
data_out = []
for img in imgs:
    annIds_det = coco_dets.getAnnIds(imgIds=img['id'])
    anns_det = coco.loadAnns(annIds_det)
    captions = []
    annId_cap = coco_caps.getAnnIds(imgIds=img['id'])
    anns_cap = coco_caps.loadAnns(annId_cap)
    for ann in anns_cap:
        captions.append(ann['caption'])

    polys = get_polys(anns_det, img)
    importance_scores = LiM_alg.compute(anns_det, captions, img)

    labels = []
    for ann in anns_det:
        labels.append(categories[ann['category_id']])

    for i, ann in enumerate(anns_det):
        ann.update({'weight': importance_scores[i]})
    data_out.append({'img_info': img, 'anns': anns_det})

json_out = open('generated_COCO_survey_score.json','w')
json.dump(data_out,json_out)
json_out.close()
