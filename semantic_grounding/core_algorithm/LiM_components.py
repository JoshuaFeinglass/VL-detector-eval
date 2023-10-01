import numpy as np
import spacy
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.ops import cascaded_union
from gensim.models import KeyedVectors
from pygsp import graphs,filters
import cv2
import sys
from .mask import *
def get_polys(anns_det,img):
    polys = []
    for ann in anns_det:
        if 'segmentation' in ann:
            coords = ann['segmentation']
            if type(coords) == list:
                polys.append(
                    cascaded_union([Polygon([(x, y) for x, y in zip(coord[0::2], coord[1::2])]) for coord in coords]))

            else:
                if type(ann['segmentation']['counts']) == list:
                    rle = frPyObjects([ann['segmentation']],
                                                img['height'],
                                                img['width'])
                else:
                    rle = [ann['segmentation']]
                m = decode(rle)
                coords = cv2.findContours(m.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
                polys.append(
                    cascaded_union([Polygon([tuple(pt[0]) for pt in coord]) for coord in coords if len(coord) >= 3]))
        else:
            coords = ann['bbox']
            polys.append(box(coords[0], coords[1], coords[2], coords[3]))
    return polys
def graph_propogation(att, adj_mat):
    sal_graph = graphs.Graph(adj_mat)
    g = filters.Heat(sal_graph,tau=1)
    return g.filter(att)
class concept_matcher():
    def __init__(self,categories):#,priors):
        self.embedding = KeyedVectors.load_word2vec_format('requirements/numberbatch-en-19.08.txt', binary=False)
        #self.embedding = KeyedVectors.load_word2vec_format('word_models/GoogleNews-vectors-negative300.bin', binary=True)
        self.nlp = spacy.load("en_core_web_md")
        self.cat_doc = []
        self.seen = []
        for key in sorted(categories.keys()):
            sup_cat = categories[key]
            if sup_cat in self.seen:
                continue
            if sup_cat == 'hot dog':
                self.cat_doc.append('sausage')
            elif sup_cat.replace(' ','_') in self.embedding:
               self.cat_doc.append(sup_cat.replace(' ','_'))
            else:
               self.cat_doc.append(sup_cat.replace(' ',''))
            self.seen.append(sup_cat)

    def caption_typicality(self,all_sent):
        sing_sents = []
        full_set = set()
        for sent in all_sent:
            toks = self.nlp(sent)
            objs = [tok.text.lower() for tok in toks if tok.text.lower() in self.embedding and tok.dep_ in ['nsubj','dobj','pobj','ROOT'] and tok.pos_ == 'NOUN']
            objs = set(objs)
            obj_cats = []
            for obj in objs:
                cat_scores = []
                for cat in self.cat_doc:
                    try:
                        cat_scores.append(self.embedding.similarity(cat,obj))
                    except:
                        print(obj+' not present in conceptnet')
                obj_cats.append(self.seen[np.argmax(cat_scores)])
            obj_cats = set(obj_cats)
            sing_sents.append(obj_cats)
            full_set = full_set.union(obj_cats)
        stem_dict = dict()
        for concept in full_set:
            relevance = sum(concept in sent for sent in sing_sents)
            stem_dict[concept] = relevance/len(sing_sents)
        return stem_dict
