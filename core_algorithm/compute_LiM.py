from .LiM_components import concept_matcher, get_polys, graph_propogation
import numpy as np

class LiM:
    def __init__(self, categories):
        self.matcher = concept_matcher(categories)
        self.categories = categories
    def compute(self, anns_det, captions, img_info = None):
        top_concept = self.matcher.caption_typicality(captions)
        if len(anns_det)>1:
            polys = get_polys(anns_det,img_info)
            obj_cats = []
            obj_areas = {}
            adj_mat = np.zeros((len(anns_det), len(anns_det)))
            att_val = np.zeros(len(anns_det))
            for i in range(0,len(anns_det)):
                obj_cat = self.categories[anns_det[i]['category_id']]
                obj_area = anns_det[i]['area']
                if obj_cat in obj_areas:
                    obj_areas[obj_cat] += obj_area
                else:
                    obj_areas[obj_cat] = obj_area
                obj_cats.append(obj_cat)
                if obj_cat in top_concept.keys():
                    att_val[i] = top_concept[obj_cat]*obj_area
                else:
                    att_val[i] = 0

                for j in range(0,i+1):
                    if i == j:
                        adj_mat[i,j] = 0
                    else:
                        weight = 1/max(polys[i].distance(polys[j]),1)
                        adj_mat[i,j] = weight
                        adj_mat[j,i] = weight

            for i in range(0,len(anns_det)):
                att_val[i] /= obj_areas[obj_cats[i]]

            importance_score = graph_propogation(att_val, adj_mat)
            importance_score = importance_score / np.sum(importance_score)

        elif len(anns_det)==1:
            importance_score = [1]

        return importance_score