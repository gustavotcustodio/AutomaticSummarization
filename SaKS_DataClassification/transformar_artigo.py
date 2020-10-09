import os
import numpy as np


def completezeroes(attribs, max_cols):
    diff_cols = max_cols - attribs.shape[1]
    zeroes_cols = np.zeros((attribs.shape[0], diff_cols))
    return np.hstack((attribs[:, 0:1], zeroes_cols, attribs[:, 1:]))


list_papers = os.listdir("preprocessed_papers")

max_columns = 0
paper_attrs = {}
for paper in list_papers:
    with open(os.path.join("preprocessed_papers", paper)) as f:
        attribs = np.loadtxt(f, delimiter=',', skiprows=1)[:, 1:]
        if attribs.shape[1] > max_columns:
            max_columns = attribs.shape[1]
    paper_attrs[paper] = attribs

for paper in list_papers:
    cols = paper_attrs[paper].shape[1]
    if cols < max_columns:
        paper_attrs[paper] = completezeroes(paper_attrs[paper], max_columns)
    np.savetxt(os.path.join("my_papers", paper[:-4]), paper_attrs[paper])
# 0 1>-4 -4>
