# File: yta.py
# Author: Murilo Bento
#
# MIT License
#
# Copyright (c) 2020 Murilo Bento
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# --- IMPORTS ---

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------

# --- FUNCTIONS TO CLASSIFY ELEMENTS ---

def classify_views(element):
  if element > 1000000:
    return 'Above one million'
  else:
    return 'Below one million'

def classify_likes(element):
  if element > 20000:
    return 'Above 20k'
  else:
    return 'Below 20k'

def classify_dislikes(element):
  if element > 1000:
    return 'Above 1k'
  else:
    return 'Below 1k'

def classify_comments(element):
  if element > 1000:
    return 'Above 1k'
  else:
    return 'Below 1k'

# ---------------

# --- DATA ACQUISITION ---

data_canada = pd.read_csv('input/CAvideos.csv', encoding='utf8')
data_us = pd.read_csv('input/USvideos.csv', encoding='utf8')

# ---------------

# --- DATA REFINEMENT ---

dc_r = data_canada.iloc[:, [0, 1, 2, 3, 4, 7, 8, 9, 10]].copy()
dus_r = data_us.iloc[:, [0, 1, 2, 3, 4, 7, 8, 9, 10]].copy()

grc = dc_r.groupby(['video_id'])
gru = dus_r.groupby(['video_id'])

dc_r.update(grc.transform("max"))
dus_r.update(gru.transform("max"))

dc_r = dc_r.drop_duplicates("video_id", keep='last')
dus_r = dus_r.drop_duplicates("video_id", keep='last')

# ---------------

# --- COMBINING DATA ---

left = dc_r.set_index(['title', 'trending_date'])
right = dus_r.set_index(['title', 'trending_date'])

cols_to_use = right.columns.difference(left.columns)

merged = pd.merge(left=left, right=right[cols_to_use], on=['title', 'trending_date'])

# ---------------

# --- TRANSFORMING DATA ---

views_c = merged['views'].apply(classify_views)
likes_c = merged['likes'].apply(classify_likes)
dislikes_c = merged['dislikes'].apply(classify_dislikes)
comments_c = merged['comment_count'].apply(classify_comments)

classified = pd.concat([merged.loc[:, ["channel_title", "category_id"]], likes_c, dislikes_c, views_c, comments_c], axis=1)
# ---------------

# --- PLOTTING ---

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))

classified.groupby(["category_id", "views"]).size().unstack().plot.bar(title="Views", ax=ax[0])
classified.groupby(["category_id", "likes"]).size().unstack().plot.bar(title="Likes", ax=ax[1])
classified.groupby(["category_id", "dislikes"]).size().unstack().plot.bar(title="Dislikes", ax=ax[2])
classified.groupby(["category_id", "comment_count"]).size().unstack().plot.bar(title="Comment Count", ax=ax[3])

fig.suptitle("Youtube Trending Analysis", fontsize=14)

plt.savefig("output/youtube-trending-analysis.png", dpi=80)

# ---------------
