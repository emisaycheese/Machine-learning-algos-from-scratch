#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 23:38:56 2019

@author: ruiqianyang
"""

大概就是写Python code 对不同的string 找出每个词的频率*权重 然后算出来distance

i1 = "this is a shirt"
i2 = "this is another shirt"
i3 ="one more shirt"

Inputs= [i1,i2,i3]

# i1_tfidf= [tf-idf weight for w1_i1, tf-idf weight for w2_i1 ......., tf-idf for wk_i1In]
# i2_tfidf= [ ]
# i3_tfidf= [ ]
# Output = [i1_tfidf, i2_tfidf, i3_tfidf]
# #size of the vecor = size of the vocabulary
# tf-idf(w,d) = tf(w,d) * idf(d)
# tf(w,d) = # of w  that appear in document d
# idf (w) = # of documents / # of documents that word w appears in 
# idf (this)  = 3 / 2
#         this   is  a  shirt  anohter  one  more
#i1   1 * 1.5  
#i2
#i3
