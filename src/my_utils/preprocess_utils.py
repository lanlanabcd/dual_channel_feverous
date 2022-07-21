# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/10/28 20:15
# Description:

import re


pt1 = re.compile("LSB.*?RSB ")
pt2 = re.compile("LRB RRB ")
pt3 = re.compile("\(( [,ˌ;])+")
pt32 = re.compile("([,ˌ;] )+\)")
pt4 = re.compile("[LR][SCR]B ")


def clean_wiki_str(s):
    s = re.sub(pt1, '', s)
    s = re.sub(pt2, '', s)
    s = re.sub('LRB', '(', s)
    s = re.sub('RRB', ')', s)
    s = re.sub(pt3, '(', s)
    s = re.sub(pt32, ')', s)
    s = re.sub("\( \) ", '', s)
    s = re.sub(pt4, '', s)
    s = re.sub('--', '-', s)
    s = re.sub("``", '"', s)
    s = re.sub("''", '"', s)

    tokens = s.split(' ')
    mask = [1 for _ in range(len(tokens))]
    for i, token in enumerate(tokens):
        if mask[i] == 0:
            continue
        for j in range(max(0, i - 10), i):
            if tokens[i] == tokens[j]:
                k = j
                while k < i and i + k - j < len(tokens) and mask[k] == 1 and tokens[k] == tokens[i + k - j]:
                    k += 1
                if k == i:
                    for p in range(j, i):
                        mask[p] = 0
    tokens = [t for t, m in zip(tokens, mask) if m == 1]
    s = ' '.join(tokens)
    return s

def clean_wiki_title(title):
    title = re.sub("_", " ", title)
    title = re.sub("-LRB-", "( ", title)
    title = re.sub("-RRB-", " )", title)
    title = re.sub("COLON", " :", title)
    return title

def remove_bracket_w_nonascii(s):
    '''
    remove bracketed sub-strings with more than 2 or more than 10% non ascii charactors
    :param s:
    :return: cleaned s
    '''
    p = 0
    lrb = s.find("(")
    while lrb != -1:
        rrb = s.find(")", lrb)
        if rrb == -1:
            rrb = -3
            #print(s)
        r = s[lrb:rrb+1]
        innormal_chars_num = ascii(r).count("\\u") + ascii(r).count("\\x")
        try:
            if len(r) == 1 or innormal_chars_num >= 2 or innormal_chars_num * 1.0 / len(r) >= 0.1:
                s = s[:lrb] + s[rrb + 2:]
            else:
                p = rrb
        except:
            s = s[:lrb]
        lrb = s.find("(", p)
    return s