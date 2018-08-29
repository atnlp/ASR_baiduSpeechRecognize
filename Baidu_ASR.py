
# coding: utf-8

# In[1]:


from aip import AipSpeech
import wer3
import json
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
# 改成自己百度语音识别中的ID
APP_ID = '****'
API_KEY = '****'
SECRET_KEY = '****'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


# In[2]:


import importlib
importlib.reload(wer3)


# In[4]:


text = open(r'transcript/aishell_transcript_v0.8.txt', encoding='UTF-8')
# text.seek(1000, 0)
# 存放每个预测结果的列表
cache_result = []
# 存放每个文件名的列表
text_name = []
wer_score = DataFrame(columns=['id', 'WER'])
# 本次测试的文件数目
file_num = 30
result = np.zeros(file_num)
# 测试的数据的条目
for i in range(file_num):
    text_line = text.readline()
    text_name.append(text_line[: 16] + '.wav')
    # replace 去除字符串中的空格。strip只能去除开头和结尾处
    text_line = text_line[17: -1].replace(' ', '')
    text_list = []
    for s in text_line:
        text_list.append(s)
    
    # text_name.append(text_line[: 16] + '.wav')
    filename = os.path.join(r'S0002', text_name[i])
    # 读取文件
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    # 识别本地文件
    pre = client.asr(get_file_content(filename), 'pcm', 16000, {
    'dev_pid': 1536
    })
    cache_result.append(pre['result'])
    pre_list = []
    for s in ' '.join(cache_result[i]):
        pre_list.append(s)
    print('index', i+1)
    part_result = wer3.wer(pre_list, text_list)[: -1]
    result[i] = part_result
    wer_score.loc[i, 'id'] = text_name[i]
    wer_score.loc[i, 'WER'] = part_result
print('final_predict:', result.mean())

