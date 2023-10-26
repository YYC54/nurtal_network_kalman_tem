# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :main.py
# @File     :utills
# @Date     :2023/6/8 16:49
# @Author   :HaiFeng Wang
# @Email    :bigc_HF@163.com
# @Software :PyCharm
-------------------------------------------------
"""
# 存储describe
import pandas as pd
import pandas.io.formats.excel


def summary2xlsx(df):
    """
    将DataFrame的描述信息（describe）输出到Excel的xlsx中，其文件名为summary，sheet为describe
    :param df: DataFrame 数据表格
    :return: None
    """
    pandas.io.formats.excel.header_style = None
    summary_writer = pd.ExcelWriter('summary.xlsx')
    df.describe(include='all').to_excel(summary_writer, sheet_name='describe')
    summary_writer.save()
    return None

