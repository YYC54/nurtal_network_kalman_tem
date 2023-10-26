#!/usr/bin/env python

# -*- coding:utf-8 -*-
"""
@author: zhangym
@contact: 976435584@qq.com
@software: PyCharm
@file: logging_info.py
@time: 2021/6/24 9:25
@Describe 
@Version 1.0
"""
import logging
import ctypes

FOREGROUND_WHITE = 0x0007
FOREGROUND_BLUE = 0x01 # text color contains blue.
FOREGROUND_GREEN= 0x02 # text color contains green.
FOREGROUND_RED = 0x04 # text color contains red.
FOREGROUND_YELLOW = FOREGROUND_RED | FOREGROUND_GREEN


class Logger(object):
    '''
    # debug: 打印全部的日志, 详细的信息, 通常只出现在诊断问题上
    #
    # info: 打印info, warning, error, critical级别的日志, 确认一切按预期运行
    #
    # warning: 打印warning, error, critical级别的日志, 一个迹象表明, 一些意想不到的事情发生了, 或表明一些问题在不久的将来(例如。磁盘空间低”), 这个软件还能按预期工作
    #
    # error: 打印error, critical级别的日志, 更严重的问题, 软件没能执行一些功能
    #
    # critical: 打印critical级别, 一个严重的错误, 这表明程序本身可能无法继续运行

    '''
    def __init__(self,path,clevel= logging.DEBUG,Flevel=logging.DEBUG):
        '''
        Logging.Formatter：这个类配置了日志的格式，在里面自定义设置日期和时间，输出日志的时候将会按照设置的格式显示内容。
        1. 为程序提供记录日志的接口
        2. 判断日志所处级别，并判断是否要过滤
        3. 根据其日志级别将该条日志分发给不同handler
        常用函数有：
        Logger.setLevel() 设置日志级别
        Logger.addHandler() 和 Logger.removeHandler() 添加和删除一个Handler
        Logger.addFilter() 添加一个Filter,过滤作用
        Logging.Handler：Handler基于日志级别对日志进行分发，如设置为WARNING级别的Handler只会处理WARNING及以上级别的日志。
        常用函数有：
        setLevel() 设置级别
        setFormatter() 设置Formatter

        :param path: 配置log文件路径
        :param clevel: 设置级别
        :param Flevel: 设置级别
        '''
        #
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # 设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        # 设置文件日志
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)
        #  添加下面一句，在记录日志之后移除句柄
        # self.remove = self.logger.removeHandler(sh)


    def debug(self,message):
        self.logger.debug(message)

    def info(self,message):
        self.logger.info(message)
        # self.logger.handlers.pop()

    def warning(self,message,color=FOREGROUND_YELLOW):
        self.logger.warning(message)

    def error(self,message,color=FOREGROUND_RED):
        self.logger.error(message)

    def cri(self,message):
        self.logger.critical(message)

if __name__ == '__main__':
    log = Logger(path='test.log',clevel=logging.ERROR,Flevel=logging.DEBUG)
    log.debug('debug 信息')
    log.info('info 信息')
    log.info('info1 信息')
    log.info('info2 信息')
    log.warning('warning 信息')
    log.error('一个error信息')
    log.cri('一个致命critical信息')

