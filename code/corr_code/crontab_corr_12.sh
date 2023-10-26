#!/bin/bash


# 获取当前时间，每间隔6min执行一次
if [ $# -eq 0 ]; then
    echo "未提供时间参数，将使用当前时间作为默认值"
    
    current_benzhan_time=`date -d "30 minutes ago" +"%Y-%m-%d %H:%M"`
    time_benzhan_param=`date -d "$current_benzhan_time" +%s`
    time_benzhan_str_param=`date -d @$time_benzhan_param "+%Y%m%d%H%M"`

    current_xichang_time=`date -d "8 hours ago 30 minutes ago" +"%Y-%m-%d %H:%M"`
    time_xichang_param=`date -d "$current_xichang_time" +%s`
    time_xichang_str_param=`date -d @$time_xichang_param "+%Y%m%d%H%M"`

else
    time_param="$1"
fi
echo $time_benzhan_str_param
echo $time_xichang_str_param
cd /mnt/PRESKY/user/lihaifei/yuchen/46/corr_code
pwd
#source /root/miniconda3/bin/activate  /root/miniconda3/envs/py38

/home/lihaifei/lihaifei/.conda/envs/myenv/bin/python main.py 12 > /mnt/PRESKY/user/lihaifei/yuchen/46/corr_code/12.log



