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

#source /home/lihaifei/lihaifei/.conda/envs/py38
cd /mnt/PRESKY/user/lihaifei/yuchen/46/data_process_code
pwd
# 处理插值数据
/home/lihaifei/lihaifei/.conda/envs/myenv/bin/python  main.py 00 > /mnt/PRESKY/user/lihaifei/yuchen/46/data_process_code/00.log
# 对插值数据进行订正
sh /mnt/PRESKY/user/lihaifei/yuchen/46/corr_code/crontab_corr_00.sh


#current_time=`date -d "+%Y-%m-%d %H:%M"`
#time_param=`date -d "$current_time" +%s`
#input_yyyymmdd_data=`date -d @$time_param "+%Y-%m-%d"`
#input_yyyytmmddHH_data=`date -d @$time_param "+%Y-%m-%d 00"`
#echo $input_yyyymmdd_data
#echo $input_yyyytmmddHH_data

#echo "***00 EC NC Diag Start!!!***"

#cd /home/HXKJ/code/zhengduan_code/
#source /root/miniconda3/bin/activate  /root/miniconda3/envs/py38
#python Diagnose_046_NC_ec.py $input_yyyytmmddHH_data > log/Diagnose_046_NC_ec_00.log
#echo "***00 EC NC Diag End!!!***"

