# 数据预处理算法

----------------------
## 环境和依赖

- 依赖环境 myenv（相浩的python环境）
- 依赖文件 config.ini (配置文件)

## 运行方式

- 直接手动执行 `python main.py 00` 

- crontab 定时执行
    （ crontab 中的路径应与实际为准）
    ```
  30 16 * * * /home/HXKJ/code/data_process_code//main.py 00 > /home/HXKJ/code/data_process_code/00.log
  30 6 * * * /home/HXKJ/code/data_process_code//main.py 12 > /home/HXKJ/code/data_process_code/12.log
 
   ```
## 结果样例

- 配置文件中的配置文件中的save_base_path，会生成目录结构
- 插值后的站点数据应被存于： 
save_base_path/YYYY/YYYYMM/YYYYMMDD/C1DYYYYMMDDHH.csv


## 参数说明

- 时次： 00 or 12 string类型

    ps:标识世界时00时刻起报和12时刻的起报数据

## 配置文件说明：

#### 文件名config.ini

##### 以下参数需要现场修改

所需路径均来自：DataProcess_Path模块
- myenv_python_path: myenv的python绝对路径，需要与实际情况一致
- data_path: ECMWF原始文件存储路径，通常为Bz2结尾的数据
- py_path: python脚本位置 （mymain.py）
- target_path: 临时文件夹位置，被处理数据中间结果的暂存位置，一般存放解压缩后的Bz2文件和被拆分和读取后的grib文件。
- save_base_path: 插值结果的存储路径


## PS

- main.py 中的get_path函数是获取某日某时刻起报的预报数据文件路径的方法，也需要同config.ini文件配合修改。


## 出现具体的细节难题可以联系我来一起解决


#启docker 部署
### 启动
1. docker run -itd --name data_process -v /home/HXKJ/code/envs/:/opt/miniconda/envs/  -v /home/HXKJ/code/data_process_code/:/home/HXKJ/code/data_process_code -v /data/orgin_data/ECMF_DAM:/data/orgin_data/ECMF_DAM -v :/data/product_data/Forecast/corrForecast:/data/product_data/Forecast/corrForecast 153ea42154e9 bash
### 进入
2. docker exec -it data_process bash


目录：docker内部路径
      > /home/HXKJ/code/data_process_code 代码路径  [py_path]
      > /data/orgin_data/ECMF_DAM 原始预报路径ECMWF [data_path]
      > /home/HXKJ/code/data_process_code/data      [target_path]
      > /data/product_data/Forecast/corrForecast    [save_path]
      


执行时间：00 下午5.-6.
