# 订正算法

----------------------
## 环境和依赖

- 依赖环境 xgb（我自己的python环境）
- 依赖文件 config.ini (配置文件)

## 运行方式

- 直接手动执行 `python main.py 00` 

- crontab 定时执行
    （ crontab 中的路径应与实际为准）
    ```
  30 16 * * * /home/wanghaifeng/miniconda3/envs/xgb/bin/python /mnt/PRESKY/user/wanghaifeng/Projects/046/code/046_mnjf/main.py 00 > /mnt/PRESKY/user/wanghaifeng/Projects/046/code/046_mnjf/00.log
  30 16 * * * /home/wanghaifeng/miniconda3/envs/xgb/bin/python /mnt/PRESKY/user/wanghaifeng/Projects/046/code/046_mnjf/main.py 12 > /mnt/PRESKY/user/wanghaifeng/Projects/046/code/046_mnjf/00.log
 
   ```
  
## 结果样例

- 配置文件中的配置文件中的CORR_BASE_PATH，会生成目录结构
- 订正后的站点数据应被存于：
  - CORR_BASE_PATH/YYYY/YYYYMM/YYYYMMDD/YYYYMMDDHH/YYYYMMDDHH_result-Thunder.csv
  - CORR_BASE_PATH/YYYY/YYYYMM/YYYYMMDD/YYYYMMDDHH/YYYYMMDDHH_result-Tem.csv
  - CORR_BASE_PATH/YYYY/YYYYMM/YYYYMMDD/YYYYMMDDHH/YYYYMMDDHH_result-Pre.csv
  - CORR_BASE_PATH/YYYY/YYYYMM/YYYYMMDD/YYYYMMDDHH/YYYYMMDDHH_result-MaxWins_1.csv 
  - CORR_BASE_PATH/YYYY/YYYYMM/YYYYMMDD/YYYYMMDDHH/YYYYMMDDHH_result-MaxWins_2.csv


## 参数说明

- 时次： 00 or 12 string类型 （同处理数据据逻辑）

    ps:标识世界时00时刻起报和12时刻的起报数据


## 配置文件说明：

#### 文件名config.ini

##### 以下参数需要现场修改

- CORR_BASE_PATH
  - CORR_BASE_PATH: 订正结果存储路径，结果为csv文件，需要与实际情况一致
  
- STA_BASE_PATH
  - STA_BASE_PATH: 预处理好的插值到站点的数据存储位置（应与出局处理输出路径一致save_base_path）
  
- CUDA_VISIBLE_DEVICES
  - CUDA_VISIBLE_DEVICES: Cuda版本标识

- Tem_Path（温度模型所需配置）
  - scaler_path: 标准化缩放pkl位置
  - save_model_path: 温度模型保存位置
  - train_data_path: 训练数据集位置
  - is_train 训练标识 （0，1 0：预测， 1：训练 + 预测）

- MaxWins_Path （浅层风模型所需配置）
  - corr_sw_max1_path: 订正1站前层风数据存储位置
  - corr_sw_max2_path: 订正2站前层风数据存储位置
  - selected_feats_path: 特征选取模型pkl位置
  - scaler_path: 标准化缩放pkl位置
  - loss_fig_path: 损失函数曲线绘图位置
  - save_model_path： 浅层分模型保存位置
  - train_data_path： 训练数据位置
  - is_train: 训练标识 （0，1 0：预测， 1：训练 + 预测）

- Thunder_Path （雷暴模型所需配置）
  - xgboost_path： xgboost 雷暴模型存储位置
  - random_forest_path：rf 雷暴模型存储位置
  - neural_network_path：nn 雷暴模型存储位置
  - opt：训练标识 （0，1 0：预测， 1：训练 + 预测）

- Precipitation_Path （降水模型所需配置）
  - op_path: 最优百分位模型位置
  - nn_path： 神经网络模型位置

- Database （数据库参数，不过多赘述了）

## 出现具体的细节难题可以联系我来一起解决