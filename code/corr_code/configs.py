config = {
    # -----------------------------common-----------------------------------
    "CORR_BASE_PATH": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/res",
    # 22年及之前数据
    "STA_BASE_PATH": '/share/HPC/home/zhaox_hx/data/aidata/zhangym_tmp/DATA/ML_dataset/Corr_datasets',
    # 23年数据
    # "STA_BASE_PATH": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/data/",
    "CUDA_VISIBLE_DEVICES": "2",
    # -----------------------------max_winds-------------------------------
    "maxWins": {
        "corr_sw_max1_path": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/lib/Max_Winds/result/save/sw_max1_corr.csv",
        "corr_sw_max2_path": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/lib/Max_Winds/result/save/sw_max1_corr.csv",
        "selected_feats_path": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/lib/Max_Winds/result/save/selected_feats.pkl",
        "scaler_path": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/lib/Max_Winds/result/save/scaler.pkl",
        "loss_fig_path": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/lib/Max_Winds/result/save/",
        "save_model_path": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/lib/Max_Winds/result/model/",
        "train_data_path": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/lib/Max_Winds/dataset/qiancengfeng_leibao.csv",
        "is_train": False,
        "CONSTANT_FEATS":
            [
                "time",
                "time_order",
                "dtime",
                "id",
                "lon",
                "lat",
                "stime",
                "北京时间",
            ]

    },
    # -----------------------------temp-------------------------------
    "tem": {
        "scaler_path": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/lib/Tem/dataset/scaler.pkl",
        "save_model_path": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/lib/Tem/save/temp.h5",
        "train_data_path": "/mnt/PRESKY/user/wanghaifeng/Projects/046/code/mergecode_i/lib/Tem/dataset/qiancengfeng_leibao.csv",
        "is_train": False

    }

}
