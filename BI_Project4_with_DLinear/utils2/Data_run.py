import torch
import argparse
from exp.exp_main import Exp_Main

def arg_set(folder_path, data, model_name):
    args = argparse.Namespace(
        # basic config
        is_training = 1,                                  # default = 1
        train_only = False,                               # default = False
        model_id = f'D_Linear_{data}_{model_name}',
        model = 'DLinear',                                # options: [DLinear, NLinear, Autoformer, Informer, Transformer, etc...]

        # Data Loader
        data = 'custom',                                  # I dont know ... Maybe Name?
        root_path = folder_path,                         #### Data Folder Path
        data_path = data,                  #### Data File Path
        features = 'M',                                #### Forecating task, options: [M, S, MS], M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate.
        target ='현재수요(MW)',                                  #### target feature in S or MS task. Its maybe name of target Columns

        freq = '5min',
        checkpoints = './checkpoints/',

        # Forecating Task
        seq_len = 864,
        label_len = 72,
        pred_len = 72, # 향후 6시간의 5분당 예측 전력수요 예측 길이를 72로 설정 (12*6 = 72).

        individual = False,

        # Formers
        embed_type = 0,
        enc_in = 8, 
        dec_in = 7,
        c_out = 7,
        d_model = 512,
        n_heads = 8,
        e_layers = 2,
        d_layers = 1,
        d_ff = 2048,
        moving_avg = 30,
        factor = 1,
        distil = True,
        dropout = 0.1,
        embed = 'timeF',
        activation = 'gelu',
        output_attention = False,
        do_predict = True,

        # Optimization
        num_workers = 10,
        itr = 1,
        train_epochs = 5,
        batch_size = 16,
        patience = 2,
        learning_rate = 0.001,
        des = 'Exp',
        loss = 'mse',
        lradj = 'type1',
        use_amp = False,

        # GPU Setting
        use_gpu = True,
        gpu = 0,
        use_multi_gpu = False,
        devices = '0,1,2,3',
        test_flop = False
    )
    
    return args

def model_run(args):
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if not args.train_only:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                    args.model,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)

        exp = Exp(args)  # set experiments

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()