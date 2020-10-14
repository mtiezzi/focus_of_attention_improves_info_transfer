import os
import lve
import math
import copy
import collections
import shutil
from multiprocessing import Process
from argparse import ArgumentParser
import torch


def run_on_process(ins, model_options, outs, exp_root_train, model_folder, first_video_of_the_list):
    worker = lve.WorkerCAL(ins.w, ins.h, ins.c, ins.fps, options=model_options)
    vp = lve.VProcessor(ins, outs, worker, os.path.join(exp_root_train, model_folder),
                        visualization_port=-1, resume=not first_video_of_the_list)
    vp.process_video()


def run_train_and_test(exp_root, model_id, model_options, train_video_or_folder, test_video=None, foa_folder=None,
                       stream=None, rnd=None):
    # each experiment root folder has two sub-folders: train and test
    exp_root_train = os.path.join(exp_root, 'train')
    exp_root_test = os.path.join(exp_root, 'test')

    if not os.path.exists(exp_root_train):
        os.mkdir(exp_root_train)
    if not os.path.exists(exp_root_test):
        os.mkdir(exp_root_test)


    rescale = True
    if not os.path.isdir(os.path.abspath(train_video_or_folder)):
        train_video_list = [train_video_or_folder]  # single video
    else:
        train_video_list = [os.path.join(train_video_or_folder, f)
                            for f in sorted(os.listdir(train_video_or_folder)) if f.endswith(".mov")]
        if len(train_video_list) == 0:
            train_video_list = [train_video_or_folder]  # frames in a folder (we pass the folder to input stream)
            rescale = False

    first_video_of_the_list = True

    # for each video... (in most of the case it is a single video)
    for train_video in train_video_list:
        if first_video_of_the_list:
            model_folder = "model_" + os.path.basename(train_video_or_folder) + "_" + model_id
            output_folder = "output_" + os.path.basename(train_video_or_folder) + "_" + model_id

            # checking if this experiment was already performed (train)
            if os.path.exists(os.path.join(exp_root_train, model_folder)) \
                    and os.path.exists(os.path.join(exp_root_train, model_folder, "train_completed")):
                print("Skipping experiment (already done): " + str(os.path.join(exp_root_train, model_folder)))
                continue
            if os.path.exists(os.path.join(exp_root_train, model_folder)):
                shutil.rmtree(os.path.join(exp_root_train, model_folder))

        # input stream
        ins = lve.InputStream(train_video, w=240 if rescale else -1, h=180 if rescale else -1,
                              fps=None, max_frames=100000,
                              repetitions=1, force_gray=True,
                              foa_file=os.path.join(foa_folder, os.path.basename(train_video)) + ".foa"
                              if foa_folder is not None else None)
        foa_opt = {'alpha_c': 0.7,
                   'alpha_of': 0.0,
                   'alpha_fm': 0.0,
                   'alpha_virtual': 1.0,
                   'max_distance': int(0.5 * (ins.w + ins.h)) if int(0.5 * (ins.w + ins.h)) % 2 == 1 else int(
                       0.5 * (ins.w + ins.h)) + 1,
                   'dissipation': 0.15,
                   'fps': ins.fps,
                   'w': ins.w,
                   'h': ins.h,
                   'y': [ins.h // 2, ins.w // 2 + 30, -1.0, 1.0],  # fixed initialization
                   'is_online': False,
                   'fixation_threshold_speed': int(0.05 * 0.5 * (ins.w + ins.h))}

        if stream == "carpark" or stream == "call":  # foa parameters exploited in the experiments
            foa_opt["alpha_of"] = 1.0
            foa_opt["alpha_c"] = 0.1
            foa_opt["dissipation"] = 0.1
        if rnd:
            foa_opt["alpha_of"] = 0.0
            foa_opt["alpha_c"] = 0.0
            foa_opt["dissipation"] = -1

        # determining the number of repetitions to reach 100,000 frames

        frames = ins.frames_orig
        repetitions = math.ceil(100000.0 / float(frames))
        ins.set_options(repetitions=repetitions)

        # setting the input-stream-related options
        model_options['supervision_map'] = ins.sup_map
        model_options['net']['c'] = ins.c
        model_options['foa'] = foa_opt

        # output stream
        outs = lve.OutputStream(os.path.join(exp_root_train, output_folder), ins.fps,
                                virtual_save=False, tensorboard=False, save_per_frame_data=False,
                                purge_existing_data=True)

        p = Process(target=run_on_process, args=(ins, model_options, outs, exp_root_train, model_folder,
                                                 first_video_of_the_list))
        p.start()
        p.join()
        ins.close()
        outs.close()

        first_video_of_the_list = False

    # marking the experiment as completed (train)
    open(os.path.join(exp_root_train, model_folder, "train_completed"), 'w').close()

    # test stage starts here
    if test_video is None:
        test_video = train_video

    # checking if this experiment was already performed (test)
    if os.path.exists(os.path.join(exp_root_test, model_folder)) \
            and os.path.exists(os.path.join(exp_root_test, model_folder, "test_completed")):
        print("Skipping experiment (already done): " + str(os.path.join(exp_root_test, model_folder)))
        return

    if os.path.exists(os.path.join(exp_root_test, model_folder)):  # remove test folder in all cases
        shutil.rmtree(os.path.join(exp_root_test, model_folder))

    # copying model folder
    shutil.copytree(os.path.join(exp_root_train, model_folder), os.path.join(exp_root_test, model_folder))

    # input stream (single repetition of the whole test video)
    ins = lve.InputStream(test_video, w=240 if rescale else -1, h=180 if rescale else -1,
                          fps=None, max_frames=-1,
                          repetitions=1, force_gray=True,
                          foa_file=None)

    # determining the number of repetitions to reach 5,000 frames (test video)
    frames = ins.frames_orig
    repetitions = math.ceil(5000.0 / float(frames))
    ins.set_options(repetitions=repetitions)

    # setting the input-stream-related options
    model_options['supervision_map'] = ins.sup_map
    model_options['net']['c'] = ins.c
    model_options['foa'] = foa_opt

    if stream == "carpark" or stream == "call":  # foa parameters exploited in the experiments
        foa_opt["alpha_of"] = 1.0
        foa_opt["alpha_c"] = 0.1
        foa_opt["dissipation"] = 0.1

    # freezing
    model_options['net']['training']["freezeall"] = True

    # output stream
    outs = lve.OutputStream(os.path.join(exp_root_test, output_folder), ins.fps,
                            virtual_save=False, tensorboard=False, save_per_frame_data=False,
                            purge_existing_data=True)

    p = Process(target=run_on_process, args=(ins, model_options, outs, exp_root_test, model_folder,
                                             False))
    p.start()
    p.join()

    ins.close()
    outs.close()

    # marking the experiment as completed (test)
    open(os.path.join(exp_root_test, model_folder, "test_completed"), 'w').close()


def set_seedA(options, model_id):
    options['seed'] = 8988979
    op_id = 'SeedA'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_lambda_c_lambda_e(options, lambda_c, lambda_e):
    layers = options['net']['fe_layers']
    options['net']['lambda_c'] = [0.0] * layers
    options['net']['lambda_e'] = [0.0] * layers
    options['net']['lambda_c'][-1] = lambda_c
    options['net']['lambda_e'][-1] = lambda_e


def set_full_frame_mi(options, model_id):
    options['net']['foa_mi'] = False
    options['net']['foa_mi_use_window'] = False
    op_id = 'MIUNI'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_foa_mi(options, model_id):
    options['net']['foa_mi'] = True
    options['net']['foa_mi_use_window'] = False
    op_id = 'MIFOA'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_foa_window_mi(options, model_id):
    options['net']['foa_mi'] = True
    options['net']['foa_mi_use_window'] = True
    op_id = 'MIFOAW'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_not_accumulated_entropy(options, model_id):
    layers = options['net']['fe_layers']
    options['net']['lambda_s'] = [0.0] * layers
    options['net']['zeta_s'] = [1.0] * layers
    op_id = 'PLA'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_moving_average_entropy(options, model_id, zeta_s):
    layers = options['net']['fe_layers']
    options['net']['lambda_s'] = [0.0] * layers
    options['net']['zeta_s'] = [1.0] * layers
    options['net']['zeta_s'][-1] = zeta_s
    op_id = 'Avg'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_variation_based_entropy(options, model_id, lambda_s):
    layers = options['net']['fe_layers']
    options['net']['lambda_s'] = [0.0] * layers
    options['net']['zeta_s'] = [1.0] * layers
    options['net']['lambda_s'][-1] = lambda_s
    op_id = 'Var'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_layers1(options, model_id):
    options['net']['fe_layers'] = 1
    options['net']['sem_layers'] = 0
    op_id = 'Layers1'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_layers3(options, model_id):
    options['net']['fe_layers'] = 3
    options['net']['sem_layers'] = 0
    op_id = 'Layers3'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_layers6(options, model_id):
    options['net']['fe_layers'] = 6
    options['net']['sem_layers'] = 0
    op_id = 'Layers6'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_layers7(options, model_id):
    options['net']['fe_layers'] = 7
    options['net']['sem_layers'] = 0
    op_id = 'Layers7'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_order2(options, model_id):
    layers = options['net']['fe_layers']

    options['net']['order'] = 2
    options['net']['alpha'] = [0.01] * layers
    options['net']['beta'] = [0.1] * layers
    options['net']['reset_thres'] = [-1.0] * layers
    options['net']['k'] = [1e-6] * layers
    options['net']['gamma'] = [0.0] * layers
    options['net']['theta'] = [0.0] * layers

    op_id = 'Order2'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def set_order4(options, model_id):
    layers = options['net']['fe_layers']

    options['net']['order'] = 4
    options['net']['alpha'] = [7.8125] * layers
    options['net']['beta'] = [0.00000003125] * layers
    options['net']['reset_thres'] = [1500] * layers
    options['net']['k'] = [0.000000000000000000625] * layers
    options['net']['gamma'] = [0.000375] * layers
    options['net']['theta'] = 0.0001

    op_id = 'Order4'
    return (model_id + "_" + op_id) if len(model_id) > 0 else op_id


def run_experiments_on_given_videos(train_vids, test_vids, exp_root,
                                    lambda_ce_pair, lambda_s_zeta_s_pair,
                                    device="cpu", foa_folder=None, arch="D", stream=None, rnd=None):
    if arch == "D":
        m = [20, 20, 20, 20, 20, 20, 10]
    elif arch == "S":
        m = [20, 20, 10]
    elif arch == "DL":
        m = [32]
    basic_options = {
        "device": device,  # cpu, cuda:0, cuda:1, ...
        "seed": -1,  # if smaller than zero, current time is used
        "rho": 1.0,  # 0.1,
        "eta": 0.005,
        "batch_size": 1,
        'supervision_map': None,
        "foa": {'alpha_c': 0.1,
                'alpha_of': 0.0,  # warning: keep this to 0.0, it will avoid the computation of the optical flow!
                'alpha_fm': 0.0,  # ...
                'alpha_virtual': 0.0,
                'max_distance': 101,
                'dissipation': 0.0,
                'fps': 1,
                'w': None,
                'h': None,
                'y': None,
                'is_online': False,
                'fixation_threshold_speed': 0},
        "net": {'c': 1,
                'fe_layers': 1,
                'sem_layers': 0,
                'step_size': 0.04,  # fixed
                'prob_layers_sizes': [],
                'm': m,
                'kernel_size': [5, 5, 7] if arch == "S" else [5, 5, 5, 5, 5, 5, 7],
                'uniform_init': None,
                'lambda_c': [0.0],
                'lambda_e': [0.0],
                'lambda_s': [0.0],
                'lambda_m': [0.0],  # warning: keep this to 0.0, it will avoid the computation of the optical flow!
                'lambda_l': 0.0,
                'foa_mi': False,
                'foa_mi_use_window': False,
                'foa_coherence': False,
                'zeta_s': [1.0],
                'order': 4,
                'alpha': [7.8125],
                'beta': [0.00000003125],
                'reset_thres': [1500],
                'gamma': [0.000375],
                'k': [0.000000000000000000625],
                'theta': [0.0001],
                'training': {'layerwise': False,
                             'freeze_layers_below': False,
                             'layer_activation_frames': 1,
                             'last_active_layer': 0},
                'supervision': {'accumulate': False,
                                'foa_only': False,
                                'foa_only_options': {'max_per_class': -1, 'repetitions': -1, 'min_gap': 0}},
                'eval_info': True,
                'eval_info_reset': True
                }
    }

    # getting the pairs of (lambda_c, lambda_e) and (lambda_s, zeta_s)
    lambda_c = lambda_ce_pair[0]
    lambda_e = lambda_ce_pair[1]
    lambda_s = lambda_s_zeta_s_pair[0]
    zeta_s = lambda_s_zeta_s_pair[1]

    # creating the options for each model to consider
    model_options = collections.OrderedDict()

    for set_seed in [set_seedA]:  # single seed

        # layers 7, order 2
        for set_mi_region in [set_foa_mi, set_full_frame_mi, set_foa_window_mi]:  # 3 potentials

            opts = copy.deepcopy(basic_options)
            model_id = ''
            model_id = set_mi_region(opts, model_id)
            model_id = set_layers3(opts, model_id) if arch == "S" else set_layers7(opts, model_id)
            set_lambda_c_lambda_e(opts, lambda_c, lambda_e)
            model_id = set_order2(opts, model_id)
            model_id = set_variation_based_entropy(opts, model_id, lambda_s)  # variation-based case (lambda_s is used)
            model_id = set_seed(opts, model_id)
            model_options[model_id] = opts

            opts = copy.deepcopy(basic_options)
            model_id = ''
            model_id = set_mi_region(opts, model_id)
            model_id = set_layers3(opts, model_id) if arch == "S" else set_layers7(opts, model_id)
            set_lambda_c_lambda_e(opts, lambda_c, lambda_e)
            model_id = set_order2(opts, model_id)
            model_id = set_not_accumulated_entropy(opts, model_id)  # no average over time (lambda_s, zeta_s not used)
            model_id = set_seed(opts, model_id)
            model_options[model_id] = opts

            opts = copy.deepcopy(basic_options)
            model_id = ''
            model_id = set_mi_region(opts, model_id)
            model_id = set_layers3(opts, model_id) if arch == "S" else set_layers7(opts, model_id)
            set_lambda_c_lambda_e(opts, lambda_c, lambda_e)
            model_id = set_order2(opts, model_id)
            model_id = set_moving_average_entropy(opts, model_id, zeta_s)  # moving average (zeta_s is used)
            model_id = set_seed(opts, model_id)
            model_options[model_id] = opts

        # running the models whose options were generated so far over each video
        for ii in range(len(train_vids)):
            for model_id, options in model_options.items():
                run_train_and_test(exp_root, model_id, options, train_vids[ii], test_video=test_vids[ii],
                                   foa_folder=foa_folder, stream=stream, rnd=rnd)


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-device", "--device", dest="device", default="cuda:0",
                        help="device id")
    parser.add_argument("-l_c", "--lambda_c", dest="lambda_c", default=1000.0, type=float,
                        help="Lambda_c value")
    parser.add_argument("-l_e", "--lambda_e", dest="lambda_e", default=2000.0, type=float,
                        help="Lambda_e value")
    parser.add_argument("-l_s", "--lambda_s", dest="lambda_s", default=1000.0, type=float,
                        help="Lambda_s value")
    parser.add_argument("-z_s", "--zeta_s", dest="zeta_s", default=.01, type=float,
                        help="Zeta_s value")
    parser.add_argument("-stream", "--stream", dest="stream", default="sparsemnist",
                        choices=['sparsemnist', 'carpark', 'call'],
                        help="Video stream")
    parser.add_argument("-arch", "--arch", dest="arch", default="D", choices=['S', 'D', 'DL'],
                        help="architecture")
    parser.add_argument("-rnd", "--rnd", dest="rnd", action='store_true',
                        help="random foa scanpath flag")
    args = parser.parse_args()

    device = "{}".format(args.device)

    lambda_s_zeta_s_pairs = [(args.lambda_s, args.zeta_s)
                             ]  # this is another configuration, that would run after the previous one...

    lambda_ce_pairs = [(args.lambda_c, args.lambda_e),
                       ]  # this is another configuration, that would run after the previous one...
    if args.stream == "sparsemnist":
        stream = "mnist_toy/sparse1"
    elif args.stream == "carpark":
        stream = "carpark.mp4"
    elif args.stream == "call":
        stream = "call.mp4"
    else:
        print("Wrong video stream requsted!")
        exit()

    train_videos = ['data/{}'.format(stream)]  # insert multiple  videos here
    test_videos = ['data/{}'.format(stream)]  # insert multiple  videos here
    archs = [args.arch]  # can insert multiple architectures here , "S", "DL"
    foa_path = "rnd" if args.rnd else "regular"

    # -----------------------------------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------------------------------

    # loop on pairs (lambda_s, zeta_s)
    for _lambda_s_zeta_s_pair in lambda_s_zeta_s_pairs:
        lambda_s_zeta_s_pair_str = \
            str(_lambda_s_zeta_s_pair).replace('), ', '_').replace(', ', '-').replace('(', '').replace(')', '')

        # loop on pairs (lambda_c, lambda_e)
        for _lambda_ce_pair in lambda_ce_pairs:
            lambda_ce_pair_str = str(_lambda_ce_pair).replace('), ', '_').replace(', ', '-').replace('(', '').replace(
                ')', '')

            # loop over architectures
            for arch in archs:

                exp_root_folder = os.path.join('exp_mi_{}'.format(arch),
                                               'lambda_c_lambda_e_' + lambda_ce_pair_str +
                                               '_lambda_s_zeta_s_' + lambda_s_zeta_s_pair_str +
                                               '_arch_' + arch + '_foa_' + foa_path
                                               )

                if not os.path.exists(exp_root_folder):
                    os.makedirs(exp_root_folder)

                run_experiments_on_given_videos(train_videos, test_videos, exp_root_folder,
                                                _lambda_ce_pair, _lambda_s_zeta_s_pair,
                                                device=device, arch=arch, stream=args.stream, rnd=args.rnd)
