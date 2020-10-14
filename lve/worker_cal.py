import os
import numpy as np
from random import randint, uniform, randrange
import lve
import torch
import torch.nn.functional as F
import cv2
import time
import math
from collections import OrderedDict


class WorkerCAL(lve.Worker):

    def __init__(self, w, h, c, fps, options):
        super().__init__(w, h, c, fps, options)  # do not forget this
        if options["device"][-1] == 'b':
            options["device"] = options["device"][0:-1]
            torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.deterministic = True

        self.device = torch.device(options["device"] if "device" in options else "cpu")  # device
        self.net_options = self.options["net"]
        self.b = self.options["batch_size"]

        # setting up seeds for random number generators
        seed = int(time.time()) if options["seed"] < 0 else int(options["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)

        # registering supported commands
        self.register_command("reset_foa", self.__handle_command_reset_foa)
        self.register_command("supervise", self.__handle_command_supervise)

        # setting up initial supervision map, if any
        if self.net_options['lambda_l'] > 0.0 and self.net_options['sem_layers'] > 0:
            self.augment_supervision_map(self.options["supervision_map"], self.net_options["m"][-1])

        # some model parameters
        self.rho = self.options["rho"]
        self.sup_received = {"frames": None, "targets": None, "indices": None, "mask": None}

        # processors
        self.blur = lve.BlurCV(self.w, self.h, self.c, self.device)
        self.optical_flow = lve.OpticalFlowCV()
        self.geymol = lve.GEymol(self.options["foa"], self.device) if self.options["foa"] is not None else None
        self.net = lve.NetCAL(self.net_options).to(self.device)

        lr = self.net_options["step_size"]
        if lr < 0:
            self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=-lr / float(self.b))
        else:
            self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=lr / float(self.b))

        # misc (loss function and backward-related data)
        net_layers = self.net_options["fe_layers"] + self.net_options["sem_layers"]
        self.__last_sup_frame = -self.net_options['supervision']['foa_only_options']['min_gap']
        self.__frames = None
        self.__net_loss = None
        self.__net_sup_loss = None
        self.__net_sup_count = 0
        self.__last_layer_logits = None
        self.__coherence_matrices = None
        self.__probabilities_dot = None
        self.__prev_patches = None
        self.__latest_fe = None
        self.__salience_map = None
        self.__last_sup_coords = None
        self.__last_sup_index = None
        self.__last_sup_target = None
        self.__last_sup_repetitions = 0
        self.__prev_MNs = [None] * net_layers
        self.__prev_tanh_fe_outputs = None
        self.__CA = [0.0] * net_layers
        self.__CA_denominator = [0] * net_layers
        self.__T = [0] * net_layers
        self.__cal_layers_stats = [None] * net_layers
        self.__cur_frames_idxs_to_sup_received_idxs = {}
        self.__ij_grid = None
        self.__first_frame = True
        self.__sup_allowed = False
        self.__sup_class_added = False
        self.__layer_was_activated = False
        self.__missed_supervision = False
        self.__motion_needed = (self.geymol is not None and self.geymol.parameters["alpha_of"] > 0) or \
                               (self.geymol is None and len([x for x in self.net_options["lambda_m"] if x > 0.0]) > 0)

        # misc (data about the whole worker to print on screen or save to disk)
        self.__stats = OrderedDict([('rho', self.rho), ('ca', 0.), ('ulagr', 0.), ('loss', 0.),
                                    ('foax', -1), ('foay', -1), ('foavx', -1), ('foavy', -1), ('saccade', -1)])

    def process_frame(self, frame, of=None, supervisions=None, foa=None):

        # updating (due to last batch)
        self.b = len(frame)

        # blur, motion, focus of attention
        # notice that "_saccades" below is a set of boolean flags (one-per-frame of the batch) where each flag tells if
        # the movement of the focus of attention in a frame is a saccade
        self.__frames, motions, foas, _saccades, _blurred, _foas = \
            self.__compute_sequential_ops_and_batch(frame, of, foa)

        # normalization
        self.__frames = (self.__frames - 0.5) / 0.25

        # initializing coherence data due to the previous frame
        # notice that "saccade" below is a single boolean flag that aggregates a decision on the batch
        coherence_by_motion_indices, saccade = self.__initialize_coherence_data(motions, _saccades)

        # inference on the current frames
        probabilities, patches, self.__last_layer_logits, tanh_fe_outputs = self.__net_inference(self.__frames,
                                                                                                 simple_forward=False,
                                                                                                 foa_row_col=foas,
                                                                                                 saccadic=saccade)

        # initializing previous frame data and other data structures about the current frame
        self.__initialize_basic_data(patches, tanh_fe_outputs)

        # extracting probabilities on the foa coordinates and on a window around the foa coordinates
        if self.net_options['eval_info']:
            _probabilities_foa, _probabilities_foa_window = \
                self.__restrict_full_frame_probabilities_to_foa(probabilities, foas, foa_window_too=True)
        else:
            _probabilities_foa = None

        # eventually limiting the probabilities to the focus of attention coordinates
        if self.net_options['foa_mi']:
            if not self.net_options['foa_mi_use_window']:
                if _probabilities_foa is not None:
                    _probabilities = _probabilities_foa
                else:
                    _probabilities, _ = \
                        self.__restrict_full_frame_probabilities_to_foa(probabilities, foas, foa_window_too=False)
            else:
                if _probabilities_foa_window is not None:
                    _probabilities = _probabilities_foa_window
                else:
                    _, _probabilities = \
                        self.__restrict_full_frame_probabilities_to_foa(probabilities, foas, foa_window_too=True)
        else:
            _probabilities = probabilities

        # computing terms that are needed to compute the MI and coherence loss
        p_avgs, self.__probabilities_dot = self.net.compute_probabilities_over_frame_and_time(_probabilities)

        patch_data = (patches, self.__prev_patches)
        coherence_data = (coherence_by_motion_indices, self.__prev_MNs)

        self.__coherence_matrices = self.net.compute_coherence_matrices(patch_data, coherence_data)

        # computing mutual information and coherence loss function
        forward_data = (_probabilities, p_avgs, self.__coherence_matrices)
        stat_data = self.__cal_layers_stats

        self.__net_loss = self.net.compute_mi_and_coherence_loss(forward_data, stat_data)

        # computing supervision loss function (accumulated supervision)
        if self.sup_received['frames'] is not None and self.__sup_allowed:

            # inference of the store frames (without extracting patches or computing probabilities)
            sup_frames_last_layer_logits = self.__net_inference(self.sup_received['frames'],
                                                                simple_forward=True)

            stat_data = (self.__net_sup_loss, self.__net_sup_count, self.__cal_layers_stats[-1])

            self.__net_sup_loss, self.__net_sup_count = self.net.compute_supervision_loss(sup_frames_last_layer_logits,
                                                                                          self.sup_received['targets'],
                                                                                          self.sup_received['indices'],
                                                                                          self.sup_received['mask'],
                                                                                          stat_data)

        # ensure that no supervisions are repeated in case of saccades or counting the number of repetitions
        if self.net_options['supervision']['foa_only'] and self.__last_sup_index is not None and self.__sup_allowed:
            if saccade:
                self.__forget_last_supervision()
            else:
                self.__last_sup_repetitions += 1
                if self.__last_sup_repetitions >= \
                        math.ceil(self.net_options['supervision']['foa_only_options']['repetitions'] / self.b):
                    self.__forget_last_supervision()

        # computing supervision loss function (artificially repeated supervision)
        if self.__last_sup_index is not None and self.__sup_allowed:

            # artificially generating a supervision
            repeated_supervisions = self.__repeat_last_supervision()

            # formatting supervision
            logits, targets, indices, mask, sup_frames_indices = self.__format_and_filter_supervision_data(
                repeated_supervisions, foa=None, max_per_class=None, supervision_gap=None)

            stat_data = (self.__net_sup_loss, self.__net_sup_count, self.__cal_layers_stats[-1])

            self.__net_sup_loss, self.__net_sup_count = self.net.compute_supervision_loss(logits, targets,
                                                                                          indices, mask,
                                                                                          stat_data)

        # computing supervision loss function (offline supervisions about current frames)
        if supervisions is not None and self.__sup_allowed:

            # formatting and eventually filtering supervisions
            logits, targets, indices, mask, sup_frames_indices = self.__format_and_filter_supervision_data(
                supervisions,
                foa=_foas if self.net_options['supervision']['foa_only'] else None,
                max_per_class=self.net_options['supervision']['foa_only_options']['max_per_class'],
                supervision_gap=self.net_options['supervision']['foa_only_options']['min_gap'])

            # handling the (eventually filtered) supervisions
            if sup_frames_indices is not None:
                found_targets, targets_count = torch.unique(targets, return_counts=True)
                for k in range(0, torch.numel(found_targets)):
                    self.increment_supervision_count(found_targets[k].item(), num_sup=targets_count[k].item())

                # computing loss function (offline supervisions about current frames)
                stat_data = (self.__net_sup_loss, self.__net_sup_count, self.__cal_layers_stats[-1])

                self.__net_sup_loss, self.__net_sup_count = self.net.compute_supervision_loss(logits, targets,
                                                                                              indices, mask, stat_data)

                # always accumulate supervisions, they will be cleared afterwards, if not needed
                self.__accumulate_supervisions(targets, indices, mask, sup_frames_indices=sup_frames_indices)

                # saving the coordinates of the first supervision of the last frame of the batch
                if self.net_options['supervision']['foa_only']:
                    sup_index = indices[-1, 0].item()
                    sup_target = targets[-1, 0]

                    if sup_index != self.__last_sup_index:
                        self.__remember_last_supervision(index=sup_index, target=sup_target)
                        _saccades = [False] * self.b

        # computing the value of the Lagrangian (only for statistical purposes)
        stat_data = (self.__CA, self.__CA_denominator, self.__T, self.__cal_layers_stats)
        loss_detached = self.__net_loss.detach().item() + self.__net_sup_loss.detach().item()

        total_CA, total_unscaled_lagrangian = self.net.compute_ca(stat_data)

        # saving statistics that will be printed on screen or saved to disk
        self.__stats.update({'rho': self.rho,
                             'ca': total_CA,
                             'ulagr': total_unscaled_lagrangian,
                             'loss': loss_detached})

        # saving debug/eval info
        if self.net_options['eval_info']:
            self.net.eval_and_accumulate_mi_and_coherence(probabilities,
                                                          _probabilities_foa,
                                                          _probabilities_foa_window,
                                                          self.__prev_tanh_fe_outputs, tanh_fe_outputs,
                                                          coherence_by_motion_indices,
                                                          self.__cal_layers_stats,
                                                          skip_coher=saccade)

            if supervisions is not None and self.__sup_allowed:
                # formatting supervisions (no filtering)
                logits, targets, indices, mask, sup_frames_indices = self.__format_and_filter_supervision_data(
                    supervisions, foa=None, max_per_class=None, supervision_gap=None)

                self.net.eval_supervision_loss(logits, targets, indices, mask, self.__cal_layers_stats[-1])

        # saving output data related to the current frame
        if self.heavy_output_data_needed and self.__last_layer_logits is not None:
            last_layer_probabilities = torch.softmax(self.__last_layer_logits, dim=1)

        for i in range(0, self.b):
            self.__stats.update({"foax": _foas[i][0], "foay": _foas[i][1], "foavx": _foas[i][2], "foavy": _foas[i][3],
                                 "saccade": int(saccade)})

            self.add_outputs({"motion": of[i],  # binary
                              "blurred": _blurred[i],  # PNG image
                              "stats.worker": self.__stats,
                              "logs.worker": list(self.__stats.values()),  # CSV log
                              "tb.worker": self.__stats}, batch_index=i)  # tensorboard

            for l in range(0, self.net.last_active_cal_layer + 1):
                self.add_outputs({"stats.cal." + str(l): self.__cal_layers_stats[l],  # JSON
                                  "logs.cal." + str(l): list(self.__cal_layers_stats[l].values()),
                                  "tb.cal" + str(l): self.__cal_layers_stats[l]}, batch_index=i)  # tensorboard

            if self.heavy_output_data_needed:
                for l in range(0, self.net.last_active_cal_layer + 1):
                    self.add_outputs({"filters." + str(l): self.net.get_conv_filters(l)}, batch_index=i)  # binary

                    if probabilities[l] is not None:
                        self.add_output("probabilities." + str(l),
                                        probabilities[l][i, None].detach().cpu().numpy(), batch_index=i)  # binary

                    if self.__last_layer_logits is not None:
                        self.add_output("predictions",
                                        last_layer_probabilities[i, None].detach().cpu().numpy(), batch_index=i)  # bin

                    self.__add_output_about_supervisions()

        # eventually activating a new layer and updating data needed when processing the next frames
        self.__prepare_data_and_net_for_next_frame(patches, tanh_fe_outputs)

    def update_model_parameters(self):

        # packing
        extra_backward_data = (self.__coherence_matrices, self.__probabilities_dot)

        # updating network-related parameters
        full_loss = self.__net_loss + self.__net_sup_loss

        self.net.backward(full_loss, extra_backward_data)

        # update step
        self.net_optimizer.step()
        self.net.zero_grad()

        # eventually activate a new layer
        if self.net_options["training"]["layerwise"] and \
                self.net.last_active_cal_layer < self.net_options["fe_layers"] + self.net_options["sem_layers"] -1 and \
                self.__T[self.net.last_active_cal_layer] >= self.net_options["training"]["layer_activation_frames"]:
            self.net.activate_next_cal_layer()
            self.__layer_was_activated = True

    def load(self, model_folder):
        worker_model_folder = model_folder + os.sep

        # loading neural network weights
        self.net.load_state_dict(torch.load(worker_model_folder + "net.pth", map_location=self.device))

        # loading other parameters
        params = lve.utils.load_json(worker_model_folder + "params.json")

        # setting up the internal elements using the loaded parameters
        self.rho = params["rho"]
        if self.geymol is not None:
            self.geymol.reset(params["foa_y"], params["foa_t"])
            self.geymol.first_call = False

        self.net.last_active_cal_layer = 0
        for z in range(0, params["last_active_cal_layer"]):
            self.net.activate_next_cal_layer()

        self.__CA = params["cal_layers_CAs"]
        self.__CA_denominator = params["cal_layers_CAs_denominators"]
        self.__T = params["cal_layers_frames"]
        self.__last_sup_frame = params["last_sup_frame"]

        if self.net_options['lambda_l'] > 0.0 and self.net_options['sem_layers'] > 0:
            self.augment_supervision_map(params["supervision_map"],
                                         self.net_options["m"][-1],
                                         counts=params["supervision_count"])

        if self.options["supervision_map"] is not None and len(self.options["supervision_map"]) > 0:
            print("WARNING: the provided supervision map will be overwritten by the one loaded from disk!")

        # loading supervised frames
        self.sup_received = torch.load(worker_model_folder + "received_supervisions.pth")

        for k, v in self.sup_received.items():
            if isinstance(v, torch.Tensor):
                self.sup_received[k] = v.to(self.device)

        # loading latest features
        if os.path.exists(worker_model_folder + "latest_fe.pth"):
            self.__latest_fe = torch.load(worker_model_folder + "latest_fe.pth", map_location=self.device)

        # loading foa salience map
        if os.path.exists(worker_model_folder + "salience_map.pth"):
            self.__salience_map = torch.load(worker_model_folder + "salience_map.pth", map_location=self.device)

        # eventually clearing the accumulated statistics
        if self.net_options['eval_info'] and self.net_options['eval_info_reset']:
            self.net.force_reset_eval_avgs()

    def save(self, model_folder):
        worker_model_folder = model_folder + os.sep
        if not os.path.exists(worker_model_folder):
            os.makedirs(worker_model_folder)

        # saving neural network weights
        torch.save(self.net.state_dict(), worker_model_folder + "net.pth")

        # saving other parameters
        lve.utils.save_json(worker_model_folder + "params.json",
                            {"rho": self.rho,
                             "foa_y": list(self.geymol.y) if self.geymol is not None else [0., 0., 0., 0.],
                             "foa_t": self.geymol.t if self.geymol is not None else 0,
                             "last_active_cal_layer": self.net.last_active_cal_layer,
                             "cal_layers_CAs": self.__CA,
                             "cal_layers_CAs_denominators": self.__CA_denominator,
                             "cal_layers_frames": self.__T,
                             "last_sup_frame": self.__last_sup_frame,
                             "supervision_map": self.get_supervision_map(),
                             "supervision_count": self.get_supervision_count()})

        # saving received supervisions
        torch.save(self.sup_received, worker_model_folder + "received_supervisions.pth")

        # saving latest features
        if self.__latest_fe is not None:
            torch.save(self.__latest_fe, worker_model_folder + "latest_fe.pth")
            torch.save(self.__latest_fe[:, self.net.invariant_fe_mask], worker_model_folder + "latest_fe_invariant.pth")

        # saving latest foa salience map
        if self.__salience_map is not None:
            torch.save(self.__salience_map, worker_model_folder + "salience_map.pth")
            cv2.imwrite(worker_model_folder + "salience_map_nonzero.png",
                        ((self.__salience_map / self.__salience_map) * 255.0).cpu().to(torch.uint8).numpy())
            cv2.imwrite(worker_model_folder + "salience_map.png",
                        ((self.__salience_map / torch.max(self.__salience_map)) * 255.0).cpu().to(torch.uint8).numpy())

    def get_output_types(self):
        output_types = {  # the output element "frames" is already registered by default
            "blurred": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "stats.worker": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "logs.worker": {'data_type': lve.OutputType.TEXT, 'per_frame': False},
            "logs.worker__header": ['frame'] + list(self.__stats.keys()),  # first line of CSV
            "sup": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "sup.indices": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup.targets": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup.map": {'data_type': lve.OutputType.JSON, 'per_frame': False}
        }

        for i in range(0, self.net_options["fe_layers"] + self.net_options["sem_layers"]):
            output_types.update({
                "probabilities." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True},
                "filters." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True},
                "stats.cal." + str(i): {'data_type': lve.OutputType.JSON, 'per_frame': True},
                "logs.cal." + str(i): {'data_type': lve.OutputType.TEXT, 'per_frame': False},
                "logs.cal." + str(i) + "__header": ['frame'] + list(self.net.get_layer_stat_keys())  # first line of CSV
            })

        if self.net_options["sem_layers"] > 0:
            output_types.update({"predictions": {'data_type': lve.OutputType.BINARY, 'per_frame': True}})

        return output_types

    def print_info(self):
        s = "   wor {" + (", ".join((k + (": {0:.3e}".format(v) if abs(v) >= 1000 else ": {0:.3f}".format(v)))
                                    for k, v in self.__stats.items())) + "}"
        fe_layers = self.net_options["fe_layers"]
        for l in range(0, self.net.last_active_cal_layer + 1):
            if l < fe_layers:
                s += "\n   fe" + str(l) + " {"
            else:
                s += "\n   se" + str(l) + " {"

            first = True
            for k, v in self.__cal_layers_stats[l].items():
                if v < -1.0001 or v > -0.999:
                    if not first:
                        s += ", "
                    if abs(v) < 1000:
                        s += k + (": {0:.3f}".format(v))
                    else:
                        s += k + (": {0:.3e}".format(v))
                    first = False
            s += "}"
        print(s)

    def __handle_command_reset_foa(self, command_value, batch_index=0):
        if batch_index:
            pass
        if self.geymol is not None:
            self.geymol.reset([command_value['y'], command_value['x'],
                               2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1)),
                               2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))])
            return True
        else:
            return False

    def __handle_command_supervise(self, command_value, batch_index=0):
        if self.net_options['lambda_l'] <= 0.0 or self.net_options['sem_layers'] <= 0:
            return False  # nothing to do with a supervision

        class_name = command_value['class'].lower().strip()
        row = command_value['y']
        col = command_value['x']
        n_cols = command_value['w']
        n_rows = command_value['h']
        n_sup = n_cols * n_rows

        target, new_class_added = self.get_target(class_name, self.net_options["m"][-1])
        if target is None:
            return False  # fail

        self.__sup_class_added = self.__sup_class_added or new_class_added

        self.increment_supervision_count(target, num_sup=n_sup)

        # preparing supervision data
        mask = torch.ones((1, n_sup), dtype=torch.float32, device=self.device)

        targets = torch.ones((1, n_sup), dtype=torch.long, device=self.device)
        targets = targets * target

        _indices = [None] * n_rows
        for r in range(0, n_rows):
            _indices[r] = np.arange(n_cols, dtype=np.long) + (row + r) * self.w + col
        _indices = np.concatenate(_indices)
        indices = torch.from_numpy(_indices).view(1, n_sup).to(self.device)

        # getting the frame to supervise
        sup_frame_index = torch.tensor(batch_index, dtype=torch.long, device=self.device).view(1)
        last_layer_logits = torch.index_select(self.__last_layer_logits, 0, sup_frame_index)

        # augmenting the loss function
        stat_data = (self.__net_sup_loss, self.__net_sup_count, self.__cal_layers_stats[-1])

        self.__net_sup_loss, self.__net_sup_count = self.net.compute_supervision_loss(last_layer_logits,
                                                                                      targets, indices, mask,
                                                                                      stat_data)

        if self.net_options['supervision']['accumulate_sup']:
            self.__accumulate_supervisions(targets, indices, mask, sup_frames_indices=sup_frame_index)

        if self.heavy_output_data_needed:
            self.__add_output_about_supervisions()
        return True

    def __initialize_basic_data(self, first_frame_patches, first_frame_tanh_fe_outputs):
        self.__sup_allowed = self.net_options['lambda_l'] > 0.0 and \
                             self.net_options['sem_layers'] > 0 and \
                             self.net.last_active_cal_layer == self.net_options['sem_layers'] + self.net_options[
                                 'fe_layers'] - 1
        self.__cur_frames_idxs_to_sup_received_idxs = {}
        self.__sup_class_added = False
        self.__net_sup_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
        self.__net_sup_count = 0

        if self.__first_frame:
            self.__first_frame = False
            self.__layer_was_activated = True
            self.__sup_class_added = True  # this will force a save operation of the class map

            if not self.net_options["foa_coherence"]:
                self.__ij_grid = lve.nn.create_image_plane_coords(self.b, self.h, self.w, device=self.device)

            layers = len(self.__cal_layers_stats)
            self.__prev_patches = [None] * layers
            self.__prev_tanh_fe_outputs = [None] * layers

            if self.geymol is not None and self.net.last_active_cal_layer >= self.net_options["fe_layers"] - 1:
                if self.__latest_fe is None:
                    with torch.no_grad():
                        _, _, _, fe_outputs = self.net(self.__frames, simple_forward=False)
                        self.__latest_fe = fe_outputs
                        self.__latest_fe = torch.cat(fe_outputs[0:self.net_options["fe_layers"]], dim=1)[self.b - 1, :,
                                           :, :].detach()
                        self.__latest_fe = self.__latest_fe.view(self.net.total_fe_m, self.h * self.w).t().contiguous()

                if self.__salience_map is None:
                    self.__salience_map = torch.zeros((self.h, self.w), dtype=torch.float, device=self.device)

        if self.__layer_was_activated:
            self.__layer_was_activated = False

            for l in range(0, self.net.last_active_cal_layer + 1):
                if first_frame_patches[l] is not None:
                    self.__prev_patches[l] = first_frame_patches[l][-1, None].detach()
                    self.__prev_tanh_fe_outputs[l] = first_frame_tanh_fe_outputs[l][-1, None].detach()
                else:
                    self.__prev_patches[l] = None
                    self.__prev_tanh_fe_outputs[l] = None

    def __initialize_coherence_data(self, motions, saccades):
        if not self.net_options["foa_coherence"] and self.__motion_needed:
            ij_grid_prev = lve.nn.coher_indices_motion(self.__ij_grid, motions)

            # TODO
            # pos = (ij_grid_prev == self.__last_sup_index).nonzero()
            # if pos.size() > 0:
            #     self.__last_sup_index = pos[0].item()

            return ij_grid_prev, any(saccades)
        else:
            return None, any(saccades)

    def __prepare_data_and_net_for_next_frame(self, patches, tanh_fe_outputs):

        # clearing accumulated supervisions, if needed
        if not self.net_options['supervision']['accumulate']:
            self.sup_received = {"frames": None, "targets": None, "indices": None, "mask": None}

        # storing data
        for l in range(0, self.net.last_active_cal_layer + 1):
            self.__prev_patches[l] = patches[l][-1, None].detach() if patches[l] is not None else None
            self.__prev_MNs[l] = self.__coherence_matrices[l][-1] if self.__coherence_matrices[l] is not None else None
            self.__CA[l] = self.__cal_layers_stats[l]['ca']
            self.__CA_denominator[l] = self.__cal_layers_stats[l]['cad']
            self.__T[l] += 1
            self.__prev_tanh_fe_outputs[l] = \
                tanh_fe_outputs[l][-1, None].detach() if tanh_fe_outputs[l] is not None else None

    def __compute_sequential_ops_and_batch(self, batch_frames_np_uint8, batch_motion_np_float32, batch_foa_np_float32):
        batch_frames = [None] * self.b
        batch_motions = [None] * self.b
        batch_foas = [None] * self.b
        _batch_saccades = [None] * self.b
        _batch_foa_np_float64 = [None] * self.b
        _batch_blurred_frames_np_uint8 = [None] * self.b

        # sequential operations on the batched data
        for i in range(0, self.b):

            # blurring factor
            if self.rho < 1.0 and (not self.__first_frame or i > 0):
                diff_rho = 1.0 - self.rho
                self.rho = self.rho + self.options["eta"] * diff_rho  # eta: hot-changeable option
                if self.rho > 0.99:
                    self.rho = 1.0

            # blurring
            frame_np_uint8 = self.blur(batch_frames_np_uint8[i], blur_factor=1.0 - self.rho).astype(np.uint8)
            frame = lve.utils.np_uint8_to_torch_float_01(frame_np_uint8, device=self.device)
            _batch_blurred_frames_np_uint8[i] = frame_np_uint8

            # grayscale-instance of the (blurred) input frame
            if not self.frame_is_gray_scale:
                frame_gray_np_uint8 = cv2.cvtColor(frame_np_uint8, cv2.COLOR_BGR2GRAY).reshape(self.h, self.w, 1)
                frame_gray = lve.utils.np_uint8_to_torch_float_01(frame_gray_np_uint8, device=self.device)
            else:
                frame_gray_np_uint8 = frame_np_uint8
                frame_gray = frame

            # optical flow
            if batch_motion_np_float32 is None or batch_motion_np_float32[i] is None:
                if self.__motion_needed:
                    motion_np_float32 = self.optical_flow(frame_gray_np_uint8)  # it returns np.float32, h x w x 2
                else:
                    motion_np_float32 = np.zeros((self.h, self.w, 2), dtype=np.float32)
                motion = lve.utils.np_float32_to_torch_float(motion_np_float32, device=self.device)  # 1 x 2 x h x w

                if batch_motion_np_float32 is not None:
                    batch_motion_np_float32[i] = motion_np_float32  # updating
            else:
                motion_np_float32 = batch_motion_np_float32[i]  # h x w x 2
                motion = lve.utils.np_float32_to_torch_float(motion_np_float32, device=self.device)  # 1 x 2 x h x w

            # focus of attention
            if batch_foa_np_float32 is None or batch_foa_np_float32[i] is None:
                if self.geymol is not None:
                    if self.options["foa"]["dissipation"] >= 0.0:
                        foa, saccade = self.geymol.next_location(frame_gray, motion,
                                                                 frame_gray_uint8_cpu=frame_gray_np_uint8,
                                                                 virtualmass_xy=self.__last_sup_coords,
                                                                 virtualmass_vxvy=[0., 0.]
                                                                 if self.__last_sup_repetitions == 1 else None)
                    else:
                        foa = np.array([randrange(0, self.h-1), randrange(0, self.w-1), 0., 0.], dtype=np.float64)
                        saccade = False
                else:
                    foa = np.array([0., 0., 0., 0.], dtype=np.float64)
                    saccade = False
            else:
                foa = batch_foa_np_float32[i][0:4]
                saccade = bool(batch_foa_np_float32[i][-1])

            # storing references
            batch_frames[i] = frame
            batch_motions[i] = motion
            batch_foas[i] = torch.from_numpy(foa[0:2].astype(np.long)).to(self.device).view(1, 2)
            _batch_saccades[i] = saccade
            _batch_foa_np_float64[i] = foa

        frames = torch.cat(batch_frames, dim=0)
        motions = torch.cat(batch_motions, dim=0)
        foas = torch.cat(batch_foas, dim=0)

        return frames, motions, foas, _batch_saccades, _batch_blurred_frames_np_uint8, _batch_foa_np_float64

    def __net_inference(self, frames, foa_row_col=None, simple_forward=False, saccadic=False):
        if not simple_forward:
            if not self.net_options["foa_coherence"]:
                out = self.net(frames, simple_forward=False)
            else:
                # reset on saccadic movements
                if saccadic and self.net_options['foa_coherence']:
                    self.net.force_reset()

                out = self.net(frames, simple_forward=False, patch_row_col=foa_row_col)

                if foa_row_col is not None:
                    with torch.no_grad():
                        if self.__latest_fe is not None:
                                b = self.b - 1
                                self.__latest_fe[foa_row_col[b,0] * self.w + foa_row_col[b,1], :] = \
                                    torch.cat(out[3][0:self.net_options["fe_layers"]], dim=1)[b, :, 0, 0].detach()

                        if self.__salience_map is not None:
                            indices = (foa_row_col[:,0] * self.w + foa_row_col[:,1]).to(torch.long)
                            self.__salience_map.view(-1)[indices] += 1

            return out
        else:
            out = self.net(frames, simple_forward=True)
            return out

    def __restrict_full_frame_probabilities_to_foa(self, probabilities, foas, foa_window_too=False):
        _probabilities = [None] * len(probabilities)
        _probabilities_w = [None] * len(probabilities)
        foa_indices = (foas[:, 1] + foas[:, 0] * self.w).to(torch.long)
        for i in range(0, len(probabilities)):
            if probabilities[i] is not None:
                m = probabilities[i].shape[1]

                # foa
                _probabilities[i] = torch.gather(probabilities[i].view(self.b, m, self.w * self.h),
                                                 2, foa_indices.unsqueeze(1).unsqueeze(1).repeat(1, m, 1))\
                    .view(self.b, m, 1, 1)

                # foa window
                if foa_window_too:
                    crop_size = int(max(self.w, self.h) * 0.15)
                    k = crop_size // 2
                    padded_probabilities = F.pad(probabilities[i], pad=(k, k, k, k, 0, 0, 0, 0), value=1.0/m)

                    x_cord = torch.arange(crop_size, device=self.device).repeat(crop_size).view(crop_size, crop_size)
                    y_cord = x_cord.t().contiguous()
                    x_cord = x_cord.view(-1).unsqueeze(0).repeat(self.b, 1) + foas[:, 1]
                    y_cord = y_cord.view(-1).unsqueeze(0).repeat(self.b, 1) + foas[:, 0]
                    indices = (y_cord * self.w + x_cord).view(self.b, 1, crop_size * crop_size).repeat(1, m, 1)

                    wh_padded = padded_probabilities.shape[2] * padded_probabilities.shape[3]

                    _probabilities_w[i] = torch.gather(padded_probabilities.view(self.b, m, wh_padded), 2, indices)\
                        .view(self.b, m, crop_size, crop_size)

        return _probabilities, _probabilities_w

    def __forget_last_supervision(self):
        self.__last_sup_index = None
        self.__last_sup_coords = None
        self.__last_sup_target = None
        self.__last_sup_repetitions = 0

    def __remember_last_supervision(self, index=None, target=None):
        self.__last_sup_target = target
        self.__last_sup_index = index
        self.__last_sup_coords = [index // self.w, index - (index // self.w) * self.w]
        self.__last_sup_repetitions = 0

    def __format_and_filter_supervision_data(self, supervisions, foa=None, max_per_class=None, supervision_gap=None):
        max_supervised_pixels = 0

        targets = [None] * self.b
        indices = [None] * self.b
        masks = [None] * self.b

        # these are used to keep track of the supervised frames and on how they were supervised
        sup_frames_indices = torch.ones(self.b, dtype=torch.long, device=self.device)
        num_sup_frames = self.b

        # formatting and collecting targets, indices, mask
        k = 0
        for i in range(0, self.b):

            # if supervision should only be taken in the focus-of-attention coordinates,
            # then we filter out the provided supervision tuple (also considering the max_per_class attribute)
            # warning: max_per_class is guaranteed to work only if mini-batch size is 1
            # (otherwise the system can go a bit beyond max_per_class)
            supervision_tuple = self.__filter_supervisions(supervisions[i],
                                                           foa[i] if foa is not None else None,
                                                           max_per_class,
                                                           supervision_gap)

            # handling (the eventually filtered) supervision
            if supervision_tuple is not None:
                _targets, _indices = supervision_tuple  # _indices can be None
                _indices = np.arange(0, self.w * self.h, dtype=np.long) if _indices is None else _indices
                _mask = np.ones(_indices.shape, dtype=np.float32)

                nsup = _targets.size
                max_supervised_pixels = max(max_supervised_pixels, nsup)

                targets[k] = _targets.astype(np.long)
                indices[k] = _indices.astype(np.long)
                masks[k] = _mask
                sup_frames_indices[k] = i
                k += 1
            else:
                num_sup_frames -= 1

        if num_sup_frames == 0:
            return None, None, None, None, None

        # excluding unsupervised frames
        if num_sup_frames < self.b:
            sup_frames_indices = sup_frames_indices[0:num_sup_frames]
            last_layer_logits = torch.index_select(self.__last_layer_logits, 0, sup_frames_indices)
            targets = targets[0:num_sup_frames]
            indices = indices[0:num_sup_frames]
            masks = masks[0:num_sup_frames]
        else:
            last_layer_logits = self.__last_layer_logits

        # zero-padding
        if max_supervised_pixels > 0:
            for k in range(0, num_sup_frames):
                if len(targets[k] < max_supervised_pixels):  # this condition is never true in fully-sup frames
                    targets[k].resize(max_supervised_pixels, refcheck=False)
                    indices[k].resize(max_supervised_pixels, refcheck=False)
                    masks[k].resize(max_supervised_pixels, refcheck=False)

        # stacking
        targets = np.stack(targets, axis=0)
        indices = np.stack(indices, axis=0)
        mask = np.stack(masks, axis=0)

        # moving to torch
        targets = torch.from_numpy(targets).to(torch.long).to(self.device)
        mask = torch.from_numpy(mask).to(self.device)
        indices = torch.from_numpy(indices).to(torch.long).to(self.device)

        return last_layer_logits, targets, indices, mask, sup_frames_indices

    def __filter_supervisions(self, supervision_tuple, foa, max_per_class, supervision_gap):
        if foa is not None:
            foa_index = int(foa[0]) * self.w + int(foa[1])
            targets, indices = supervision_tuple

            # searching for the focus-of-attention coordinates in the supervised indices, and filtering the
            # supervision signal (in this case, if max_per_class is provided, supervision is discarded if
            # the target class has already received the max number of supervisions)
            found_foa = False

            if indices is None:
                found_foa = True

                if max_per_class is not None and max_per_class >= 0:
                    class_name = self.get_class_name(targets[int(foa[0]), int(foa[1])])
                    if class_name is not None:
                        sup_count = self.get_supervision_count()[self.get_class_name(targets[int(foa[0]), int(foa[1])])]
                    else:
                        sup_count = 0
                    if sup_count >= max_per_class:
                        found_foa = False

                if found_foa:
                    indices = np.array([foa_index], dtype=targets.dtype)
                    targets = np.array([targets[int(foa[0]), int(foa[1])]], dtype=targets.dtype)
            else:
                z_foa_index = -1

                for z in range(0, indices.size):
                    if indices[z] == foa_index:
                        z_foa_index = z
                        found_foa = True
                        break

                if found_foa:
                    if max_per_class is not None and max_per_class >= 0:
                        class_name = self.get_class_name(targets[z_foa_index])
                        if class_name is not None:
                            sup_count = self.get_supervision_count()[
                                self.get_class_name(targets[z_foa_index])]
                        else:
                            sup_count = 0
                        if sup_count >= max_per_class:
                            found_foa = False

                if found_foa:
                    indices = np.array([foa_index], dtype=indices.dtype)
                    targets = np.array([targets[z_foa_index]], dtype=targets.dtype)

            if not found_foa:
                supervision_tuple = None
            else:
                supervision_tuple = (targets, indices)  # filtered supervision

                if supervision_gap is not None and supervision_gap > 0:
                    if (self.__T[-1] - self.__last_sup_frame) >= supervision_gap:
                        self.__last_sup_frame = self.__T[-1]
                    else:
                        supervision_tuple = None

        return supervision_tuple

    def __repeat_last_supervision(self):
        supervisions = [None] * self.b

        for i in range(0, self.b):
            targets = np.array([self.__last_sup_target], dtype=np.long)
            indices = np.array([self.__last_sup_index], dtype=np.long)
            supervisions[i] = (targets, indices)

        return supervisions

    def __accumulate_supervisions(self, targets, indices, mask, sup_frames_indices=None):
        sf = self.sup_received['frames']
        si = self.sup_received['indices']
        st = self.sup_received['targets']
        sm = self.sup_received['mask']

        if sup_frames_indices is not None:
            b = sup_frames_indices.numel()
            frames = torch.index_select(self.__frames, 0, sup_frames_indices)
        else:
            b = self.b
            frames = self.__frames

        number_of_new_supervisions = torch.sum(mask, dim=1).cpu().to(torch.int).numpy()

        some_frames_of_the_current_batch_were_already_supervised = False
        for i in range(0, b):
            z = sup_frames_indices[i].item()
            if z in self.__cur_frames_idxs_to_sup_received_idxs:
                some_frames_of_the_current_batch_were_already_supervised = True

        if np.min(number_of_new_supervisions) > 0 and not some_frames_of_the_current_batch_were_already_supervised:

            # if all the frames of the batch are supervised, we can quickly add them to the supervision buffer
            if sf is None:

                # if these are the first supervisions we get, directly put them in the supervision buffer
                for i in range(0, b):
                    z = sup_frames_indices[i].item() if sup_frames_indices is not None else i
                    self.__cur_frames_idxs_to_sup_received_idxs[z] = i

                self.sup_received['frames'] = frames
                self.sup_received['indices'] = indices
                self.sup_received['targets'] = targets
                self.sup_received['mask'] = mask
            else:

                # if there are already supervisions in the buffer and the new supervisions are not about the already
                # buffered frames, then directly append the new supervisions to the buffer
                for i in range(0, b):
                    z = sup_frames_indices[i].item() if sup_frames_indices is not None else i
                    self.__cur_frames_idxs_to_sup_received_idxs[z] = len(sf) + i

                gap = int(sm.shape[1] - mask.shape[1])
                if gap < 0:
                    gap = -gap
                    sm = F.pad(sm, (0, gap))
                    si = F.pad(si, (0, gap))
                    st = F.pad(st, (0, gap))
                elif gap > 0:
                    indices = F.pad(indices, (0, gap))
                    targets = F.pad(targets, (0, gap))
                    mask = F.pad(mask, (0, gap))

                self.sup_received['frames'] = torch.cat((sf, frames), dim=0)
                self.sup_received['indices'] = torch.cat((si, indices), dim=0)
                self.sup_received['targets'] = torch.cat((st, targets), dim=0)
                self.sup_received['mask'] = torch.cat((sm, mask), dim=0)
        else:

            # if not all the frames of the batch are supervised or if some new supervisions are about already buffered
            # frames, then we go through the new supervisions one-by-one, and we decide what to do
            k = 0
            offset = len(sf) if sf is not None else 0

            for i in range(0, b):

                # obviously we only consider frames of the current batch that are supervised
                if number_of_new_supervisions[i] > 0:
                    z = sup_frames_indices[i].item() if sup_frames_indices is not None else i
                    if z in self.__cur_frames_idxs_to_sup_received_idxs:

                        # if we are supervising a frame that was already in the buffer, we append the new supervisions
                        # one-by-one, since sometimes we will have to pad tensors, sometimes there will be already room
                        # for the supervisions
                        sup_received_index = self.__cur_frames_idxs_to_sup_received_idxs[z]

                        for u in range(0, number_of_new_supervisions[i]):
                            cur_ns = int(torch.sum(sm[sup_received_index, :]).item())
                            max_ns = sm.shape[1]

                            if cur_ns == max_ns:
                                self.sup_received['indices'] = F.pad(si, (0, 1))
                                self.sup_received['targets'] = F.pad(st, (0, 1))
                                self.sup_received['mask'] = F.pad(sm, (0, 1))
                                si = self.sup_received['indices']
                                st = self.sup_received['targets']
                                sm = self.sup_received['mask']

                                self.sup_received['indices'][sup_received_index, cur_ns] = indices[i][u]
                            self.sup_received['targets'][sup_received_index, cur_ns] = targets[i][u]
                            self.sup_received['mask'][sup_received_index, cur_ns] = 1.0
                    else:

                        # if we are supervising a never-supervised-before frame, than we include all the supervisions
                        # in the buffer (either the buffer or the new supervisions might need to be padded/masked)
                        self.__cur_frames_idxs_to_sup_received_idxs[z] = offset + k
                        k += 1

                        if sf is None:
                            self.sup_received['frames'] = frames[i, None]
                            self.sup_received['indices'] = indices[i, None]
                            self.sup_received['targets'] = targets[i, None]
                            self.sup_received['mask'] = mask[i, None]
                        else:
                            gap = sm.shape[1] - mask.shape[1]
                            if gap < 0:
                                gap = -gap
                                sm = F.pad(sm, (0, gap))
                                si = F.pad(si, (0, gap))
                                st = F.pad(st, (0, gap))
                            elif gap > 0:
                                indices = F.pad(indices, (0, gap))
                                targets = F.pad(targets, (0, gap))
                                mask = F.pad(mask, (0, gap))

                            self.sup_received['frames'] = torch.cat((sf, frames[i, None]), dim=0)
                            self.sup_received['indices'] = torch.cat((si, indices[i, None]), dim=0)
                            self.sup_received['targets'] = torch.cat((st, targets[i, None]), dim=0)
                            self.sup_received['mask'] = torch.cat((sm, mask[i, None]), dim=0)

                        sf = self.sup_received['frames']
                        si = self.sup_received['indices']
                        st = self.sup_received['targets']
                        sm = self.sup_received['mask']

    def __add_output_about_supervisions(self):
        if self.__sup_class_added:
            self.add_output("sup.map", self.get_supervision_map())  # JSON

        for i in range(0, self.b):
            if i in self.__cur_frames_idxs_to_sup_received_idxs:
                k = self.__cur_frames_idxs_to_sup_received_idxs[i]
                mask = self.sup_received["mask"][k, :]
                indices = self.sup_received["indices"][k, mask == 1.0].cpu().numpy().astype(np.int32)  # cast needed!
                targets = self.sup_received["targets"][k, mask == 1.0].cpu().numpy().astype(np.int32)  # cast needed!

                self.add_outputs({"sup.targets": targets, "sup.indices": indices}, batch_index=i)  # binary

    def __convert_binary_supervision_to_readable_list(self, targets, indices, mask=None):
        nsup = indices.size
        list_of_sup_dictionaries = [None] * nsup
        for j in range(0, nsup):
            if mask is None or mask[j] > 0.0:
                row = int(indices[j] // self.w)
                col = int(indices[j] - row * self.w)
                list_of_sup_dictionaries[j] = {"class": self.get_class_name(targets[j]),
                                               "row": row, "col": col}

    def __convert_list_of_class_names_to_targets(self, list_of_class_names):
        max_num_classes = self.net_options["m"][-1]
        nsup = len(list_of_class_names)
        targets = np.zeros(nsup, dtype=np.long)
        for j in range(0, len(list_of_class_names)):
            target, new_class_added = self.get_target(list_of_class_names[j], max_num_classes)
            if target is not None:
                targets[j] = target

        return targets
