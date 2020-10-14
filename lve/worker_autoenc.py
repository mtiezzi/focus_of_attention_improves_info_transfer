import os
import numpy as np
from random import randint, uniform
import lve
import torch
import torch.nn.functional as F
import cv2
import time
from collections import OrderedDict


class WorkerAutoEnc(lve.Worker):

    def __init__(self, w, h, c, fps, options):
        super().__init__(w, h, c, fps, options)  # do not forget this
        if options["device"][-1] == 'b':
            options["device"] = options["device"][0:-1]
            torch.backends.cudnn.benchmark = True
        self.device = torch.device(options["device"] if "device" in options else "cpu")  # device
        self.net_options = self.options["net"]
        self.b = self.options["batch_size"]

        # setting up seeds for random number generators
        seed = int(time.time()) if options["seed"] < 0 else int(options["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)

        # some model parameters
        self.rho = self.options["rho"]

        # processors
        self.blur = lve.BlurCV(self.w, self.h, self.c, self.device)
        self.net = lve.NetAutoEnc(self.net_options).to(self.device)
        self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.net_options["step_size"] / float(self.b))

        # misc (loss function and backward-related data)
        net_layers = self.net_options["fe_layers"]
        self.__frames = None
        self.__net_loss = None
        self.__avg_loss = [0.0] * net_layers
        self.__avg_loss_denominator = [0] * net_layers
        self.__T = [0] * net_layers
        self.__layers_stats = [None] * net_layers
        self.__first_frame = True

        # misc (data about the whole worker to print on screen or save to disk)
        self.__stats = OrderedDict([('rho', self.rho), ('loss', 0.)])

    def process_frame(self, frame, of=None, supervisions=None):

        # updating (due to last batch)
        self.b = len(frame)

        # blur, motion, focus of attention
        self.__frames, _blurred = self.__compute_sequential_ops_and_batch(frame)

        # normalization
        self.__frames = (self.__frames - 0.5) / 0.25

        # inference on the current frames
        feature_maps, patches, reconstructed_patches = self.__net_inference(self.__frames, simple_forward=False)

        # initializing previous frame data and other data structures about the current frame
        self.__initialize_data()

        # computing loss function and those elements that will be needed to complete back-prop
        forward_data = (patches, feature_maps, reconstructed_patches)
        stat_data = (self.__avg_loss, self.__avg_loss_denominator, self.__layers_stats)

        self.__net_loss = self.net.compute_loss(forward_data,
                                                stat_data)

        # saving statistics that will be printed on screen or saved to disk
        self.__stats.update({'rho': self.rho, 'loss': self.__net_loss.detach().item()})

        # saving output data related to the current frame
        for i in range(0, self.b):
            self.add_outputs({"blurred": _blurred[i],  # PNG image
                              "stats.worker": self.__stats,
                              "logs.worker": list(self.__stats.values()),  # CSV log
                              "tb.worker": self.__stats}, batch_index=i)  # tensorboard

            for l in range(0, self.net.last_active_layer + 1):
                self.add_outputs({"stats.cal." + str(l): self.__layers_stats[l],  # JSON
                                  "logs.cal." + str(l): list(self.__layers_stats[l].values()),
                                  "tb.cal" + str(l): self.__layers_stats[l]}, batch_index=i)  # tensorboard

            if self.heavy_output_data_needed:
                for l in range(0, self.net.last_active_layer + 1):
                    self.add_outputs({"filters." + str(l): self.net.get_conv_filters(l)}, batch_index=i)  # binary

                    if feature_maps[l] is not None:
                        self.add_output("probabilities." + str(l),
                                        feature_maps[l][i, None].detach().cpu().numpy(), batch_index=i)  # binary

        # eventually activating a new layer and updating data needed when processing the next frames
        self.__prepare_data_and_net_for_next_frame()

    def update_model_parameters(self):

        # updating network-related parameters
        self.net.backward(self.__net_loss)

        # update step
        self.net_optimizer.step()
        self.net.zero_grad()

        # eventually activate a new layer
        if self.net_options["training"]["layerwise"] and \
                self.__T[self.net.last_active_layer] >= self.net_options["training"]["layer_activation_frames"]:
            self.net.activate_next_layer()

    def load(self, model_folder):
        worker_model_folder = model_folder + os.sep

        # loading neural network weights
        self.net.load_state_dict(torch.load(worker_model_folder + "net.pth"))

        # loading other parameters
        params = lve.utils.load_json(worker_model_folder + "params.json")

        # setting up the internal elements using the loaded parameters
        self.rho = params["rho"]

        self.net.last_active_cal_layer = 0
        for z in range(0, params["last_active_layer"]):
            self.net.activate_next_layer()

        self.__avg_loss = params["layers_avglosses"]
        self.__avg_loss_denominator = params["layers_avglosses_denominators"]
        self.__T = params["layers_frames"]

    def save(self, model_folder):
        worker_model_folder = model_folder + os.sep
        if not os.path.exists(worker_model_folder):
            os.makedirs(worker_model_folder)

        # saving neural network weights
        torch.save(self.net.state_dict(), worker_model_folder + "net.pth")

        # saving other parameters
        lve.utils.save_json(worker_model_folder + "params.json",
                            {"rho": self.rho,
                             "last_active_layer": self.net.last_active_layer,
                             "layers_avglosses": self.__avg_loss,
                             "layers_avglosses_denominators": self.__avg_loss_denominator,
                             "layers_frames": self.__T})

    def get_output_types(self):
        output_types = {  # the output element "frames" is already registered by default
            "blurred": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "stats.worker": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "logs.worker": {'data_type': lve.OutputType.TEXT, 'per_frame': False},
            "logs.worker__header": ['frame'] + list(self.__stats.keys()),  # first line of CSV
        }

        for i in range(0, self.net_options["fe_layers"]):
            output_types.update({
                "probabilities." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True},
                "filters." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True},
                "stats.cal." + str(i): {'data_type': lve.OutputType.JSON, 'per_frame': True},
                "logs.cal." + str(i): {'data_type': lve.OutputType.TEXT, 'per_frame': False},
                "logs.cal." + str(i) + "__header": ['frame'] + list(self.net.get_layer_stat_keys())  # first line of CSV
            })

        return output_types

    def print_info(self):
        s = "   wor {" + (", ".join((k + (": {0:.3e}".format(v) if abs(v) >= 1000 else ": {0:.3f}".format(v)))
                                    for k, v in self.__stats.items())) + "}"
        fe_layers = self.net_options["fe_layers"]
        for l in range(0, self.net.last_active_layer + 1):
            if l < fe_layers:
                s += "\n   fe" + str(l) + " {"
            else:
                s += "\n   se" + str(l) + " {"

            first = True
            for k, v in self.__layers_stats[l].items():
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

    def __initialize_data(self):
        self.__net_sup_loss = 0.
        self.__net_sup_count = 0

        if self.__first_frame:
            self.__first_frame = False

    def __prepare_data_and_net_for_next_frame(self):

        # storing data
        for l in range(0, self.net.last_active_layer + 1):
            self.__avg_loss[l] = self.__layers_stats[l]['avgloss']
            self.__avg_loss_denominator[l] = self.__layers_stats[l]['avglossd']
            self.__T[l] += 1

    def __compute_sequential_ops_and_batch(self, batch_frames_np_uint8):
        batch_frames = [None] * self.b
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

            # storing references
            batch_frames[i] = frame

        frames = torch.cat(batch_frames, dim=0)

        return frames, _batch_blurred_frames_np_uint8

    def __net_inference(self, frames, simple_forward=False):
        return self.net(frames, simple_forward)

