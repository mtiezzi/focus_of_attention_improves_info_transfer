import os
import numpy as np
import lve


class WorkerBlur(lve.Worker):

    def __init__(self, w, h, c, fps, options):
        super().__init__(w, h, c, fps, options)  # do not forget this
        self.blur_op = lve.BlurCV(self.w, self.h, self.c)
        self.rho = options["rho"]

    def process_frame(self, frame_numpy_uint8, of=None, supervisions=None):

        # blurring
        frame_numpy_uint8 = frame_numpy_uint8[0]
        frame_numpy_uint8 = self.blur_op(frame_numpy_uint8, blur_factor=1.0 - self.rho)  # it returns np.float32
        frame_numpy_uint8 = frame_numpy_uint8.astype(np.uint8)  # keeping np.uint8 format
        
        # saving output data related to the current frame
        self.add_output("blurred", frame_numpy_uint8)

    def update_model_parameters(self):

        # blurring factor update rule
        if self.rho < 1.0:
            diff_rho = 1.0 - self.rho
            self.rho = self.rho + self.options["eta"] * diff_rho  # eta: hot-changeable option
            if self.rho > 0.99:
                self.rho = 1.0

    def load(self, model_folder):
        worker_model_folder = model_folder + os.sep

        # loading parameters
        params = lve.utils.load_json(worker_model_folder + "params.json")

        # setting up the internal elements using the loaded parameters
        self.rho = params["rho"]

    def save(self, model_folder):
        worker_model_folder = model_folder + os.sep
        if not os.path.exists(worker_model_folder):
            os.makedirs(worker_model_folder)

        # saving parameters
        lve.utils.save_json(worker_model_folder + "params.json", {"rho": self.rho})

    def get_output_types(self):
        output_types = { # the output element "frames" is already registered by default
            "blurred": {"data_type": lve.OutputType.IMAGE, "per_frame": True}
        }

        return output_types

    def print_info(self):
        print("   {rho: " + str(self.rho) + ", eta: " + str(self.options["eta"]) + "}")
