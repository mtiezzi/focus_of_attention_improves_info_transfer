import numpy as np
import cv2
from math import isnan
from random import randint, uniform
from scipy.integrate import odeint
import time
import os


class GEymolCV:

    def __init__(self, parameters, device=None):
        self.parameters = parameters
        self.h = parameters['h']
        self.w = parameters['w']
        self.fixation_threshold_speed = parameters["fixation_threshold_speed"]
        self.t = 0
        self.y = GEymolCV.__generate_initial_conditions(self.h, self.w)  # y = [x (row), y (col), velocity_x, velocity_y]
        self.is_online = parameters['is_online'] if 'is_online' in parameters else False
        self.saccades_per_second = float(1.0)  # it must be float
        self.real_time_last_saccade = time.clock()
        self.first_call = False

        if parameters['max_distance'] <= 0 or parameters['max_distance'] % 2 == 0:
            raise ValueError("Invalid filter size, it must be odd! (max_distance)")

        # generating the distance matrix
        self.gravitational_filter = GEymolCV.__create_gravitational_filter(parameters['max_distance'])

        # matrix to mark pixels to which inhibit return
        self.IOR_matrix = np.zeros((self.h, self.w), dtype=np.float32)

        # precomputed IOR mask
        self.centered_gaussian_2d_approx = \
            self.__generate_approximation_of_2d_gaussian([self.h // 2, self.w // 2],
                                                         ray=5,
                                                         blur=self.parameters['max_distance'])

        # face detector and related map
        base_path = os.path.dirname(cv2.__file__) + os.sep + "data"
        self.face_detector = cv2.CascadeClassifier(base_path + os.sep + 'haarcascade_frontalface_default.xml')
        self.face_map = np.zeros((self.h, self.w), dtype=np.float32)

    def reset(self, y=[], t=0):
        self.t = t
        self.y = y

        if not self.y:
            self.y = GEymolCV.__generate_initial_conditions(self.h, self.w)

        self.first_call = True

    def next_location(self, frame_t, of_t, lock=None, frame_gray_uint8_cpu=None):

        # ensuring data is well shaped
        if (frame_t.ndim == 3 and frame_t.shape[2] != 1) or (frame_t.ndim != 2 and frame_t.ndim != 3):
            raise ValueError("Unsupported data format for the input frame: " + str(frame_t.shape) +
                             " (expected h x w x 1 or h x w)")
        if of_t.ndim != 3 or of_t.shape[2] != 2:
            raise ValueError("Unsupported data format for the optical flow data: " + str(of_t.shape) +
                             " (expected h x w x 2)")

        # let's start from the initial position
        if self.first_call:
            self.first_call = False
            return self.y

        # computing features
        gradient_norm_t = GEymolCV.__get_gradient_norm(frame_t) * (1.0 - self.IOR_matrix)
        of_norm_t = GEymolCV.__get_opticalflow_norm(of_t)

        # stacking features
        if self.parameters['alpha_fm'] > 0.0:
            face_map_t = self.face_map * (1.0 - self.IOR_matrix)
            feature_maps = (gradient_norm_t, of_norm_t, face_map_t)
        else:
            feature_maps = (gradient_norm_t, of_norm_t)

        # integrating ODE
        if lock is not None:
            with lock:
                y = odeint(GEymolCV.__my_ode, self.y, np.arange(self.t, self.t + 1, .1),  # instants to integrate (10)
                           args=(feature_maps, self.parameters, self.gravitational_filter),
                           mxstep=00, rtol=0.1, atol=0.1
                           )
                self.y = list(y[-1])  # picking up the latest integrated time instant
        else:
                y = odeint(GEymolCV.__my_ode, self.y, np.arange(self.t, self.t + 1, .1),  # instants to integrate (10)
                           args=(feature_maps, self.parameters, self.gravitational_filter),
                           mxstep=100, rtol=0.1, atol=0.1
                           )
                self.y = list(y[-1])  # picking up the latest integrated time instant

        # next time instant
        self.t += 1

        # avoid predicting out-of-frame locations
        foa_xy_and_velxy = self.y
        foa_xy_and_velxy[0], foa_xy_and_velxy[1] = \
            GEymolCV.__stay_inside_fix_nans_round_to_int((self.h, self.w), foa_xy_and_velxy[0:2])

        # add pixel coordinates to the inhibition of return matrix
        if not self.is_online:
            if self.t % max(int(float(self.parameters['fps']) / self.saccades_per_second),1) == 0:
                self.IOR_matrix = self.__inhibit_return_in(self.IOR_matrix, row_col=foa_xy_and_velxy[0:2])
        else:
            if time.clock() - self.real_time_last_saccade >= (1.0 / self.saccades_per_second):
                self.IOR_matrix = self.__inhibit_return_in(self.IOR_matrix, row_col=foa_xy_and_velxy[0:2])
                self.real_time_last_saccade = time.clock()  # update real time of the last saccade

        # update the face map
        if self.face_map is not None:
            self.__update_face_map(frame_t)

        return foa_xy_and_velxy

    def __update_face_map(self, frame_t, updating_factor=.3):

        if frame_t.dtype != np.uint8:
            frame_t = (frame_t * 255.0).astype(np.uint8)

        # add potential in locations of faces
        faces = self.face_detector.detectMultiScale(frame_t, 1.3, 5)
        face_map_new = np.zeros_like(self.face_map)
        for (y, x, h, w) in faces:
            face_map_new[x:x+w, y:y+h] = 1.0  # tested

        # update as weighted sum
        self.face_map = (1.0 - updating_factor) * self.face_map + updating_factor * face_map_new

    def __generate_approximation_of_2d_gaussian(self, row_col, ray=25, blur=151):
        row, col = row_col
        blank_image_with_circle = np.zeros((self.h, self.w), dtype=np.float32)
        cv2.circle(blank_image_with_circle, (col, row), ray, 1.0, -1)  # draw a filled circle (setting it to 1.0)
        gaussian_2d_approx = cv2.GaussianBlur(blank_image_with_circle, (blur,blur), 0)  # blur the whole image
        max_val = np.max(gaussian_2d_approx)

        if max_val < 1.0:
            gaussian_2d_approx = gaussian_2d_approx / max_val # normalize in [0,1]
        return gaussian_2d_approx

    def __inhibit_return_in(self, frame, row_col):
        cx = (self.h // 2)
        cy = (self.w // 2)
        ox = cx - row_col[0]
        oy = cy - row_col[1]
        gaussian_2d_approx = \
            GEymolCV.__extract_patch(self.centered_gaussian_2d_approx, [cx + ox, cy + oy], [self.h, self.w])

        frame = 0.9 * frame + gaussian_2d_approx
        frame[frame > 1.0] =  1.0
        return frame

    @staticmethod
    def __generate_initial_conditions(h, w):
        init_ray = int(min(h, w) * 0.17)  # arbitrary (it should be improved)
        x1_init = int(h / 2) + randint(-init_ray, init_ray)  # arbitrary (it should be improved)
        x2_init = int(w / 2) + randint(-init_ray, init_ray)  # arbitrary (it should be improved)
        v1_init = 2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))  # arbitrary (it should be improved)
        v2_init = 2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))  # arbitrary (it should be improved)
        return [x1_init, x2_init, v1_init, v2_init]

    @staticmethod
    def __create_gravitational_filter(filter_size):
        filter_matrix = np.zeros((2, filter_size, filter_size))  # size of the filter: 2 x filter_size x filter_size
        center_x, center_y = (filter_size // 2), (filter_size // 2)

        for i in range(filter_size):
            for j in range(filter_size):
                if not (i == center_x and j == center_y):  # avoid mid of the filter (set it to zero)
                    filter_matrix[0, i, j] = (filter_size / 10.0 + 1.0) * float(i - center_x) / (
                            ((i - center_x) ** 2 + (j - center_y) ** 2) + (filter_size / 10.0))

        for i in range(filter_size):
            for j in range(filter_size):
                if not (i == center_x and j == center_y):  # avoid mid of the filter (set it to zero)
                    filter_matrix[1, i, j] = (filter_size / 10.0 + 1.0) * float(j - center_y) / (
                            ((i - center_x) ** 2 + (j - center_y) ** 2) + (filter_size / 10.0))

        return filter_matrix

    @staticmethod
    def __get_gradient_norm(frame_t):
        sobel_x = cv2.Sobel(frame_t, cv2.CV_32F, 1, 0, ksize=5)  # float32 (kernel size could be reduced...)
        sobel_y = cv2.Sobel(frame_t, cv2.CV_32F, 0, 1, ksize=5)  # float32

        # getting norm
        grad_norm = np.sqrt(sobel_x**2 + sobel_y**2)

        return grad_norm

    @staticmethod
    def __get_opticalflow_norm(of_t):

        # getting norm
        of_norm = np.squeeze(np.sqrt(np.sum(of_t**2, axis=2)))

        # get outliers (it solves ego-motion)
        of_norm = of_norm - np.mean(of_norm)
        of_norm = np.abs(of_norm)

        return of_norm

    @staticmethod
    def __stay_inside_fix_nans_round_to_int(frame_hw, row_col, ray=5):
        row, col = row_col

        if isnan(row) or isnan(col):
            row, col = 0, 0
        else:
            row, col = int(row), int(col)

        if row - ray < 0:
            row = ray
        else:
            if row + ray >= frame_hw[0]:
                row = frame_hw[0] - ray - 1
        if col - ray < 0:
            col = ray
        else:
            if col + ray >= frame_hw[1]:
                col = frame_hw[1] - ray - 1

        return row, col

    @staticmethod
    def __extract_patch(frame, patch_center_xy, patch_size_xy, normalize=False):
        x, y = patch_center_xy
        odd_x = patch_size_xy[0] % 2
        odd_y = patch_size_xy[0] % 2
        d_x = patch_size_xy[0] // 2
        d_y = patch_size_xy[1] // 2
        h, w = frame.shape[0], frame.shape[1]

        # avoid extracting patches that are centered out-of-the-frame-coordinates
        if x < 0:
            x = 0
        elif x >= h:
            x = h-1

        if y < 0:
            y = 0
        elif y >= w:
            y = w-1

        # integer coordinates
        x = int(x)
        y = int(y)

        # handling borders
        f_x = x - d_x
        t_x = x + d_x
        f_y = y - d_y
        t_y = y + d_y
        if t_x >= h or f_x < 0 or t_y >= w or f_y < 0:
            patch = np.zeros((patch_size_xy[0], patch_size_xy[1]), dtype=frame.dtype)

            sf_x = 0
            st_x = 0
            sf_y = 0
            st_y = 0
            cf_x = f_x
            cf_y = f_y
            ct_x = t_x
            ct_y = t_y

            if f_x < 0:
                cf_x = 0
                sf_x = -f_x
            if t_x >= h:
                ct_x = h-1
                st_x = t_x - ct_x
            if f_y < 0:
                cf_y = 0
                sf_y = -f_y
            if t_y >= w:
                ct_y = w-1
                st_y = t_y - ct_y

            patch[sf_x:patch_size_xy[0]-st_x, sf_y:patch_size_xy[1]-st_y] = frame[cf_x:ct_x+odd_x, cf_y:ct_y+odd_y]
        else:
            patch = frame[f_x:t_x+1, f_y:t_y+1]

        # normalizing
        if normalize:
            max_val = np.max(patch)
            if max_val > 0.0:
                return patch / max_val
            else:
                return patch
        else:
            return patch

    @staticmethod
    def __my_ode(y, t, feature_maps, parameters, gravitational_filter):
        dissipation = parameters['dissipation']
        alpha_c = parameters['alpha_c']
        alpha_of = parameters['alpha_of']
        alpha_fm = parameters['alpha_fm']
        filter_size = parameters['max_distance']
        filter_sizes = [filter_size, filter_size]
        filter_area = filter_size * filter_size

        # extracting patches from the considered features
        gradient_norm_t_patch = GEymolCV.__extract_patch(feature_maps[0], y[0:2], filter_sizes, normalize=True)
        of_norm_t_patch = GEymolCV.__extract_patch(feature_maps[1], y[0:2], filter_sizes, normalize=True)

        # computing gravitational fields contributions
        matrix_gravitational_filter = gravitational_filter.reshape((2, filter_area))  # same array, no copies
        gravitational_grad = alpha_c * np.dot(matrix_gravitational_filter,
                                              gradient_norm_t_patch.reshape(filter_area))  # matrix-by-vec
        gravitational_of = alpha_of * np.dot(matrix_gravitational_filter,
                                             of_norm_t_patch.reshape(filter_area))  # matrix-by-vec

        if parameters['alpha_fm'] > 0.0:
            face_map_t_patch = GEymolCV.__extract_patch(feature_maps[2], y[0:2], filter_sizes, normalize=False)
            gravitational_faces = alpha_fm * np.dot(matrix_gravitational_filter,
                                                    face_map_t_patch.reshape(filter_area))
        else:
            gravitational_faces = np.zeros_like(gravitational_grad)

        # building the system of differential equations (4 equations)
        # y[2]
        # y[3]
        # gravitational_grad[0] + gravitational_of[0] + gravitational_faces[0] - dissipation * y[2]
        # gravitational_grad[1] + gravitational_of[1] + gravitational_faces[1] - dissipation * y[3]
        dy = np.concatenate([np.array(y[2:]),
                             gravitational_grad +
                             gravitational_of +
                             gravitational_faces -
                             dissipation * np.array(y[2:])])

        return dy
