#
# Copyright (C) 2020 Enrico Meloni, Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# L2S is licensed under a MIT license.
#
# You should have received a copy of the license along with this
# work. If not, see <https://en.wikipedia.org/wiki/MIT_License>.

# Import packages

import base64
import numpy as np
from lve.unity_client import AgentApi, SimulationApi
import cv2
import socket
from enum import IntFlag


# Import src

class FrameFlags(IntFlag):
    NONE = 0
    MAIN = 1
    CATEGORY = 1 << 2
    OBJECT = 1 << 3
    OPTICAL = 1 << 4
    DEPTH = 1 << 5


class SocketAgent:
    """
    TODO: summary ??? Maybe some more check to avoid connection errors?

    """

    def __init__(self,
                 main_frame_active: bool = True,
                 object_frame_active: bool = True,
                 category_frame_active: bool = True,
                 flow_frame_active: bool = True,
                 depth_frame_active: bool = True,
                 host: str = "localhost",
                 port: int = 8080,
                 width: int = 512,
                 height: int = 384):
        """

        :param main_frame_active: True if the virtual world should generate the main camera view
        :param object_frame_active: True if the virtual world should generate object instance supervisions
        :param category_frame_active: True if the virtual world should generate category supervisions
        :param flow_frame_active: True if the virtual world should generate optical flow data
        :param host: address on which the unity virtual world is listening
        :param port: port on which the unity virtual world is listening
        :param width: width of the stream, should be multiple of 8
        :param height: height of the stream, should be multiple of 8
        """
        self.flow_frame_active: bool = flow_frame_active
        self.category_frame_active: bool = category_frame_active
        self.object_frame_active: bool = object_frame_active
        self.main_frame_active: bool = main_frame_active
        self.depth_frame_active: bool = depth_frame_active

        self.flags = 0
        self.flags |= FrameFlags.MAIN if main_frame_active else 0
        self.flags |= FrameFlags.CATEGORY if category_frame_active else 0
        self.flags |= FrameFlags.OBJECT if object_frame_active else 0
        self.flags |= FrameFlags.DEPTH if depth_frame_active else 0
        self.flags |= FrameFlags.OPTICAL if flow_frame_active else 0

        self.id: int = -1
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.endianness = 'little'

    def register(self):
        """
        Register the agent on the Unity server and set its id.
        """
        self.socket.connect((self.host, self.port))
        # todo: check the representation given by to_bytes
        resolution_bytes = self.width.to_bytes(4, self.endianness) + \
                           self.height.to_bytes(4,self.endianness)  # todo: check endianness
        self.socket.send(resolution_bytes)

        agent_id_size = 4  # sizeof(int) in unity (c#)
        received = 0
        data = b""
        data += self.socket.recv(4)

        self.id = int.from_bytes(data, self.endianness)

    def delete(self):
        """
        Delete the agent on the Unity server.
        """
        self.socket.send(b'\x01')  # x01 is the code unity expects for deleting an agent

    def get_frame(self):
        """
        Get the frame from the cameras on the Unity server.

        :return: a dict of frames indexed by keywords main, object, category and flow.
        """
        frame = {}
        flags_bytes = self.flags.to_bytes(1, self.endianness)
        request_bytes = b'\x00' + flags_bytes

        self.socket.send(request_bytes)

        # start reading images from socket in the following order:
        # main, category, object, optical flow, depth
        if self.main_frame_active:
            frame_bytes = self.receive_frame()

            frame["main"] = cv2.imdecode(self.__decode_image(frame_bytes), cv2.IMREAD_COLOR)
        else:
            frame["main"] = None

        if self.category_frame_active:
            frame_bytes = self.receive_frame()

            cat_frame = self.__decode_category(frame_bytes)
            cat_frame = np.reshape(cat_frame, (self.height, self.width, 3))
            cat_frame = cat_frame[:, :, 0]
            frame["category"] = cat_frame.flatten()
            # frame["category_debug"] = self.__decode_image(base64_images["CategoryDebug"])
        else:
            frame["category"] = None
            # frame["category_debug"] = None

        if self.object_frame_active:
            frame_bytes = self.receive_frame()
            frame["object"] = self.__decode_image(frame_bytes)  # TODO: probably wrong format in unity
        else:
            frame["object"] = None

        if self.flow_frame_active:
            frame_bytes = self.receive_frame()
            flow = self.__decode_image(frame_bytes, np.float32)
            frame["flow"] = self.__decode_flow(flow)
        else:
            frame["flow"] = None

        if self.depth_frame_active:
            frame_bytes = self.receive_frame()
            frame["depth"] = cv2.imdecode(self.__decode_image(frame_bytes), cv2.IMREAD_COLOR)
        else:
            frame["depth"] = None

        # if "Timings" in content:
        #    # Convert elapsed milliseconds to seconds
        #    frame["timings"] = {key: (value / 1000) for key, value in content["Timings"].items()}
        #    frame["timings"]["Http"] = total_seconds

        return frame

    def receive_frame(self):
        frame_length = int.from_bytes(self.socket.recv(4), self.endianness)  # first we read the length of the frame
        return self.socket.recv(frame_length)

    def get_categories(self):
        return []
        # content = self.sim_api.get_categories()
        # categories = {}
        # for cat in content:
        #   categories[cat["Name"]] = cat["Id"]

        # return categories

    @staticmethod
    def __decode_image(bytes, dtype=np.uint8) -> np.ndarray:
        """
        Decode an image from the given bytes representation to a numpy array.

        :param bytes: the bytes representation of an image
        :return: the numpy array representation of an image
        """
        return np.frombuffer(bytes, dtype)

    def __decode_category(self, input) -> np.ndarray:
        """
        Decode the category supervisions from the given base64 representation to a numpy array.
        :param input: the base64 representation of categories
        :return: the numpy array containing the category supervisions
        """

        cat_frame = self.__decode_image(input)
        cat_frame = np.reshape(cat_frame, (self.height, self.width, -1))
        cat_frame = np.flipud(cat_frame)
        cat_frame = np.reshape(cat_frame, (-1))
        cat_frame = np.ascontiguousarray(cat_frame)
        return cat_frame

    def __decode_flow(self, flow_frame: np.ndarray) -> np.ndarray:
        flow = flow_frame
        flow = np.reshape(flow, (self.height, self.width, -1))
        flow = np.flipud(flow)
        flow = np.ascontiguousarray(flow)
        return flow
