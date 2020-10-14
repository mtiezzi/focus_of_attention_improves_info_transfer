import json
import torch
import numpy as np

device_cpu = torch.device("cpu")


def load_json(json_file_path):
    f = open(json_file_path, "r")
    if f is None or not f or f.closed:
        raise IOError("Cannot read: " + json_file_path)
    json_loaded = json.load(f)
    f.close()
    return json_loaded


def save_json(json_file_path, json_to_save):
    f = open(json_file_path, "w")
    if f is None or not f or f.closed:
        raise IOError("Cannot access: " + json_file_path)
    json.dump(json_to_save, f, indent=4)
    f.close()


def np_uint8_to_torch_float_01(numpy_img, device=None):
    if numpy_img.ndim == 2:
        h = numpy_img.shape[0]
        w = numpy_img.shape[1]
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img).float().div_(255.0).resize_(1, 1, h, w)
        else:
            return torch.from_numpy(numpy_img).float().resize_(1, 1, h, w).to(device).div_(255.0)
    elif numpy_img.ndim == 3:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().unsqueeze_(0).div_(255.0)
        else:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().to(device).unsqueeze_(0).div_(255.0)
    elif numpy_img.ndim == 4:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float().div_(255.0)
        else:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float().to(device).div_(255.0)
    else:
        raise ValueError("Unsupported image type.")


def np_float32_to_torch_float(numpy_img, device=None):
    if numpy_img.ndim == 2:
        h = numpy_img.shape[0]
        w = numpy_img.shape[1]
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img).resize_(1, 1, h, w)
        else:
            return torch.from_numpy(numpy_img).resize_(1, 1, h, w).to(device)
    elif numpy_img.ndim == 3:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().unsqueeze_(0)
        else:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().unsqueeze_(0).to(device)
    elif numpy_img.ndim == 4:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float()
        else:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float().to(device)
    else:
        raise ValueError("Unsupported image type.")


def torch_float32_to_grayscale_float32(torch_img):
    return torch.sum(torch_img *
                     torch.tensor([[[[0.114]], [[0.587]], [[0.299]]]],
                                  dtype=torch.float32, device=torch_img.device), 1, keepdim=True)


def torch_float_01_to_np_uint8(torch_img):
    return (torch_img * 255.0).cpu().astype(np.uint8)


def torch_2d_tensor_to_csv(tensor, file):
    with open(file, 'w+') as f:
        for i in range(0, tensor.shape[0]):
            for j in range(0, tensor.shape[1]):
                f.write(str(tensor[i,j].item()))
                if j < tensor.shape[1] - 1:
                    f.write(',')
            f.write('\n')


