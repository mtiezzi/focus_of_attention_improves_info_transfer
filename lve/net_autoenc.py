import torch
import torch.nn as nn
import torch.nn.functional as F
import lve
from collections import OrderedDict


class NetAutoEnc(nn.Module):

    def __init__(self, options):
        super(NetAutoEnc, self).__init__()

        # fixing options
        NetAutoEnc.__check_and_fix_options(options)

        self.convs = nn.ModuleList()
        self.deconvs = nn.ModuleList()

        # building feature-extraction layers
        for l in range(0, options["fe_layers"]):

            # shortcuts
            m = options["m"][l]

            prev_m = options["m"][l-1] if l > 0 else options["c"]
            ks = options["kernel_size"][l]

            # feature extraction (fe) layer (encoder)
            conv = lve.nn.Conv2dUnf(in_channels=prev_m, out_channels=m, kernel_size=ks, padding=ks // 2,
                                    bias=False, uniform_init=options["uniform_init"])

            deconv = nn.Conv2d(in_channels=m, out_channels=ks * ks * prev_m, kernel_size=1, padding=0, bias=False)

            # adding data to the main network module
            self.convs.append(conv)
            self.deconvs.append(deconv)

        # determining the number of layers on which we have to forward the signal
        self.last_active_layer = options["fe_layers"] - 1
        self.freeze = options["training"]["freeze_layers_below"]

        if options["training"]["layerwise"]:
            self.last_active_layer = options["training"]["last_active_layer"]
            if self.last_active_layer < 0:
                self.last_active_layer = options["fe_layers"] - 1
                
        # keeping track of the network options
        self.options = options

    def forward(self, frame, simple_forward=True):
        fe_outputs = []
        reconstructed_patches = []
        patches = []
        x = frame

        for l in range(0, self.last_active_layer + 1):

            # going forward on the current layer
            x, pat = self.convs[l](x, patches_needed=not simple_forward)
            x = torch.tanh(x)  # non-linearity to propagate signal to the upper layer

            if not simple_forward:
                patches.append(pat)
                reconstructed_patches.append(self.deconvs[l](x).reshape(pat.shape))

            # saving data
            fe_outputs.append(x)

        if simple_forward:
            return torch.cat(fe_outputs, dim=1)
        else:
            return fe_outputs, patches, reconstructed_patches

    def backward(self, loss):

        # real backward
        loss.backward()

    def compute_loss(self, forward_data, stat_data):

        # unpacking
        patches, feature_maps, reconstructed_patches = forward_data
        avg_losses, avg_losses_denominator, layers_stats = stat_data
        layers = self.last_active_layer + 1

        # data to return
        net_loss = 0.

        for l in range(0, layers):
            
            # shortcuts (options)
            lr = self.options["lambda_r"][l]
            ls = self.options["lambda_s"][l]

            # shortcuts (data)
            pat = patches[l]
            feature_map = feature_maps[l]
            rec = reconstructed_patches[l]
            avg_loss = avg_losses[l]
            avg_loss_denominator = avg_losses_denominator[l]

            layer_loss = lr * F.l1_loss(pat, rec, reduction='mean') + \
                ls * torch.mean(torch.sum(torch.abs(feature_map), dim=1))

            # augmenting the whole network loss
            net_loss += layer_loss

            with torch.no_grad():
                avg_loss, avg_loss_denominator = lve.nn.update_average(avg_loss, avg_loss_denominator,
                                                                       layer_loss, 1.0)
                norm_enc = torch.sum(self.convs[l].weight * self.convs[l].weight)
                norm_dec = torch.sum(self.deconvs[l].weight * self.deconvs[l].weight)

                # saving statistics that will be printed on screen or saved to disk
                layers_stats[l] = OrderedDict([
                    ('avgloss', avg_loss.item()),
                    ('avglossd', avg_loss_denominator),
                    ('loss', layer_loss.item()),
                    ('nenc', norm_enc.item()),
                    ('ndec', norm_dec.item())])

        return net_loss

    def get_conv_filters(self, layer):
        return self.convs[layer].weight.detach().cpu().numpy()

    @staticmethod
    def get_layer_stat_keys():
        return ['avgloss', 'avglossd', 'loss', 'nenc', 'ndec']

    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad.zero_()

    def activate_next_layer(self):
        layers = self.options['fe_layers']

        self.last_active_layer += 1
        if self.last_active_layer >= layers:
            self.last_active_layer = layers - 1

        if self.freeze:
            self.__stop_gradients_up_to_layer(self.last_active_layer)

    def __stop_gradients_up_to_layer(self, l):
        for i in range(0, l):
            for name, param in self.convs[i].named_parameters():
                param.requires_grad_(False)
            for name, param in self.deconvs[i].named_parameters():
                param.requires_grad_(False)

    @staticmethod
    def __check_and_fix_options(options):
        if options['fe_layers'] < 1:
            raise ValueError("Invalid value provided to option fe_layers: " + str(options['fe_layers']))

        layers = options['fe_layers']

        if options['training']['last_active_layer'] >= layers:
            raise ValueError("Invalid value provided to option last_active_layer: " + str(options['last_active_layer']))

        # those layers with no-defined-options will inherit them from the last layer with-defined-options
        for opt, val in options.items():
            if val is not None and isinstance(val, list):
                if len(val) < layers:
                    ref = val[-1] if len(val) > 0 else []
                    for i in range(layers - len(val)):
                        val.append(ref)
                if len(val) > layers:
                    for i in range(len(val) - layers):
                        val.pop()

    def __print_parameters(self):
        params = list(self.parameters())
        print("Number of tensor params: " + str(len(params)))
        for i in range(0, len(params)):
            p = params[i]
            print("   Tensor size: " + str(p.size()) + " (req. grad = " + str(p.requires_grad) + ")")