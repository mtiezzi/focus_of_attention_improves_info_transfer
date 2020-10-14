import torch
import torch.nn as nn
import torch.nn.functional as F
import lve
from collections import OrderedDict


class NetCAL(nn.Module):

    def __init__(self, options):
        super(NetCAL, self).__init__()

        # fixing options
        NetCAL.__check_and_fix_options(options)

        self.convs = nn.ModuleList()
        self.cal_mods = nn.ModuleList()
        self.probs = nn.ModuleList()
        self.calprob_mods = nn.ModuleList()
        self.pm = [None] * (options["fe_layers"] + options["sem_layers"])

        self.total_fe_m = 0
        for l in range(0, options["fe_layers"]):
            self.total_fe_m += options["m"][l]
        self.invariant_fe_mask = torch.zeros(self.total_fe_m, device=torch.device("cpu"), dtype=torch.bool)
        k = 0
        for l in range(0, options["fe_layers"]):
            if options["lambda_m"][l] > 0.0:
                self.invariant_fe_mask[k:k+options["m"][l]] = True
            k += options["m"][l]

        # building CAL layers
        for l in range(0, options["fe_layers"] + options["sem_layers"]):

            # shortcuts
            m = options["m"][l]
            prob_layers_sizes = options["prob_layers_sizes"][l] if options["prob_layers_sizes"] is not None else None
            num_prob_layers = len(prob_layers_sizes) if prob_layers_sizes is not None else 0
            if num_prob_layers > 0:
                self.pm[l] = prob_layers_sizes[-1]
            else:
                self.pm[l] = m

            prev_m = (options["m"][l-1] if l > 0 else options["c"]) \
                if l != options["fe_layers"] or options["fe_layers"] == 0 else sum(options["m"][0:options["fe_layers"]])
            ks = options["kernel_size"][l]
            order = options["order"]
            opt = {x: options[x][l] for x in ["k", "gamma", "theta", "alpha", "beta", "reset_thres", "lambda_m",
                                              "zeta_s", "lambda_s"]}

            # feature extraction (fe) layer
            conv = lve.nn.Conv2dUnf(in_channels=prev_m, out_channels=m, kernel_size=ks, padding=ks // 2, bias=False,
                                    uniform_init=options["uniform_init"])

            if order == 4:
                cal_mod = lve.nn.CALReg4(conv, options=opt)
            elif order == 2:
                cal_mod = lve.nn.CALReg2(conv, options=opt)
            elif order == 1:
                cal_mod = lve.nn.CALReg1(conv, options=opt)
            else:
                raise ValueError("Unknown/unsupported CAL order.")

            probs = nn.ModuleList()
            if options["lambda_c"][l] > 0.0 and options["lambda_e"][l] > 0.0:
                prev_ps = m
                prob_layer = conv
                if num_prob_layers > 0:
                    for ps in prob_layers_sizes:
                        prob_layer = nn.Conv2d(in_channels=prev_ps, out_channels=ps, kernel_size=1, bias=False)
                        prev_ps = ps
                        probs.append(prob_layer)
                calprob_mod = lve.nn.CALProb(prob_layer, options=opt)
            else:
                calprob_mod = None

            # adding data to the main network module
            self.convs.append(conv)
            self.cal_mods.append(cal_mod)
            self.probs.append(probs)
            self.calprob_mods.append(calprob_mod)

        # determining the number of layers on which we have to forward the signal
        self.last_active_cal_layer = options["fe_layers"] + options["sem_layers"] - 1
        self.freeze = options["training"]["freeze_layers_below"]

        if options["training"]["layerwise"]:
            self.last_active_cal_layer = options["training"]["last_active_layer"]
            if self.last_active_cal_layer < 0:
                self.last_active_cal_layer = options["fe_layers"] + options["sem_layers"] - 1

        if "freezeall" in options["training"] and options["training"]["freezeall"]:
            self.__stop_gradients_up_to_cal_layer(options["fe_layers"] + options["sem_layers"])

        # keeping track of the network options
        self.options = options

    def forward(self, frame, simple_forward=True, patch_row_col=None):
        probabilities = []
        patches = []
        last_layer_logits = None
        conv_nonlin_outputs = []
        x = frame
        patches_needed = not simple_forward

        for l in range(0, self.last_active_cal_layer + 1):
            prob_needed = not simple_forward and \
                          (self.options["lambda_c"][l] > 0.0 and self.options["lambda_e"][l] > 0.0)

            # going forward on the current layer
            x, pat = self.convs[l](x, patches_needed=patches_needed, patch_row_col=patch_row_col)

            if prob_needed:
                p = x
                for prob_layer in self.probs[l]:
                    p = prob_layer(p)
                p = F.softmax(p, dim=1)
            else:
                p = None

            # CAL-related forward operations
            self.cal_mods[l]()
            if prob_needed:
                self.calprob_mods[l]()

            # saving data that will be returned by this function
            probabilities.append(p)
            patches.append(pat)

            if self.options["sem_layers"] <= 0 or l < self.options["fe_layers"] + self.options["sem_layers"] - 1:
                x = torch.tanh(x)  # non-linearity to propagate signal to the upper layergitgit
            else:
                last_layer_logits = x

            # saving data for the first semantic layer and to return it
            conv_nonlin_outputs.append(x)

            # preparing input of the first semantic layer
            if l == self.options["fe_layers"] - 1 and self.options["sem_layers"] > 0:
                x = torch.cat(conv_nonlin_outputs, dim=1)

        # restricting
        if patch_row_col is not None:
            for l in range(len(conv_nonlin_outputs)):
                w = conv_nonlin_outputs[l].shape[3]
                conv_nonlin_outputs_reshaped = conv_nonlin_outputs[l].view(conv_nonlin_outputs[l].shape[0],
                                                                           conv_nonlin_outputs[l].shape[1],
                                                                           conv_nonlin_outputs[l].shape[2] *
                                                                           conv_nonlin_outputs[l].shape[3])

                indices = (patch_row_col[:, 0] * w + patch_row_col[:, 1]).to(torch.long).unsqueeze(1).unsqueeze(1). \
                    repeat(1, conv_nonlin_outputs[l].shape[1], 1)

                conv_nonlin_outputs[l] = torch.gather(conv_nonlin_outputs_reshaped, 2, indices).view(
                    conv_nonlin_outputs[l].shape[0], conv_nonlin_outputs[l].shape[1], 1, 1)

        if not simple_forward:
            return probabilities, patches, last_layer_logits, conv_nonlin_outputs
        else:
            # if we have semantic layers running, we get back the logits of the last semantic layer;
            # if we do not have semantic layers or if the last semantic layer has not been activated yet, then we get
            # the stacked features extracted by the fe-layers
            return last_layer_logits if last_layer_logits is not None else (torch.cat(conv_nonlin_outputs, dim=1))

    def backward(self, loss, extra_backward_data):

        # unpacking
        coherence_matrices, probabilities_dot = extra_backward_data

        # real backward
        loss.backward()

        # completing regularization
        for l in range(0, self.last_active_cal_layer + 1):

            # CAL-related backward operations
            self.cal_mods[l].backward(coherence_matrices[l])

            if self.calprob_mods[l] is not None:
                self.calprob_mods[l].backward(probabilities_dot[l])

    def compute_probabilities_over_frame_and_time(self, probabilities):
        cal_layers = self.last_active_cal_layer + 1
        delta = self.options["step_size"]

        # data to return
        p_avgs = [None] * cal_layers
        p_dots = [None] * cal_layers

        for l in range(0, cal_layers):

            # shortcuts (options)
            lc = self.options["lambda_c"][l]
            le = self.options["lambda_e"][l]

            # shortcuts (data)
            p = probabilities[l]
            calprob_mod = self.calprob_mods[l]

            if lc > 0.0 and le > 0.0:
                p_dot, p_avg = calprob_mod.build_prob_vectors(p, delta)
            else:
                p_dot = None
                p_avg = None

            p_avgs[l] = p_avg
            p_dots[l] = p_dot

        return p_avgs, p_dots

    def compute_coherence_matrices(self, patch_data, coherence_data):

        # unpacking
        patches, prev_patches = patch_data
        coherence_indices, prev_M_N_mat = coherence_data  # motion-related data
        cal_layers = self.last_active_cal_layer + 1
        delta = self.options["step_size"]

        # data to return
        coherence_matrices = [None] * cal_layers

        for l in range(0, cal_layers):

            # shortcuts (options)
            lm = self.options["lambda_m"][l] / 2.0

            # shortcuts (data)
            pat = patches[l]
            prev_pat = prev_patches[l]
            prev_M_N = prev_M_N_mat[l]
            cal_mod = self.cal_mods[l]

            if lm > 0.0:
                coher_matrices = cal_mod.build_coherence_matrices(pat, prev_pat, delta, prev_M_N, coherence_indices)
            else:
                coher_matrices = None

            # saving references
            coherence_matrices[l] = coher_matrices

        return coherence_matrices

    def compute_mi_and_coherence_loss(self, forward_data, stat_data):

        # unpacking
        probabilities, p_avgs, coherence_matrices = forward_data  # data about current frame
        cal_layers_stats = stat_data  # stat-related data
        cal_layers = self.last_active_cal_layer + 1

        # data to return
        net_loss = 0.

        for l in range(0, cal_layers):
            
            # shortcuts (options)
            lc = self.options["lambda_c"][l]
            le = self.options["lambda_e"][l]
            lm = self.options["lambda_m"][l] / 2.0
            ls = self.options["lambda_s"][l]
            pm = self.pm[l]

            # shortcuts (data)
            p = probabilities[l]
            cal_mod = self.cal_mods[l]
            calprob_mod = self.calprob_mods[l]
            coher_matrices = coherence_matrices[l]
            p_avg = p_avgs[l]

            if lc > 0.0 and le > 0.0:

                # computing elements that are needed to setup the potential
                mi_loss = calprob_mod.compute_mi_approx(p, p_avg, weights=(lc, le))

                # unpacking
                weighed_mi, cond_entropy, entropy, p_avg_penalty = mi_loss
                mi = entropy - cond_entropy

            else:

                # fake values to be printed when the mutual information loss is off
                p_avg_penalty = torch.tensor(-1.)
                weighed_mi, cond_entropy, entropy, entropy_constraint_penalty = \
                    (torch.tensor(0., dtype=torch.float, device=self.convs[0].weight.device, requires_grad=True),
                     -1., -1., -1.)
                cond_entropy = torch.tensor(-1. * (pm - 1) / pm - 1.)
                entropy = torch.tensor(-1. * (pm - 1) / pm - 1.)
                mi = torch.tensor(-1. * (pm - 1) / pm)
                ls = 0.

            if lm > 0.0:

                # computing elements that are needed to setup the potential
                coher_loss = cal_mod.compute_coherence(coher_matrices)

                # unpacking
                coher, dirty_coher = coher_loss
            else:

                # fake values to be printed when the coherence loss is off
                coher = torch.tensor(-1.)
                dirty_coher = torch.tensor(-1.)
                lm = 0.

            # potential
            weighed_potential = - weighed_mi + lm * (coher + dirty_coher) + ls * p_avg_penalty

            # augmenting the whole network loss
            net_loss += weighed_potential

            # saving statistics that will be printed on screen or saved to disk
            cal_layers_stats[l] = OrderedDict([
                ('ca', 0.),
                ('cad', 0),
                ('lagr', weighed_potential),  # temporarily storing the weighed potential here (used below)
                ('mi', mi.item() * pm / (pm - 1.)),  # [0,1]
                ('ce', (cond_entropy.item() + 1.) * (pm / (pm - 1.))),  # [0,1]
                ('e', (entropy.item() + 1.) * (pm / (pm - 1.))),  # [0,1]
                ('sdotp', p_avg_penalty.item()),
                ('coher', coher.item()),
                ('coherk', dirty_coher.item()),
                ('n2', 0.),
                ('n1', 0.),
                ('nm', 0.),
                ('n', 0.),
                ('f', 0.),
                ('sup', -1),
                ('nsup', -1),
                ('evalmi', -1),
                ('evalmis', -1),
                ('evalmifoa', -1),
                ('evalmisfoa', -1),
                ('evalmifoaw', -1),
                ('evalmisfoaw', -1),
                ('evalgmi', -1),
                ('evalgmis', -1),
                ('evalgmiz', -1),
                ('evalgmisz', -1),
                ('evalgmifoa', -1),
                ('evalgmisfoa', -1),
                ('evalgmifoaz', -1),
                ('evalgmisfoaz', -1),
                ('evalgmifoaw', -1),
                ('evalgmisfoaw', -1),
                ('evalgmifoawz', -1),
                ('evalgmisfoawz', -1),
                ('evalgcoher', -1),
                ('evalgcoherk', -1),
                ('evalcohers', -1),
                ('evalgcohers', -1),
                ('evalgcohersz', -1),
                ('evalaccsup', -1),
                ('evalsup', -1)])

        return net_loss

    def compute_supervision_loss(self, logits, targets, indices, mask, stat_data):

        # unpacking
        net_sup_loss, net_num_sup, sup_layers_stats = stat_data

        sup_layer = self.options["fe_layers"] + self.options["sem_layers"] - 1
        num_sup = 0
        b = logits.shape[0]
        c = logits.shape[1]
        h = logits.shape[2]
        w = logits.shape[3]
        ll = self.options["lambda_l"]

        if self.last_active_cal_layer < sup_layer or ll <= 0.0:
            return net_sup_loss, num_sup
        else:
            if indices is not None:
                logits_flat = logits.view(b, c, h * w)  # b x c x wh

                if b == 1:
                    indices = indices.view(-1)  # from b x ns to flat vector
                    logits = logits_flat[:, :, indices]  # perhaps indexed_select is faster than torch.gather?
                else:
                    indices = indices.unsqueeze(1).expand(-1, c, -1)  # from b x ns to b x c x ns
                    logits = torch.gather(logits_flat, 2, indices)  # b x c x ns

                sup_loss = torch.sum(mask * F.cross_entropy(logits, targets, reduction='none'))
                num_sup = torch.sum(mask).item()
            else:
                sup_loss = F.cross_entropy(logits, targets, reduction='sum')  # do not average here!
                num_sup = b * w * h

            weighed_sup_loss = ll * sup_loss

            # saving statistics that will be printed on screen or saved to disk
            if sup_layers_stats['sup'] != -1:
                p_num_sup = sup_layers_stats['nsup']
                p_sup_loss = sup_layers_stats['sup']
                sup_layers_stats['sup'] = p_sup_loss + (sup_loss.item() - p_sup_loss * num_sup) / (p_num_sup + num_sup)
                sup_layers_stats['nsup'] = p_num_sup + num_sup
            else:
                sup_layers_stats['sup'] = sup_loss.item() / num_sup
                sup_layers_stats['nsup'] = num_sup

            net_sup_loss = net_sup_loss + (weighed_sup_loss - net_sup_loss * num_sup) / (net_num_sup + num_sup)
            net_num_sup = net_num_sup + num_sup

            sup_layers_stats['lagr'] = sup_layers_stats['lagr'] + net_sup_loss  # temporarily stored here, see below
        return net_sup_loss, net_num_sup

    def eval_and_accumulate_mi_and_coherence(self,
                                             probabilities,
                                             probabilities_foa,
                                             probabilities_foa_window,
                                             prev_tanh_fe_outputs, cur_tanh_fe_outputs, coherence_indices,
                                             cal_layers_stats,
                                             skip_coher=False):
        cal_layers = self.last_active_cal_layer + 1

        for l in range(0, cal_layers):

            # shortcuts (options)
            lc = self.options["lambda_c"][l]
            le = self.options["lambda_e"][l]
            lm = self.options["lambda_m"][l] / 2.0
            pm = self.pm[l]

            # shortcuts (data)
            p = probabilities[l]
            p_foa = probabilities_foa[l]
            p_foa_window = probabilities_foa_window[l]
            cal_mod = self.cal_mods[l]
            calprob_mod = self.calprob_mods[l]
            prev_f = prev_tanh_fe_outputs[l]
            cur_f = cur_tanh_fe_outputs[l]

            cal_layers_stats[l]['evalmi'] = -1.
            cal_layers_stats[l]['evalmis'] = -1.
            cal_layers_stats[l]['evalmifoa'] = -1.
            cal_layers_stats[l]['evalmisfoa'] = -1.
            cal_layers_stats[l]['evalmifoaw'] = -1.
            cal_layers_stats[l]['evalmisfoaw'] = -1.
            cal_layers_stats[l]['evalgmi'] = -1.
            cal_layers_stats[l]['evalgmiz'] = -1.
            cal_layers_stats[l]['evalgmis'] = -1.
            cal_layers_stats[l]['evalgmisz'] = -1.
            cal_layers_stats[l]['evalgmifoa'] = -1.
            cal_layers_stats[l]['evalgmifoaz'] = -1.
            cal_layers_stats[l]['evalgmisfoa'] = -1.
            cal_layers_stats[l]['evalgmisfoaz'] = -1.
            cal_layers_stats[l]['evalgmifoaw'] = -1.
            cal_layers_stats[l]['evalgmifoawz'] = -1.
            cal_layers_stats[l]['evalgmisfoaw'] = -1.
            cal_layers_stats[l]['evalgmisfoawz'] = -1.

            if lc > 0.0 and le > 0.0:

                # mi loss from the beginning of the network's life
                mi_global, mi_global_moving, mi_last = \
                    calprob_mod.compute_and_accumulate_global_mi_approx([p, p_foa, p_foa_window])
                mi_shannon_global, mi_shannon_global_moving, mi_shannon_last = \
                    calprob_mod.compute_and_accumulate_global_mi_shannon([p, p_foa, p_foa_window])

                cal_layers_stats[l]['evalmi'] = mi_last[0].item() * pm / (pm - 1.)
                cal_layers_stats[l]['evalmis'] = mi_shannon_last[0].item()

                cal_layers_stats[l]['evalmifoa'] = mi_last[1].item() * pm / (pm - 1.)
                cal_layers_stats[l]['evalmisfoa'] = mi_shannon_last[1].item()

                cal_layers_stats[l]['evalmifoaw'] = mi_last[2].item() * pm / (pm - 1.)
                cal_layers_stats[l]['evalmisfoaw'] = mi_shannon_last[2].item()

                cal_layers_stats[l]['evalgmi'] = mi_global[0].item() * pm / (pm - 1.)
                cal_layers_stats[l]['evalgmiz'] = mi_global_moving[0].item() * pm / (pm - 1.)
                cal_layers_stats[l]['evalgmis'] = mi_shannon_global[0].item()
                cal_layers_stats[l]['evalgmisz'] = mi_shannon_global_moving[0].item()

                cal_layers_stats[l]['evalgmifoa'] = mi_global[1].item() * pm / (pm - 1.)
                cal_layers_stats[l]['evalgmifoaz'] = mi_global_moving[1].item() * pm / (pm - 1.)
                cal_layers_stats[l]['evalgmisfoa'] = mi_shannon_global[1].item()
                cal_layers_stats[l]['evalgmisfoaz'] = mi_shannon_global_moving[1].item()

                cal_layers_stats[l]['evalgmifoaw'] = mi_global[2].item() * pm / (pm - 1.)
                cal_layers_stats[l]['evalgmifoawz'] = mi_global_moving[2].item() * pm / (pm - 1.)
                cal_layers_stats[l]['evalgmisfoaw'] = mi_shannon_global[2].item()
                cal_layers_stats[l]['evalgmisfoawz'] = mi_shannon_global_moving[2].item()

            if lm > 0.0:
                coher = cal_layers_stats[l]['coher']
                coherk = cal_layers_stats[l]['coherk']

                cohers = self.compute_symbolic_coherence(prev_f, cur_f, coherence_indices)

                # coherence from the beginning of the network's life (skipping some instants)
                coher_global, coherk_global, cohers_global, cohers_global_moving = \
                    cal_mod.accumulate_coherence(coher, coherk, cohers, fake=skip_coher)
            else:
                coher_global = torch.tensor(-1.)
                coherk_global = torch.tensor(-1.)
                cohers = torch.tensor(-1.)
                cohers_global = torch.tensor(-1.)
                cohers_global_moving = torch.tensor(-1.)

            cal_layers_stats[l]['evalgcoher'] = coher_global.item()
            cal_layers_stats[l]['evalgcoherk'] = coherk_global.item()
            cal_layers_stats[l]['evalcohers'] = cohers.item()
            cal_layers_stats[l]['evalgcohers'] = cohers_global.item()
            cal_layers_stats[l]['evalgcohersz'] = cohers_global_moving.item()

    def compute_symbolic_coherence(self, prev_tanh_fe_outputs, cur_tanh_fe_outputs, coherence_indices):
        with torch.no_grad():
            cur_f = cur_tanh_fe_outputs
            prev_f = prev_tanh_fe_outputs

            if cur_f is not None and prev_f is not None:
                if coherence_indices is not None:
                    prev_f = lve.nn.swap_by_indices(prev_f, coherence_indices[0, None])
                    if self.b > 1:
                        prev_f_batch = lve.nn.swap_by_indices(cur_f[0:-1, :], coherence_indices[1:, :])
                        prev_f = torch.cat((prev_f, prev_f_batch), dim=0)
                else:
                    prev_f = torch.cat((prev_f, cur_f[0:-1, :]), dim=0)

                prev_f = (prev_f > 0).to(torch.float)
                cur_f = (cur_f > 0).to(torch.float)
                cohers = torch.mean(torch.abs(prev_f - cur_f))

            return cohers

    def eval_supervision_loss(self, logits, targets, indices, mask, sup_layer_stats):
        with torch.no_grad():
            sup_layer = self.options["fe_layers"] + self.options["sem_layers"] - 1
            b = logits.shape[0]
            c = logits.shape[1]
            h = logits.shape[2]
            w = logits.shape[3]
            ll = self.options["lambda_l"]

            # data to compute
            net_sup_loss = 0.
            net_acc = 0.

            if self.last_active_cal_layer < sup_layer or ll <= 0.0:
                return net_sup_loss, net_acc
            else:
                if indices is not None:
                    logits_flat = logits.view(b, c, h * w)  # b x c x wh

                    if b == 1:
                        indices = indices.view(-1)  # from b x ns to flat vector
                        logits = logits_flat[:, :, indices]  # perhaps indexed_select is faster than torch.gather?
                    else:
                        indices = indices.unsqueeze(1).expand(-1, c, -1)  # from b x ns to b x c x ns
                        logits = torch.gather(logits_flat, 2, indices)  # b x c x ns

                    sup_loss = torch.sum(mask * F.cross_entropy(logits, targets, reduction='none'))
                    sup_right = torch.sum(mask * (torch.argmax(logits, dim=1) == targets.to(torch.float)))
                    num_sup = torch.sum(mask).item()
                else:
                    sup_loss = F.cross_entropy(logits, targets, reduction='sum')  # do not average here!
                    sup_right = torch.sum(torch.argmax(logits, dim=1) == targets).to(torch.float)
                    num_sup = b * w * h

                net_sup_loss = sup_loss / num_sup
                net_acc = sup_right / num_sup

            sup_layer_stats['evalsup'] = net_sup_loss.item()
            sup_layer_stats['evalaccsup'] = net_acc.item()

    def compute_ca(self, stat_data):
        CA, CA_denominator, T, cal_layers_stats = stat_data  # current value of the CALs and time indices
        cal_layers = self.last_active_cal_layer + 1
        total_CA = 0.0
        total_unscaled_lagrangian = 0.0

        for l in range(0, cal_layers):

            # shortcuts (data)
            ca = CA[l]
            ca_denominator = CA_denominator[l]
            t = T[l]
            cal_mod = self.cal_mods[l]

            with torch.no_grad():

                # potential energy
                weighed_potential = cal_layers_stats[l]['lagr']  # it was temporarily stored here

                # kinetic energy
                kinetic = cal_mod.compute_kinetic()
                weighed_kinetic, dot_dot_norm, dot_norm, mixed_norm, norm = kinetic

                # lagrangian
                lagrangian, lagrangian_scaling = cal_mod.compute_lagrangian(weighed_kinetic, weighed_potential, t)

                # CA
                ca, ca_denominator = lve.nn.update_average(ca, ca_denominator, lagrangian, lagrangian_scaling)

                # accumulating
                total_CA += ca.item()
                total_unscaled_lagrangian += lagrangian.item()

                # saving statistics that will be printed on screen or saved to disk
                cal_layers_stats[l].update(OrderedDict([
                    ('ca', ca.item()),
                    ('cad', ca_denominator),
                    ('lagr', lagrangian.item() * lagrangian_scaling),
                    ('n2', dot_dot_norm.item()),
                    ('n1', dot_norm.item()),
                    ('nm', mixed_norm.item()),
                    ('n', norm.item()),
                    ('f', self.convs[l].weight[0, 0, 0, 0].item())]))

        return total_CA, total_unscaled_lagrangian

    def force_reset(self):
        with torch.no_grad():
            for cal_reg in self.cal_mods:
                if cal_reg is not None:
                    cal_reg.zero_parameters()

    def force_reset_eval_avgs(self):
        with torch.no_grad():
            for calprob in self.calprob_mods:
                if calprob is not None:
                    calprob.reset_counters()

    def get_conv_filters(self, cal_layer):
        return self.convs[cal_layer].weight.detach().cpu().numpy()

    @staticmethod
    def get_layer_stat_keys():
        return ['ca', 'cad', 'lagr', 'mi', 'ce', 'e', 'sdotp', 'coher', 'coherk', 'n2', 'n1', 'nm', 'n', 'f',
                'sup', 'nsup',
                'evalmi', 'evalmis', 'evalmifoa', 'evalmisfoa', 'evalmifoaw', 'evalmisfoaw', 'evalgmi', 'evalgmis',
                'evalgmiz', 'evalgmisz', 'evalgmifoa', 'evalgmisfoa', 'evalgmifoaz', 'evalgmisfoaz', 'evalgmifoaw',
                'evalgmisfoaw', 'evalgmifoawz', 'evalgmisfoawz',
                'evalgcoher', 'evalgcoherk', 'evalcohers',
                'evalgcohers', 'evalgcohersz', 'evalaccsup', 'evalsup']

    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad.zero_()

    def activate_next_cal_layer(self):
        layers = self.options['fe_layers'] + self.options['sem_layers']

        self.last_active_cal_layer += 1
        if self.last_active_cal_layer >= layers:
            self.last_active_cal_layer = layers - 1

        if self.freeze:
            self.__stop_gradients_up_to_cal_layer(self.last_active_cal_layer)

    def __stop_gradients_up_to_cal_layer(self, l):
        for i in range(0, l):
            for name, param in self.convs[i].named_parameters():
                param.requires_grad_(False)
            for name, param in self.probs[i].named_parameters():
                param.requires_grad_(False)

            if self.cal_mods[i] is not None:
                self.cal_mods[i].disable_backward()
            if self.calprob_mods[i] is not None:
                self.calprob_mods[i].disable_backward()

    @staticmethod
    def __check_and_fix_options(options):
        if options['fe_layers'] < 1 and not (options['fe_layers'] == 0 and options['sem_layers'] > 0):
            raise ValueError("Invalid value provided to option fe_layers: " + str(options['fe_layers']))
        if options['sem_layers'] < 0:
            raise ValueError("Invalid value provided to option sem_layers: " + str(options['sem_layers']))

        layers = options['fe_layers'] + options['sem_layers']

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

    def print_parameters(self):
        params = list(self.parameters())
        print("Number of tensor params: " + str(len(params)))
        for i in range(0, len(params)):
            p = params[i]
            print("   Tensor size: " + str(p.size()) + " (req. grad = " + str(p.requires_grad) + ")")