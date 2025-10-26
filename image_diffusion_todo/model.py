from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
    

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    #ddpm.py의 compute_loss()와 동일
    def get_loss(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)        

        # noise - train.py에서 not given.
        if noise is None:
            noise = torch.randn_like(x0)

        # 3) forward (q_sample): x_t = sqrt(bar_alpha_t) x0 + sqrt(1-bar_alpha_t) eps
        x_t, _ = self.var_scheduler.add_noise(x0, timestep, eps=noise)
        
        # predict noise
        eps_theta = self.network(x_t, timestep, class_label)

        # compute loss
        loss = F.mse_loss(eps_theta, noise)
        ######################
        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
    ):
        # 초기 노이즈 샘플링
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:

            ######## TODO ########
            # Assignment 2. Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            raise NotImplementedError("TODO")
            #######################

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## TODO ########
                # Assignment 2. Implement the classifier-free guidance.
                raise NotImplementedError("TODO")
                #######################
            else:
                noise_pred = self.network(x_t, timestep=t.to(self.device))

            # 우리가 구현한 step을 사용
            x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)

            # trajactory에 추가
            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    # 모델 파라미터 저장
    def save(self, file_path):
        # 객체 자체를 저장
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            } 
        # 모델 파라미터 저장
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    # 모델 파라미터 로드
    def load(self, file_path):
        # pytorch 2.6 option 수정.
        dic = torch.load(file_path, map_location="cpu", weights_only=False)
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
