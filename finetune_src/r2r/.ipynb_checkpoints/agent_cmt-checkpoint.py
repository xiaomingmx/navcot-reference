import json
import os
import sys
import numpy as np
import random
#from random import *
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.misc import length2mask
from utils.logger import print_progress

from models.model_HAMT import VLNBertCMT, Critic

from .eval_utils import cal_dtw

from .agent_base import BaseAgent

from transformers import get_linear_schedule_with_warmup

from utils.prompt_manager import PromptManager


class Seq2SeqCMTAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }
    # for k, v in env_actions.items():
    #     env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # Initialize scoring weight parameters
        self.score_alpha = getattr(args, 'score_alpha', 1)  # SAS weight default 0.6
        self.score_beta = getattr(args, 'score_beta', 0.3)    # POS weight default 0.3
        self.score_gamma = getattr(args, 'score_gamma', 0.1)  # MFS weight default 0.1
        self.n_candidates = getattr(args, 'n_candidates', 5)  # 候选数量

        # Models
        self._build_model()

        if not self.args.llm_predict:
            if self.args.world_size > 1:
                self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
                self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

            self.models = (self.vln_bert, self.critic)
            self.device = torch.device('cuda:%d'%self.rank) #TODO

            # Optimizers
            if self.args.optim == 'rms':
                optimizer = torch.optim.RMSprop
            elif self.args.optim == 'adam':
                optimizer = torch.optim.Adam
            elif self.args.optim == 'adamW':
                optimizer = torch.optim.AdamW
            elif self.args.optim == 'sgd':
                optimizer = torch.optim.SGD
            else:
                assert False
            if self.default_gpu:
                print('Optimizer: %s' % self.args.optim)

            self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)

            self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
            self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

            # Evaluations
            self.losses = []
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, size_average=False)

        if self.args.llm_predict:
            self.semantic_model = None
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded sentence-transformer model for semantic similarity")
            except ImportError:
                print("sentence-transformers not available, falling back to simple similarity")
                self.semantic_model = None


        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _build_model(self):
        if self.args.llm_predict:
            self.prompt_manager = PromptManager(self.args)

            sys.path.append("/root/autodl-tmp/navcot/NavCoT/LLaMA2-Accessory/accessory")
            from util.tensor_type import default_tensor_type
            from util.tensor_parallel import load_tensor_parallel_model_list
            from model.meta import MetaModel
            
            target_dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }.get(self.args.dtype, torch.float16) # Added .get for safety

            # Determine if vLLM should be used (e.g., from args)
            should_use_vllm = getattr(self.args, 'use_vllm', False) # Default to False if not specified
            hf_r2r_model_path = "/root/autodl-tmp/navcot/NavCoT/finetune_src/datasets/R2R/exprs/finetune/agent/finetuned_model-r2r-hf" # Your HF model path

            vllm_init_args = {}
            if should_use_vllm:
                # Populate vllm_init_args from self.args if they exist
                vllm_init_args = {
                    "tensor_parallel_size": getattr(self.args, 'vllm_tensor_parallel_size', 1),
                    "dtype": getattr(self.args, 'vllm_dtype', self.args.dtype), # Use general dtype if vllm_dtype not specified
                    "gpu_memory_utilization": getattr(self.args, 'vllm_gpu_memory_utilization', 0.90),
                    "tokenizer": getattr(self.args, 'vllm_tokenizer_path', hf_r2r_model_path), # vLLM often finds tokenizer in model_path
                    # Add any other vLLM specific args you want to control via self.args
                }

            if should_use_vllm:
                print("Attempting to initialize MetaModel with vLLM...")
                self.llm = MetaModel(
                    # llama_type, llama_config are not strictly needed by MetaModel when use_vllm=True,
                    # but pass them as None or from args if MetaModel __init__ expects them positionally before use_vllm
                    llama_type=self.args.llama_type, 
                    llama_config=self.args.llama_config,
                    tokenizer_path=self.args.tokenizer_path, # For pad_id fallback in MetaModel if vLLM tokenizer lacks it
                    max_seq_len=self.args.max_seq_len, # May be used by vLLM or MetaModel
                    with_visual=False, # Assuming text-only for this LLM part in R2R
                    use_vllm=True,
                    vllm_hf_model_path=hf_r2r_model_path,
                    vllm_args=vllm_init_args
                )
            else:
                print("Initializing MetaModel for native LLaMA execution...")
                with default_tensor_type(dtype=target_dtype, device="cuda"):
                    self.llm = MetaModel(
                        llama_type=self.args.llama_type,
                        llama_config=self.args.llama_config,
                        tokenizer_path=self.args.tokenizer_path,
                        max_seq_len=self.args.max_seq_len,
                        with_visual=False,
                        use_vllm=False
                    )
            
            # Conditional loading of .pth weights
            if not self.llm.weights_loaded_by_vllm:
                print(f"Loading .pth pretrained weights from {self.args.pretrained_path} for native LLaMA model.")
                load_result = load_tensor_parallel_model_list(self.llm, self.args.pretrained_path)
                print("Load result for .pth weights: ", load_result)
            else:
                print("Skipping .pth weight loading as vLLM has loaded the Hugging Face model.")

            print("LLM instance created = %s" % str(self.llm))

            # Conditional .cuda() call
            if not self.llm.use_vllm:
                if self.llm.llma is not None:
                    print("Moving native LLaMA model (self.llm.llma) to CUDA.")
                    self.llm.llma.cuda() # Move the actual nn.Module part (self.llma) to CUDA
                else:
                    print("Native LLaMA model (self.llm.llma) is None, skipping .cuda() call.")
            elif self.llm.use_vllm:
                print("vLLM engine is managing the model on GPU.")

        else: # Not self.args.llm_predict
            self.vln_bert = VLNBertCMT(self.args).cuda()
            self.critic = Critic(self.args).cuda()

    def get_direction(self, rel_heading, rel_elevation):
        if rel_elevation > 0:
            direction_text = 'go up to'
        elif rel_elevation < 0:
            direction_text = 'go down to'
        else:
            if rel_heading < 0:
                if rel_heading >= -math.pi / 2:
                    direction_text = 'turn left to'
                elif rel_heading < -math.pi / 2 and rel_heading > -math.pi * 3 / 2:
                    direction_text = 'go back to'
                else:
                    direction_text = 'turn right to'
            elif rel_heading > 0:
                if rel_heading <= math.pi / 2:
                    direction_text = 'turn right to'
                elif rel_heading > math.pi / 2 and rel_heading < math.pi * 3 / 2:
                    direction_text = 'go back to'
                else:
                    direction_text = 'turn left to'
            elif rel_heading == 0:
                direction_text = 'go forward to'
        return direction_text

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=int)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor)
        mask = torch.from_numpy(mask)
        return seq_tensor.long().cuda(), mask.cuda(), seq_lengths

    def _cand_pano_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        ob_cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        ob_lens = []
        ob_img_fts, ob_ang_fts, ob_nav_types = [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            cand_img_fts, cand_ang_fts, cand_nav_types = [], [], []
            cand_pointids = np.zeros((self.args.views, ), dtype=bool)
            for j, cc in enumerate(ob['candidate']):
                cand_img_fts.append(cc['feature'][:self.args.image_feat_size])
                cand_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                cand_pointids[cc['pointId']] = True
                cand_nav_types.append(1)
            # add [STOP] feature
            cand_img_fts.append(np.zeros((self.args.image_feat_size, ), dtype=float))
            cand_ang_fts.append(np.zeros((self.args.angle_feat_size, ), dtype=float))
            cand_img_fts = np.vstack(cand_img_fts)
            cand_ang_fts = np.vstack(cand_ang_fts)
            cand_nav_types.append(2)

            # add pano context
            pano_fts = ob['feature'][~cand_pointids]
            cand_pano_img_fts = np.concatenate([cand_img_fts, pano_fts[:, :self.args.image_feat_size]], 0)
            cand_pano_ang_fts = np.concatenate([cand_ang_fts, pano_fts[:, self.args.image_feat_size:]], 0)
            cand_nav_types.extend([0] * (self.args.views - np.sum(cand_pointids)))

            ob_lens.append(len(cand_nav_types))
            ob_img_fts.append(cand_pano_img_fts)
            ob_img_fts.append(cand_pano_img_fts)
            ob_ang_fts.append(cand_pano_ang_fts)
            ob_nav_types.append(cand_nav_types)

        # pad features to max_len
        max_len = max(ob_lens)
        for i in range(len(obs)):
            num_pads = max_len - ob_lens[i]
            ob_img_fts[i] = np.concatenate([ob_img_fts[i], \
                np.zeros((num_pads, ob_img_fts[i].shape[1]), dtype=float)], 0)
            ob_ang_fts[i] = np.concatenate([ob_ang_fts[i], \
                np.zeros((num_pads, ob_ang_fts[i].shape[1]), dtype=float)], 0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0] * num_pads)

        ob_img_fts = torch.from_numpy(np.stack(ob_img_fts, 0)).cuda()
        ob_ang_fts = torch.from_numpy(np.stack(ob_ang_fts, 0)).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()

        return ob_img_fts, ob_ang_fts, ob_nav_types, ob_lens, ob_cand_lens

    def _candidate_variable(self, obs, previous_angle=None):

        if previous_angle is not None:
            batch_cand_index = []  #cand index(0, 12, 35...)
            batch_cand_action = []

        cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        max_len = max(cand_lens)
        cand_img_feats = np.zeros((len(obs), max_len, self.args.image_feat_size), dtype=np.float_)
        cand_ang_feats = np.zeros((len(obs), max_len, self.args.angle_feat_size), dtype=np.float_)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int_)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            if previous_angle is not None:
                cand_index = []
                cand_action = []
                cand_vpids = []
            for j, cc in enumerate(ob['candidate']):
                cand_img_feats[i, j] = cc['feature'][:self.args.image_feat_size]
                cand_ang_feats[i, j] = cc['feature'][self.args.image_feat_size:]
                cand_nav_types[i, j] = 1
                # cand_vpids.append(cc['viewpointId'])
                if previous_angle is not None:
                    direction = self.get_direction(cc['absolute_heading'] - previous_angle[i]['heading'],
                                                        cc['absolute_elevation'] - 0)
                    cand_index.append(cc['pointId'])

                    action_text = direction + f"<{cc['caption']}>"
                    cand_action.append(action_text)
            if previous_angle is not None:
                cand_action.append('stop')
                batch_cand_index.append(cand_index)
                batch_cand_action.append(cand_action)
                # batch_cand_vpids.append(cand_vpids)
            cand_nav_types[i, cand_lens[i]-1] = 2 # stop

        cand_img_feats = torch.from_numpy(cand_img_feats).cuda()
        cand_ang_feats = torch.from_numpy(cand_ang_feats).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()
        if previous_angle is not None:
            return {
                'cand_img_feats':cand_img_feats,
                'cand_ang_feats':cand_ang_feats,
                'cand_nav_types':cand_nav_types,
                'cand_lens':cand_lens,
                'cand_action':batch_cand_action,
                'cand_index':batch_cand_index,
                # 'cand_vpids':batch_cand_vpids
            }
        else:
            return cand_img_feats, cand_ang_feats, cand_nav_types, cand_lens

    def _history_variable(self, obs):
        hist_img_feats = np.zeros((len(obs), self.args.image_feat_size), np.float_)
        for i, ob in enumerate(obs):
            hist_img_feats[i] = ob['feature'][ob['viewIndex'], :self.args.image_feat_size]
        hist_img_feats = torch.from_numpy(hist_img_feats).cuda()

        if self.args.hist_enc_pano:
            hist_pano_img_feats = np.zeros((len(obs), self.args.views, self.args.image_feat_size), np.float_)
            hist_pano_ang_feats = np.zeros((len(obs), self.args.views, self.args.angle_feat_size), np.float_)
            for i, ob in enumerate(obs):
                hist_pano_img_feats[i] = ob['feature'][:, :self.args.image_feat_size]
                hist_pano_ang_feats[i] = ob['feature'][:, self.args.image_feat_size:]
            hist_pano_img_feats = torch.from_numpy(hist_pano_img_feats).cuda()
            hist_pano_ang_feats = torch.from_numpy(hist_pano_ang_feats).cuda()
        else:
            hist_pano_img_feats, hist_pano_ang_feats = None, None

        return hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int_)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    # assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _get_ig_probs(self, obs):
        cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        max_len = max(cand_lens)
        ig_probs = np.zeros((len(obs), max_len, self.args.ig_head), dtype=np.float_)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int_)
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                ig_probs[i, j] = cc['ig_probs']
                cand_nav_types[i, j] = 1
            ig_probs[i, cand_lens[i]-1] = cc['ig_probs']
            cand_nav_types[i, cand_lens[i]-1] = 2

        ig_probs = torch.from_numpy(ig_probs).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()

        return ig_probs, cand_nav_types

    def _get_ig_target(self, obs, ended):
        ig_target = np.zeros((len(obs), 196), dtype=np.int_)
        for i, ob in enumerate(obs):
            if ended[i]:
                ig_target[i,:] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:
                        ig_target[i,:] = candidate['ig_probs']
                        break
                else:
                    assert ob['teacher'] == ob['viewpoint']
                    ig_target[i,:] = self.args.ignoreid

        ig_target = torch.from_numpy(ig_target).cuda()

        return ig_target

    def _get_ig_probs_full(self, obs):
        ig_probs = np.zeros((len(obs), 196, self.args.ig_head), dtype=np.float_)
        # cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int_)
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                if cc["viewpointId"] == ob['teacher']:
                    ig_probs[i] = cc["ig_probs"]
                    break
            else:
                assert ob['teacher'] == ob['viewpoint']         # All zeros probs

        ig_probs = torch.from_numpy(ig_probs).cuda()
        # cand_nav_types = torch.from_numpy(cand_nav_types).cuda()

        return ig_probs

    def _get_ig_probs_target(self, obs):
        ig_probs = np.zeros((len(obs), self.args.ig_head), dtype=np.float_)
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                if cc["viewpointId"] == ob['teacher']:
                    ig_probs[i] = cc["ig_probs"]
                    break
            else:
                assert ob['teacher'] == ob['viewpoint']  # All zeros probs

        ig_probs = torch.from_numpy(ig_probs).cuda()

        return ig_probs

    def make_equiv_action(self, a_t, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[i].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while self.env.env.sims[i].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[i].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx'])

                state = self.env.env.sims[i].getState()
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        batch_size = len(obs)

        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float_)
        last_ndtw = np.zeros(batch_size, np.float_)
        for i, ob in enumerate(obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        # for backtrack
        visited = [set() for _ in range(batch_size)]

        hist_embeds = [self.vln_bert('history').expand(batch_size, -1)]  # global embedding
        hist_lens = [1 for _ in range(batch_size)]

        for t in range(self.args.max_action_len):
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,    # history before t step
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True if self.feedback == 'sample' else False
            }

            t_outputs = self.vln_bert(**visual_inputs)
            logit = t_outputs[0]
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            if train_ml is not None:
                # Supervised training
                target = self._teacher_action(obs, ended)
                ml_loss += self.criterion(logit, target)

            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:
                bt_masks = torch.zeros(ob_nav_types.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                logit.masked_fill_(bt_masks, -float('inf'))

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (ob_cand_lens[i]-1) or next_id == self.args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len-1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float_)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': hist_img_feats,
                    'hist_ang_feats': prev_act_angle,
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': t,
                }
                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                hist_embeds.append(t_hist_embeds)

                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs(t=t+1)

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float_)
                ndtw_score = np.zeros(batch_size, np.float_)
                reward = np.zeros(batch_size, np.float_)
                mask = np.ones(batch_size, np.float_)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:                              # If the action now is end
                            if dist[i] < 3.0:                             # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:                                         # Incorrect
                                reward[i] = -2.0
                        else:                                             # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:                           # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True
            }

            _, last_h_ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float_)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5 # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss
                if self.feedback == 'sample':
                    rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                self.logs['critic_loss'].append(t_critic_loss.item())
                self.logs['policy_loss'].append(t_policy_loss.item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item()) # critic loss + policy loss + entropy loss

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        return traj


    def rollout_ig(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        batch_size = len(obs)

        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float_)
        last_ndtw = np.zeros(batch_size, np.float_)
        for i, ob in enumerate(obs):  # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        ig_loss = 0.

        # for backtrack
        visited = [set() for _ in range(batch_size)]

        hist_embeds = [self.vln_bert('history').expand(batch_size, -1)]  # global embedding
        hist_lens = [1 for _ in range(batch_size)]

        for t in range(self.args.max_action_len):
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,  # history before t step
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True if self.feedback == 'sample' else False
            }

            if self.args.weighted_token:
                visual_inputs['position_id'] = torch.from_numpy(np.ones((batch_size, 196), dtype=np.float_)/196).to(ob_masks.get_device())

            t_outputs = self.vln_bert(**visual_inputs)
            logit_action = t_outputs[0]
            ig_probs = t_outputs[-1]            # batch_size, ig_head
            if self.args.weighted_token:
                dvae_target = self._get_ig_probs_target(obs)
                ig_probs = F.log_softmax(ig_probs, dim=-1)
            else:
                cand_probs, cand_masks = self._get_ig_probs(obs)

                cand_probs = torch.transpose(cand_probs, 1, 2)
                logit_dvae = torch.matmul(ig_probs, cand_probs).squeeze(1)
                logit_dvae.masked_fill_(cand_masks==0, -float('inf'))

            logit = logit_action
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            target = self._teacher_action(obs, ended)

            if train_ml is not None:
                # Supervised training
                ml_loss += self.criterion(logit, target)

            if self.args.weighted_token:
                ig_loss += F.kl_div(ig_probs, dvae_target, reduction='none').sum(dim=1).mean()
            else:
                ig_loss += self.criterion(logit_dvae, target)

            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:
                bt_masks = torch.zeros(ob_nav_types.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                logit.masked_fill_(bt_masks, -float('inf'))

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)  # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (ob_cand_lens[i] - 1) or next_id == self.args.ignoreid or ended[
                    i]:  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len - 1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float_)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': hist_img_feats,
                    'hist_ang_feats': prev_act_angle,
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': t,
                }
                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                hist_embeds.append(t_hist_embeds)

                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs(t=t + 1, shortest_teacher=train_rl or self.feedback == 'argmax')

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float_)
                ndtw_score = np.zeros(batch_size, np.float_)
                reward = np.zeros(batch_size, np.float_)
                mask = np.ones(batch_size, np.float_)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:  # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True
            }

            if self.args.weighted_token:
                visual_inputs['position_id'] = torch.from_numpy(np.ones((batch_size, 196), dtype=np.float_)/196).to(ob_masks.get_device())

            _, last_h_, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float_)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss
                if self.feedback == 'sample':
                    rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                self.logs['critic_loss'].append(t_critic_loss.item())
                self.logs['policy_loss'].append(t_policy_loss.item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())  # critic loss + policy loss + entropy loss

        if train_ml is not None:
            ml_loss += self.args.train_ig * ig_loss
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())


        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        return traj

    def _calculate_sas(self, action_text, nav_input, obs_item, t):
        """
        计算语义对齐性评分 (Semantic Alignment Score, SAS)
        评估候选动作的语义描述与当前指令步骤的匹配程度
        """
        try:
            # 获取当前指令文本
            instruction = obs_item.get('instruction', '')
            if not instruction:
                print("Warning: No instruction found in obs_item")
                return 0.0
            
            # 1. 语义相似度评分 (替代传统关键词匹配)
            if self.semantic_model is not None:
                # 使用BERT等模型计算语义相似度
                try:
                    # 对指令和动作文本进行编码
                    instruction_embedding = self.semantic_model.encode(instruction, convert_to_tensor=False)
                    action_embedding = self.semantic_model.encode(action_text, convert_to_tensor=False)
                    
                    # 计算余弦相似度
                    import torch.nn.functional as F
                    import torch
                    
                    instruction_tensor = torch.tensor(instruction_embedding).unsqueeze(0)
                    action_tensor = torch.tensor(action_embedding).unsqueeze(0)
                    
                    semantic_similarity = F.cosine_similarity(instruction_tensor, action_tensor).item()
                    
                    # 将相似度从[-1,1]范围映射到[0,1]
                    semantic_score = (semantic_similarity + 1) / 2
                    
                    print(f"Semantic similarity: {semantic_similarity:.3f}, Score: {semantic_score:.3f}")
                    
                except Exception as e:
                    print(f"Error computing semantic similarity: {e}")
                    # 回退到传统方法
                    semantic_score = self._compute_traditional_keyword_score(instruction, action_text)
            else:
                # 回退到传统关键词匹配方法
                semantic_score = self._compute_traditional_keyword_score(instruction, action_text)
            
            # 2. 动作时序一致性检查
            sequence_score = 1.0  # 默认分数
            
            # 检查是否是"go back"类型的动作（通常不被鼓励，除非指令明确要求）
            if 'go back' in action_text.lower() or 'return' in action_text.lower():
                if 'back' not in instruction.lower() and 'return' not in instruction.lower():
                    sequence_score = 0.3  # 惩罚意外的返回动作
            
            # 检查停止动作的合理性
            if 'stop' in action_text.lower():
                # 如果指令包含停止相关词汇，给予高分
                if any(word in instruction.lower() for word in ['stop', 'end', 'finish', 'done', 'reach']):
                    sequence_score = 1.2
                else:
                    # 根据历史步数判断停止是否合理（简化判断）
                    if t < 2:  # 太早停止
                        sequence_score = 0.1
                    elif t > 10:  # 很晚才停止，可能合理
                        sequence_score = 0.8
            
            # 3. 方向一致性检查（简化版本）
            direction_score = 1.0
            direction_words = ['left', 'right', 'forward', 'straight', 'ahead']
            
            for direction in direction_words:
                if direction in instruction.lower() and direction in action_text.lower():
                    direction_score = 1.3  # 方向匹配奖励
                    break
                elif direction in instruction.lower() and direction not in action_text.lower():
                    # 指令要求某方向但动作不包含，可能需要检查其他线索
                    pass
            
            # 综合评分
            final_sas = semantic_score * 0.4 + sequence_score * 0.4 + direction_score * 0.2
            
            # 限制分数范围在 [0, 1] 之间
            final_sas = max(0.0, min(1.0, final_sas))
            
            return final_sas
            
        except Exception as e:
            print(f"Error in _calculate_sas: {e}")
            return 0.0

    def _select_best_action(self, n_candidate_outputs, n_candidate_action_idx, batch_idex, cand_inputs, nav_input, obs, t):
        """
        对 batch中每个样本的 N 个候选动作进行评分并选择最佳动作。
        """
        best_score = -float('inf')
        best_action_index_in_candidates = -1 # N个候选中的索引
        final_action_id = -1 # 最终选出的动作ID (对应环境)

        if not n_candidate_action_idx: # 如果没有有效的候选动作
             print("Warning: No valid candidate actions generated.")
             # 默认选择停止，或者你可以定义其他回退逻辑
             stop_action_id = cand_inputs['cand_lens'][batch_idex] - 1 if not self.args.stop_first else 0
             return stop_action_id # 返回第一个候选的索引

        for i in range(len(n_candidate_action_idx)):
            action_id = n_candidate_action_idx[i]
            action_text = n_candidate_outputs[i] # 假设原始LLM输出文本也需要用于SAS

            # 确保 action_id 是有效的候选动作索引 (非 ignore id)

            # --- 实现 action_id 到 candidate 索引的转换 ---
            # 这里假设 get_output 返回的是基于选项列表的索引（可能需要调整 get_output）
            # 或者直接修改 get_output 让它返回原始候选列表中的索引
            current_cand_action_index = action_id # 假设 action_id 是 cand_inputs['cand_action'][obs_index] 的索引
            
            # 检查索引有效性
            if action_id < 0 or action_id >= len(cand_inputs['cand_action'][batch_idex]):
                 print(f"Warning: Skipping invalid action_id {action_id} from LLM output.")
                 continue # 跳过无效的动作 ID

            # --- 计算分数 ---
            sas_score = self._calculate_sas(action_text, nav_input, obs[batch_idex], t)
            # pos_score = self._calculate_pos(current_cand_action_index, cand_inputs, obs[obs_index], obs_index, t) # 传递 obs_index
            # mfs_score = self._calculate_mfs(current_cand_action_index, cand_inputs, obs[obs_index], obs_index, t) # 传递 obs_index

            final_score = self.score_alpha * sas_score
            #               self.score_beta * pos_score + \
            #               self.score_gamma * mfs_score

            # print(f"  Candidate {i}: ActionID={action_id}, Text='{action_text[:30]}...', SAS={sas_score:.2f}, POS={pos_score:.2f}, MFS={mfs_score:.2f}, Final={final_score:.2f}")
            print(f"  Candidate {i}: ActionID={action_id}, Text='{action_text[:30]}...', SAS={sas_score:.2f} \n")
            if final_score > best_score:
                best_score = final_score
                best_action_index_in_candidates = i


        if best_action_index_in_candidates == -1:
            # 如果所有候选都无效或评分都为负无穷，选择默认动作（例如停止）
            print("Warning: Could not select a best action, defaulting to stop.")
            return 0 # 返回第一个候选的索引和停止动作ID

        print(f"Selected Candidate {best_action_index_in_candidates}, Score: {best_score:.2f}")
        # 返回最佳候选的索引（在N个候选中）
        return best_action_index_in_candidates

    # Helper function within _calculate_pos and _calculate_mfs
    def _get_candidate_info(self, action_index, cand_inputs, obs_index):
        """ Safely retrieves candidate information """
        if action_index < 0 or action_index >= len(cand_inputs['cand_action'][obs_index]):
            return None # Invalid index
        # You might want to return more info from cand_inputs based on the index
        return {
            'action_text': cand_inputs['cand_action'][obs_index][action_index],
            'viewpoint_id': obs[obs_index]['candidate'][action_index]['viewpointId'] if action_index < len(obs[obs_index]['candidate']) else None, # Check index boundary for viewpointId
            # Add other relevant candidate details here
        }

    # Modified _calculate_pos with obs_index
    def _calculate_pos(self, candidate_action_index, cand_inputs, obs_item, obs_index, t):
        """
        计算路径最优性 (Path Optimality Score - POS)
        你需要在这里实现具体的计算逻辑。
        例如：考虑候选动作对应的视图与目标的距离、与最短路径的偏差等。
        """
        candidate_info = self._get_candidate_info(candidate_action_index, cand_inputs, obs_index)
        if candidate_info is None:
            print(f"Warning: _calculate_pos got invalid index {candidate_action_index} for obs {obs_index}. Returning 0.")
            return 0.0

        # Placeholder: 返回一个随机分数或固定值
        # 简单的示例：倾向于不停止的动作 (这只是一个非常基础的例子)
        is_stop_action = (candidate_info['action_text'] == 'stop') # More robust check for stop
        score = 0.1 if is_stop_action else 1.0
        # 你可以在这里访问 obs_item['distance'], obs_item['candidate'][candidate_action_index]['distance_to_target'] (如果存在) 等信息
        print(f"Warning: _calculate_pos not implemented for action index {candidate_action_index}. Returning basic score: {score}")
        return score # <--- 在此实现 POS 计算

    # Modified _calculate_mfs with obs_index
    def _calculate_mfs(self, candidate_action_index, cand_inputs, obs_item, obs_index, t):
        """
        计算多模态融合分数 (Multimodal Fusion Score - MFS)
        你需要在这里实现具体的计算逻辑。
        例如：评估动作对应的视觉特征（图像、角度）与指令/历史的一致性。
        也许可以利用模型的内部注意力或表示。
        """
        candidate_info = self._get_candidate_info(candidate_action_index, cand_inputs, obs_index)
        if candidate_info is None:
            print(f"Warning: _calculate_mfs got invalid index {candidate_action_index} for obs {obs_index}. Returning 0.")
            return 0.0

        # Placeholder: 返回一个随机分数或固定值
        # 你可以在这里访问 cand_inputs['cand_img_feats'][obs_index][candidate_action_index], etc.
        print(f"Warning: _calculate_mfs not implemented for action index {candidate_action_index}. Returning 0.")
        return 0.0 # <--- 在此实现 MFS 计算

    def rollout_llm(self, train_ml=None, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param reset:       Reset the environment

        :return:
        """
        # obs 是一个batch中每个样本的观察数据组成的的列表，每个 ob 包含一个样本的观察数据
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        batch_size = len(obs)

        previous_angle = [{'heading': 0.,
                               'elevation': 0.} for ob in obs]


        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        #just_ended = np.array([False] * batch_size) # Seems unused

        #self.loss = 0

        for t in range(self.args.max_action_len):
            #print(t)
            if self.args.dataset == 'r2r':
                nav_targets = self._teacher_action(
                    obs=obs, ended=ended)

            if t == 0:
                for i in range(batch_size):
                    self.prompt_manager.history[i] = ''

            if self.args.ob_type == 'cand':
                cand_inputs = self._candidate_variable(obs = obs, previous_angle = previous_angle)

                nav_input = self.prompt_manager.get_prompt(mode = 'navigation', cand_inputs = cand_inputs, obs = obs, t = t)
                
                if self.args.use_vllm != True:
                    nav_output = self.llm.generate(nav_input["prompts"],images=None,max_gen_len=64,temperature=self.args.temperature)
                    a_t = self.prompt_manager.get_output(nav_output=nav_output,
                                                         only_options_batch=nav_input["only_options"],
                                                         cand_inputs=cand_inputs, t=t)
                    # print(f"a_t: {a_t} \n")
                else:
                    best_action_index_by_batch = []
                    # 使用 vLLM 生成多个候选
                    # Best-of-N 策略: 生成多个候选，然后选择最佳的
                    print(f"Generating {self.n_candidates} candidates for best-of-n selection...")
                    
                    # 使用 vLLM 的 SamplingParams 生成多个候选
                    # 为每个样本生成 N 个候选，结果形状为 [batch_size, n_candidates]
                    candidate_texts_by_batch = self.llm.generate_with_vllm(
                        nav_input["prompts"], 
                        num_candidates=self.n_candidates,  # 生成N个候选
                        max_gen_len=64, 
                        temperature=0.8,  # 使用较高温度确保多样性
                        use_beam_search=False  # 用采样而不是beam search
                    )

                    for i in range(batch_size):
                        print(f"candidate_texts_by_batch[{i}]: {candidate_texts_by_batch[i]} \n")
                        print(f"action_options_batch[{i}]: {nav_input['action_options'][i]} \n")
                        n_candidate_action_text, n_candidate_action_idx = self._extract_action_txt_Idx(n_candidate_outputs=candidate_texts_by_batch[i], nav_input=nav_input, bth_idx=i)
                        
                        # 并行计算所有样本的最佳候选
                        best_action_index = self._select_best_action(
                            n_candidate_outputs=n_candidate_action_text,
                            n_candidate_action_idx=n_candidate_action_idx,
                            batch_idex=i,
                            cand_inputs=cand_inputs,
                            nav_input=nav_input,
                            obs=obs,
                            t=t
                        )
                        best_action_index_by_batch.append(n_candidate_action_idx[best_action_index])
                        print(f"best_action_index_by_batch[{i}]: {best_action_index_by_batch[i]} \n")

                    a_t = best_action_index_by_batch 
                    print(f"a_t: {a_t} \n")

            else:
                assert False
            #?
            if self.feedback == 'teacher':
                a_t = nav_targets
            # else:
            #     print(self.feedback)
            #     sys.exit('Invalid feedback option')

            # Prepare environment action

            cpu_a_t = a_t
            for i, next_id in enumerate(cpu_a_t):
                if self.args.stop_first:
                    if next_id == 0 or next_id == self.args.ignoreid or ended[i] or (t == self.args.max_action_len - 1):  # The last action is <end>
                        cpu_a_t[i] = -1  # Change the <end> and ignore action to -1
                    else:
                        cpu_a_t[i] = cpu_a_t[i]-1
                else:
                    if next_id == (cand_inputs['cand_lens'][i]-1) or next_id == self.args.ignoreid or ended[i] or (t == self.args.max_action_len - 1):    # The last action is <end>
                        cpu_a_t[i] = -1             # Change the <end> and ignore action to -1


            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, traj)

            if self.feedback == 'teacher':
                obs = self.env._get_obs(t=t + 1)
            else:
                obs = self.env._get_obs(t=t + 1, shortest_teacher=True)

            previous_angle = [{'heading': ob['heading'],
                                   'elevation': ob['elevation']} for ob in obs]
            self.prompt_manager.make_history(a_t, nav_input, t)

            if isinstance(cpu_a_t, torch.Tensor):
                cpu_a_t = cpu_a_t.cpu()

            ended[:] = np.logical_or(ended, np.array([x == -1 for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            #self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.
            self.losses.append(self.loss / self.args.max_action_len)  # This argument is useless.

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if self.args.llm_predict:
            self.llm.eval()
        else:
            if use_dropout:
                self.vln_bert.train()
                self.critic.train()
            else:
                self.vln_bert.eval()
                self.critic.eval()
        if self.args.use_ig:
            super().test(iters=iters, rollout_function=self.rollout_ig)
        elif self.args.llm_predict:
            super().test(iters=iters, rollout_function=self.rollout_llm)
        else:
            super().test(iters=iters)

    def test_ig(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super().test(iters=iters, rollout_function=self.rollout_ig)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []

        if self.args.use_ig:
            rollout_function = self.rollout_ig
        else:
            rollout_function = self.rollout
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                rollout_function(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    rollout_function(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                rollout_function(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if self.args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']

            new_dict = {}
            for k,v in state_dict.items():
                if not k.startswith('vln_bert.apwig_head') and not k.startswith('vln_bert.mapwig_head'):
                    new_dict[k] = v
            state_dict = new_dict

            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                for k in load_keys:
                    if k not in model_keys:
                        print("missing in model keys", k)
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1

    def _compute_traditional_keyword_score(self, instruction, action_text):
        """
        传统的关键词匹配评分方法 (作为语义相似度计算的回退)
        """
        # 提取指令中的关键词（简化版本）
        instruction_keywords = set(instruction.lower().split())
        action_keywords = set(action_text.lower().split())
        
        # 定义重要的导航关键词权重
        important_keywords = {
            'bedroom', 'bathroom', 'kitchen', 'living', 'room', 'door', 'window',
            'left', 'right', 'forward', 'back', 'turn', 'go', 'enter', 'exit',
            'sink', 'bed', 'table', 'chair', 'sofa', 'desk', 'mirror',
            'first', 'second', 'third', 'last', 'next', 'previous'
        }
        
        # 计算关键词匹配分数
        matched_keywords = instruction_keywords.intersection(action_keywords)
        important_matches = matched_keywords.intersection(important_keywords)
        
        keyword_score = 0.0
        if instruction_keywords:
            keyword_score = len(matched_keywords) / len(instruction_keywords)
            # 重要关键词额外加权
            if important_matches:
                keyword_score += 0.2 * len(important_matches) / len(instruction_keywords)
        
        return keyword_score

    def _extract_action_txt_Idx(self, n_candidate_outputs, nav_input, bth_idx):
        n_cand_size = len(n_candidate_outputs)
        n_cand_action_txt = []
        n_cand_action_idx = []

        only_options_batch = nav_input["only_options"]

        for i in range(n_cand_size):
            output = n_candidate_outputs[i].strip()
            substr = "Action: "
            index = output.find(substr)
            if index == -1:
                cand_action_idx = random.randint(0, len(only_options_batch[bth_idx]) - 1)
                n_cand_action_idx.append(cand_action_idx)
                cand_action_txt = nav_input["only_actions"][bth_idx][cand_action_idx]
                n_cand_action_txt.append(cand_action_txt)
            else:
                option = output[index+8]

                if option in only_options_batch[bth_idx]:
                    cand_action_idx = only_options_batch[bth_idx].index(option)
                    cand_action_txt = nav_input["only_actions"][bth_idx][cand_action_idx]
                    n_cand_action_idx.append(cand_action_idx)
                    n_cand_action_txt.append(cand_action_txt)
                else:
                    cand_action_idx = random.randint(0, len(only_options_batch[bth_idx]) - 1)
                    cand_action_txt = nav_input["only_actions"][bth_idx][cand_action_idx]
                    n_cand_action_idx.append(cand_action_idx)
                    n_cand_action_txt.append(cand_action_txt)

        print(f"n_cand_action_txt: {n_cand_action_txt} \n")
        print(f"n_cand_action_idx: {n_cand_action_idx} \n")

        return n_cand_action_txt, n_cand_action_idx