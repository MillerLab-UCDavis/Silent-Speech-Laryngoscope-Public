import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
# import torchvision.models as models
from neural_synthesis.models.SE_ResNet import se_resnet34
from neural_synthesis.models.transformer import TransformerEncoderLayer, TransformerDecoderLayer, MultiHeadAttention
import warnings
warnings.filterwarnings('ignore')
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_size', 512, 'number of hidden dimensions')
flags.DEFINE_integer('decoder_feat_dim', 768, 'decoder feature dimension')
flags.DEFINE_integer('video_feature_size', 512, 'number of hidden dimensions')
flags.DEFINE_integer('num_enc_layers', 5, 'number of encoder layers')
flags.DEFINE_integer('num_dec_layers', 5, 'number of decoder layers')
flags.DEFINE_float('dropout', .7, 'dropout')
flags.DEFINE_float('beta', 1 / np.sqrt(2), 'beta')

class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(4, FLAGS.model_size, 2, dropout_p=FLAGS.dropout),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2, dropout_p=FLAGS.dropout),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2, dropout_p=FLAGS.dropout),
        )

        self.decoder_cnn_block = DecoderCNNBlock(FLAGS.model_size, FLAGS.decoder_feat_dim, dropout=FLAGS.dropout)

        self.video_model = VideoModel()

        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)
        # self.w_decoder_in = nn.Linear(FLAGS.decoder_feat_dim, FLAGS.model_size)
        self.w_decoder_in = nn.Linear(FLAGS.decoder_feat_dim, FLAGS.model_size)
        self.w_vid_in = nn.Linear(FLAGS.video_feature_size, FLAGS.model_size)
        self.emg_to_common_space = nn.Linear(4, FLAGS.model_size)

        self.n_heads = 4

        encoder_layer = TransformerEncoderLayer(d_model=FLAGS.model_size, nhead=self.n_heads, relative_positional=True, relative_positional_distance=100, dim_feedforward=1024, dropout=FLAGS.dropout)
        decoder_layer = TransformerDecoderLayer(d_model=FLAGS.model_size, n_head=self.n_heads, drop_prob=FLAGS.dropout)
        self.emg_vid_cross_attention = MultiHeadAttention(d_model=FLAGS.model_size, n_head=self.n_heads, relative_positional=False, dropout=FLAGS.dropout)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, FLAGS.num_enc_layers)
        self.transformerDecoder = nn.TransformerDecoder(decoder_layer, FLAGS.num_dec_layers)
        self.w_out = nn.Linear(FLAGS.model_size, num_outs)
        self.softmax = nn.Softmax(dim = -1)
        self.src_pad_idx = 0
        
        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)

    def forward(self, x_emg, x_vid, tgt, session_ids, device, train=True):
        '''
            x_emg: shape(bsxtxc)
            x_vid: shape(bsxtxcxwxh)
        '''

        # if self.training:
        #     r = random.randrange(20)
        #     if r > 0:
        #         x_emg[:,:,:-r] = x_emg[:,:, r:] # shift left r
        #         x_emg[:,:,-r:] = 0
                # x_emg[:,:-r,:] = x_emg[:,r:,:] # shift left r
                # x_emg[:,-r:,:] = 0

        ## Converting sEMG to time series of feature vectors
        x_emg = x_emg.permute(0, 2, 1) # put channel before time for conv
        intermediate_activations = []

        for conv_block in self.conv_blocks:
            x_emg = conv_block(x_emg)
            intermediate_activations.append(x_emg)

        # x_emg = self.conv_blocks(x_emg)
        x_emg = x_emg.permute(0, 2, 1)
        x_emg = F.relu(self.w_raw_in(x_emg)) # [bs*time*feat_dim]
        
        # emg_mask = self.make_src_mask(x_emg, device)  
        x_emg = x_emg.permute(1, 0, 2) # [time*bs*feat_dim]
        x_emg = self.transformerEncoder(x_emg, mask=None)

        # # Converting video frames to time series of feature vectors
        # x_vid = x_vid.permute(0, 2, 1, 3, 4) # bsxcxtxwxh
        # x_vid = self.video_model(x_vid)
        # x_vid = self.w_vid_in(x_vid) 
        # # vid_mask = self.make_src_mask(x_vid, device) # tgt [bs*time*feat_enc]
        # x_vid = x_vid.permute(1, 0, 2)
        # # x_vid = self.transformerEncoder(x_vid, mask=None)

        # # sEMG and video cross attention layer
        # x_emg_vid = self.emg_vid_cross_attention(x_emg, y=x_vid)
        
        # alpha = 1.0
        # inp = alpha*x_emg_vid + (1-alpha)*x_emg
        inp = x_emg

        # Shift 1 time step for tgt

        # tgt = tgt.permute(0, 2, 1)
        # tgt = self.decoder_cnn_block(tgt)
        # tgt = tgt.permute(0, 2, 1)

        # if train:
        tgt = self.w_decoder_in(tgt)
        tgt = F.relu(tgt)

        new_timestep = torch.zeros(tgt.shape[0], 1, tgt.shape[2], device=tgt.device)
        tgt = torch.cat([new_timestep, tgt[:, :-1, :]], dim=1)

        trg_mask = self.make_trg_mask(tgt, 0.15)
        
        tgt = tgt.transpose(0,1) # put time first [time, bs, feat_dim]
        
        tgt = self.transformerDecoder(tgt, inp, trg_mask, memory_mask=None)  
        x_dec = tgt.transpose(0,1)   ## [bs, time, feat_dim]

        # else:
        #     # Start with a tensor representing the start token, which should have the same size as an embedding vector.
        #     max_length = tgt.shape[1]
        #     start_token = torch.zeros(1, 1, FLAGS.model_size, device=inp.device)
        #     tgt = start_token
        #     # tgt = start_token.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, feature_dim]
            
        #     outputs = []
        #     for _ in range(max_length):
        #         with torch.no_grad():
        #             # Prepare input for the current step
        #             # tgt_in = self.w_decoder_in(tgt)
        #             tgt_in = tgt
                    
        #             trg_mask = self.make_trg_mask(tgt_in, 0.15)  # Recreate target mask for current sequence length
        #             tgt_in = tgt_in.transpose(0, 1)  # Transformer expects [seq_len, batch, feature_dim]

        #             # print("target input ", tgt_in.shape)
        #             decoder_output = self.transformerDecoder(tgt_in, inp, trg_mask, memory_mask=None)
        #             # print("decoder output ", decoder_output.shape)
        #             next_token = decoder_output[-1].unsqueeze(1)  # Output from the last timestep
        #             # print("Next Token ", next_token.shape)
        #             outputs.append(next_token)

        #             # print("tgt size:", tgt.size())
        #             # print("next_token size:", next_token.size())

        #             tgt = torch.cat([tgt, next_token], dim=1)

        #             del tgt_in, trg_mask, decoder_output, next_token
        #             torch.cuda.empty_cache()

        #     x_dec = torch.cat(outputs, dim=1)


        # x_dec = inp.transpose(0, 1)     
        x_enc = inp.transpose(0, 1)

        # del trg_mask

        torch.cuda.empty_cache()

        return self.w_out(x_dec), self.w_out(x_enc), intermediate_activations
        
    def make_src_mask(self, src, device):
        bs, time, _ = src.shape
        src_mask = torch.any(src != 0, dim=-1).unsqueeze(1).expand(bs, self.n_heads, time).to(device)
        return src_mask
    
    def make_trg_mask(self, trg, mask_prob=0.15, device='cuda'):
        bs, time_steps, _ = trg.shape
        
        lookahead_mask = (1 - torch.triu(torch.ones((time_steps, time_steps), device=device), diagonal=1)).bool()

        random_mask = torch.ones_like(lookahead_mask, device=device).bool()

        for i in range(time_steps):
            allowed_timesteps = i + 1  
            num_masked_tokens = int(mask_prob * allowed_timesteps)
            
            if allowed_timesteps > 1 and num_masked_tokens > 0:
                masked_indices = torch.randperm(allowed_timesteps, device=device)[:num_masked_tokens]
                random_mask[i, masked_indices] = False
        
        combined_mask = lookahead_mask & random_mask
        # combined_mask = lookahead_mask

        combined_mask = combined_mask.unsqueeze(0).expand(bs, self.n_heads, -1, -1)
        
        return combined_mask


class DecoderCNNBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        # self.ln1 = nn.LayerNorm([num_outs, None])
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        # self.ln2 = nn.LayerNorm([num_outs, None])
        self.bn2 = nn.BatchNorm1d(num_outs)
        self.conv3 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        # self.ln3 = nn.LayerNorm(num_outs)
        self.bn3 = nn.BatchNorm1d(num_outs)

        self.dropout = nn.Dropout(dropout)  

    def forward(self, x):

        x = self.conv1(x) 
        # x = x.permute(0, 2, 1)
        x = self.bn1(x)
        # x = x.permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        # x = x.permute(0, 2, 1)
        x = self.bn2(x)
        # x = x.permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        # x = x.permute(0, 2, 1)
        x = self.bn3(x)
        # x = x.permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1, dropout_p=0.1, beta = 1.0):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.ln1 = nn.LayerNorm(num_outs)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.ln2 = nn.LayerNorm(num_outs)
        self.bn2 = nn.BatchNorm1d(num_outs)
        self.dropout = nn.Dropout(dropout_p)  
        self.act1 = nn.GELU()
        self.act2 = nn.ReLU()
        self.beta = beta

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            # self.res_norm = nn.BatchNorm1d(num_outs)
            self.res_norm = nn.LayerNorm(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        # input_value = x

        # x = self.act2(self.conv1(x))
        # # x = self.dropout(x)
        # x = self.conv2(x) * self.beta

        # if self.residual_path is not None:
        #     res = self.residual_path(input_value)
        # else:
        #     res = input_value

        # out = self.act2(x + res)

        # return out
    

        input_value = x
        x = self.act2(self.ln1(self.conv1(x).transpose(1, 2))).transpose(1, 2)  # Apply LayerNorm after conv1
        x = self.ln2(self.conv2(x).transpose(1, 2)).transpose(1, 2) * self.beta  # Apply LayerNorm after conv2

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value).transpose(1, 2)).transpose(1, 2)  # Apply LayerNorm on residual path
        else:
            res = input_value

        out = self.act2(x + res)
        return out

class VideoModel(nn.Module):
    
    def __init__(self, dropout_p=0.7):
        super(VideoModel, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.batch_norm1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.batch_norm2 = nn.BatchNorm3d(64)

        # self.resnet34 = se_resnet34(n_input_channels=64)

        self.conv3 = nn.Conv3d(64, 128, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.batch_norm3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.batch_norm4 = nn.BatchNorm3d(256)

        self.dropout_p = dropout_p
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)        
        self.dropout3d = nn.Dropout3d(self.dropout_p) 

        self._init()

    def _init(self):
        
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)

    def forward(self, x):
            
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool2(x)
        
        # x = self.resnet34(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool4(x)

        print(x.shape)
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)


        x = x.permute(1, 0, 2) # (B, T, C*H*W)

        return x