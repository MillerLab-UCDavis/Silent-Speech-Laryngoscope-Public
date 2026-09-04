import os
import sys
import torch
from read_emg_patient import EMGDataset, SizeAwareSampler
from tqdm import tqdm
from neural_synthesis import ctc_utils
from neural_synthesis import utils
import librosa
from collections import defaultdict
import soundfile as sf
import argparse
import torch.nn.functional as F
# import wandb
import yaml
import numpy as np
import neural_synthesis as ns
from neural_synthesis.models import GSLM, Model # CnnRnnClassifier
from absl import flags
import neural_synthesis.models
from neural_synthesis import ctc_utils
from neural_synthesis import utils
import torchaudio.functional as FA
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length, combine_fixed_length_video
from tensorboardX import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import autocast, GradScaler
from torch import nn
# from gtts import gTTS

FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', 'Model', 'name of model to be used')
flags.DEFINE_string('experiment_name', 'SSI', 'experiment name')
flags.DEFINE_string('run_name', '1', 'run name')
flags.DEFINE_string('config', 'example_config.yaml', 'name of config file')
flags.DEFINE_string('root_dir', '/', 'path to root directory')
flags.DEFINE_bool('debug', False, 'Set to True to put in debug mode, which cycles through evaluation faster.')
flags.DEFINE_bool('chance', False, 'Set to True to train a model on noise inputs')
flags.DEFINE_string('subject', 'bravo3', 'subject')
flags.DEFINE_float('train_data_fraction', 1.0, 'Data fraction for training set')

# os.environ["WANDB_SILENT"] = "true"

# set global path to this repo
global_path = ""

# set path to fairseq installation
# This requires https://github.com/facebookresearch/fairseq on your machine from source
YOUR_FAIRSEQ_PATH = "fairseq/"

base_path = os.path.join(global_path, "")   ## data/pub_models/GSLM/
hubert_checkpoint_path = "hubert_base_ls960.pt"
glow_model_path = "waveglow_256channels_new.pt"
glow_directory = os.path.join(YOUR_FAIRSEQ_PATH, "examples/textless_nlp/gslm/unit2speech")
kmeans_model_path = "km_100.bin"
tts_model_path = "tts_checkpoint_best_100.pt"
code_dict_path = "code_dict_100"
gslm_device='cuda:0'
gslm = GSLM(YOUR_FAIRSEQ_PATH, base_path, hubert_checkpoint_path,kmeans_model_path,tts_model_path,glow_model_path,glow_directory,code_dict_path,device=gslm_device)
writer = SummaryWriter()
phoneme_inventory = ['aa','ae','ah','ao','aw','ax','axr','ay','b','ch','d','dh','dx','eh','el','em','en','er','ey','f','g','hh','hv','ih','iy','jh','k','l','m','n','nx','ng','ow','oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh','sil']



def adjust_learning_rate(optimizer, step, initial_lr, target_lr, warmup_steps):
    """Adjusts the learning rate according to the warmup schedule."""
    # Linear warmup
    lr = initial_lr + (target_lr - initial_lr) * min(1.0, step / warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_model(config, trainset, devset, device):

    """Train the model"""

    # define models, criterion, scheduler, and optimizer
    model_class = getattr(
        neural_synthesis.models,
        config.get("model_type", "Model"),
    )
    model = {
        "model": model_class(
            **config["model_params"],
        ).to(device)
    }
    optimizer_class = getattr(
        neural_synthesis.optimizers,
        config.get("model_optimizer_type", "RAdam"),
    )
    optimizer = {
        "model": optimizer_class(
            model["model"].parameters(),
            **config["model_optimizer_params"],
        )
    }
    initial_lr = optimizer['model'].param_groups[0]['lr']
    target_lr = 0.5e-4  # Target learning rate after warmup
    warmup_steps = 8000  # Number of warmup steps
    
    scheduler_class = getattr(
        torch.optim.lr_scheduler,
        config.get("model_scheduler_type", "StepLR"),
    )
    # scheduler_class = getattr(
    #     torch.optim.lr_scheduler,
    #     config.get("model_scheduler_type",\ "LROnPlateau"),
    # )
    scheduler = {
        "model": scheduler_class(
            optimizer=optimizer["model"],
            **config["model_scheduler_params"],
        )
    }
    
    scaler = GradScaler()


    dataloader = torch.utils.data.DataLoader(trainset, pin_memory=(device=='cuda'), collate_fn=trainset.collate_raw, num_workers=0, batch_sampler=SizeAwareSampler(trainset, 20000))  ## min batch_length = 200sec with 800 sampling rate
    n_epochs = 10000
    criterion = {}
    if (config['use_ctc_loss'], False):
        print("Using CTC Loss")
        criterion['ctc'] = F.ctc_loss
    else:
        config['use_ctc_loss'] = False
    print('Created model.')

    criterion['cel'] = nn.CrossEntropyLoss() ## Cross entropy loss
    # checkpointing and config save
    checkpoint_dir = os.path.join(os.getcwd(), 'torch_models', config['experiment_name'], config['run_name'])
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    checkpointer = utils.Checkpointer(checkpoint_dir)

    """ KS: 6
  bidirectional: true
  dropout: 0.7
  in_channels: 506
  keeptime: true
  n_classes: 101
  num_layers: 3
  rnn_dim: 260
  token_input: false"""
    
    print('Training model.')
    global_step = 0
    seq_len = 200
    total_train_loss = defaultdict(float)
    total_eval_loss = defaultdict(float)
    total_ctc_loss = defaultdict(float)
    total_cel_loss = defaultdict(float)
    eval_metrics = defaultdict(float)

    for epoch_idx in tqdm(range(n_epochs)):
        # losses = []
        # batch_idx = 0
        # print(global_step, " ", len(dataloader.dataset))
        for step, batch in enumerate(dataloader):
            # x = batch['emg_true']
            # print(f"Step: {step}, Batch size: {len(batch['emg'])}, {batch['idx']}")
            global_step += epoch_idx * len(batch) + step
            audio, _ = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['audio']])
            x_emg, _ = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']])
            # x_vid, _ = combine_fixed_length_video([t.to(device, non_blocking=True) for t in batch['video_frames']])
            x_emg = x_emg.float()
            # x_vid = x_vid.float()

            # print("Length of sEMG is: ", len(x_emg))
            # x_mask = x_mask.float()
            # video_mask = video_mask.float()
            audio = audio.float()

            hubert_encodings, _, units = gslm.encode(audio)
            # print(x.shape, hubert_encodings.shape, audio.shape)
            del audio
            
            y = units
            loss = 0.0

            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with record_function("model_inference"):
            # with autocast():
            y_dec, y_enc, _ = model['model'](x_emg, None, hubert_encodings, [], device)
            # print("y_enc shape : ", y_enc.shape, " y_dec shape : ", y_dec.shape)
            del hubert_encodings
            del x_emg
            # del x_vid

            # get sequences
            sequences = y.to(device=device, dtype=torch.int64)  # recall discrete units are 1-D so shape B x DU
            target_lengths = torch.full((sequences.shape[0],), sequences.shape[1]).to(device, dtype=torch.int64)
            blank = config['model_params']['num_outs'] - 1
            estimates = F.log_softmax(y_enc, dim=-1).permute((1, 0, 2)) # time*bs*101
            input_lengths = torch.full(size=(estimates.shape[1],), fill_value=estimates.shape[0],
                                        dtype=torch.int64).to(device)
            

            # compute ctc loss
            ctc_loss = criterion['ctc'](estimates, sequences, input_lengths, target_lengths, blank=blank,
                                        zero_infinity=True) * config['ctc_params']['loss_lambda'][0]
            
           
            
            y_dec = y_dec.permute(0, 2, 1)
            cross_entropy_loss = criterion['cel'](y_dec, sequences)

            alpha = 0.5
            loss = alpha*ctc_loss + (1-alpha)*cross_entropy_loss
            # loss = ctc_loss
            # loss = cross_entropy_loss

            total_train_loss["train/loss"] += loss
            total_train_loss['train/ctc_loss'] += ctc_loss.item()
            total_train_loss['train/cel_loss'] += cross_entropy_loss

            # Backpropogate and optimize
            optimizer["model"].zero_grad()
            loss.backward()
            # scaler.scale(loss).backward()
            if config["model_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model["model"].parameters(),
                    config["model_grad_norm"],
                )
            # scaler.step(optimizer["model"])
            # scaler.update()
            optimizer["model"].step()
            if config["model_scheduler_type"] == "ReduceLROnPlateau":
                scheduler["model"].step(loss)
            else:
                scheduler["model"].step()
            global_step += 1
            

            del units
            del y
            del y_enc
            del y_dec
            del sequences
            del target_lengths
            del input_lengths
            del estimates

            torch.cuda.empty_cache()

            if global_step < warmup_steps:
                adjust_learning_rate(optimizer["model"], global_step, initial_lr, target_lr, warmup_steps)

            
            # if(batch_idx % 100 == 0):
            #     print(optimizer['model'].param_groups[0]['lr'])
            #     optimizer['model'].param_groups[0]['lr'] *= 1.5
            
            # batch_idx += 1
        # print(global_step)
        print("EPOCH: ", epoch_idx, " COMPLETE")
        print("GLOBAL STEP: ", global_step)
        #######################
        #      Evaluation     #
        #######################
        # evaluate and save model
        # if global_step % config['steps_per_summary'] == 0:
        model['model'].eval()
        test_dataloader = torch.utils.data.DataLoader(devset, pin_memory=(device=='cuda'), collate_fn=devset.collate_raw, num_workers=0, batch_sampler=SizeAwareSampler(devset, 8000))  ## min batch_length = 200sec with 800 sampling rate
        eval_model(config, device, model, global_step, criterion, test_dataloader,
                                        total_eval_loss, eval_metrics)
        model['model'].train()
        # print(f"Epoch : {epoch_idx} train_loss : {total_train_loss['train/loss']} train_ctc_loss : {total_train_loss['train/ctc_loss']} train_cel_loss : {total_train_loss['train/cel_loss']} eval_loss : {total_eval_loss['test/loss']} eval_ctc_loss : {total_eval_loss['test/ctc_loss']} eval_cel_loss : {total_eval_loss['test/cel_loss']} CER : {total_eval_loss['test/cer']}")
        print(
                f"Epoch : {epoch_idx} \
                learning_rate : {optimizer['model'].param_groups[0]['lr']} \
                train_loss : {total_train_loss['train/loss']} \
                train_ctc_loss : {total_train_loss['train/ctc_loss']} \
                train_cel_loss : {total_train_loss['train/cel_loss']} \
                eval_loss : {total_eval_loss['test/loss']} \
                eval_ctc_loss : {total_eval_loss['test/ctc_loss']} \
                eval_cel_loss : {total_eval_loss['test/cel_loss']} \
                CER : {total_eval_loss['test/cer']}"
            )
        
        writer.add_scalar('train_loss', total_train_loss["train/loss"], epoch_idx)
        writer.add_scalar('train_ctc_loss', total_train_loss['train/ctc_loss'], epoch_idx)
        writer.add_scalar('train_cel_loss', total_train_loss['train/cel_loss'], epoch_idx)
        writer.add_scalar('test_loss', total_eval_loss["test/loss"], epoch_idx)
        writer.add_scalar('test_ctc_loss', total_eval_loss['test/ctc_loss'], epoch_idx)
        writer.add_scalar('test_cel_loss', total_eval_loss['test/cel_loss'], epoch_idx)
        writer.add_scalar('CER', total_eval_loss['test/cer'], epoch_idx)
        total_train_loss = defaultdict(float)
        total_eval_loss = defaultdict(float)
        eval_metrics = defaultdict(float)

        torch.save(model['model'].state_dict(), checkpointer(epoch_idx))

def eval_test_model(config, device, model, global_step, criterion, test_dataloader, total_eval_loss, eval_metrics,
               encoder_config=None):
    
    for eval_steps_per_epoch, test_batch in enumerate(tqdm(test_dataloader), 1):
        # print(len(test_batch['audio']))
        # eval one step
        seq_len = 200

        audio, _ = combine_fixed_length([t.to(device, non_blocking=True) for t in test_batch['audio']])
        x_emg, _ = combine_fixed_length([t.to(device, non_blocking=True) for t in test_batch['emg']])
        x_vid, _ = combine_fixed_length_video([t.to(device, non_blocking=True) for t in test_batch['video_frames']])
            
        # x, x_mask = combine_fixed_length([t.to(device, non_blocking=True) for t in test_batch['emg_true']])
        x_emg = x_emg.float()
        x_vid = x_vid.float()
        audio = audio.float()
        # x_mask = x_mask.float()
        # print("x shape: ", x.shape)
        # if len(x.shape) == 2:
        #     x = torch.unsqueeze(x, 2)
        hubert_encodings, _, units = gslm.encode(audio)

        # print("units shape : ", units.shape)

        # y_ = model["model"]([], x, [], padding_mask = x_mask)
        y_dec, y_enc, _ = model['model'](x_emg, x_vid, hubert_encodings, [], device, train=False)
        
        # estimates_enc = F.log_softmax(y_enc, dim=-1)
        # decoder = ctc_utils.BeamDecoder(None)
        
        decoder = ctc_utils.Decoder(blank_index=100, silent=[None], remove_rep=True)

        probabilities =  F.softmax(y_dec, dim=1)
        estimates_dec = torch.argmax(probabilities, dim=-1)

        # seq_lengths = [estimates_enc.shape[1] for _ in range(estimates_enc.shape[0])]
        # estimated_seq = decoder.decode(estimates_enc, seq_lengths)

        estimated_sequences = torch.squeeze(torch.argmax(y_enc, dim=-1))
        estimated_seq = [torch.Tensor(decoder.process_list(estimated_sequences.detach().cpu().numpy().tolist()))]
        #
        # print(len(estimated_seq), estimated_seq[0])
        # return 

        # print(len(estimated_seq), estimated_seq[0])
        # temp = [len(estimated_seq[i]) for i in range(len(estimated_seq))]
        # print(temp)
        # lengths = [x*estimated_seq[0] for x in lengths]
        
        # estimated_seq = torch.tensor(estimated_seq)
        # lengths = [1 for _ in range len(estimated_seq)]
        # test_seq = decollate_tensor(estimated_seq, lengths)
        # print(len(test_seq), test_seq[0])
        # print(len(test_seq), test_seq[0])
        # estimated_seq = decoder.decode(estimates, seq_lengths)
        # print(len(estimated_seq), estimated_seq[0])

        # estimates = F.log_softmax(y_, dim=-1).permute((1, 0, 2))
        # estimated_seq = torch.argmax(estimates, dim=-1)

        # estimated_seq = eval_step(config, criterion, device, model, total_eval_loss, test_batch, eval_metrics)
        
        # print(estimated_seq)
        
        # print("length of generated audios ", len(gen_audio))
        # print("Number of test seq are: ",len(test_seq))
        for i in range(len(estimated_seq)):
            # curr_seq_enc = torch.tensor([estimated_seq[i]])
            curr_seq_enc = estimated_seq[i].unsqueeze(0).clone().detach().to(torch.int64)
            curr_seq_dec = estimates_dec[i].unsqueeze(0).clone().detach()

            # curr_seq_enc = torch.tensor(estimated_seq[i].unsqueeze(0).to(torch.int64))
            # curr_seq_dec = torch.tensor(estimates_dec[i]).unsqueeze(0)
            # print(curr_seq_dec)
            # print("enc ",curr_seq_enc, curr_seq_enc.shape)
            # print("dec ",curr_seq_dec, curr_seq_dec.shape)
            # _, audio = gslm.decode(torch.unsqueeze(torch.squeeze(seq),0).to(torch.int64), return_wave = True)
            _, gen_audio_enc = gslm.decode(curr_seq_enc, True)
            _, gen_audio_dec = gslm.decode(curr_seq_dec, True)

            output_path_dec = 'output_audios/multimodal/jun_25_patient/dec/'+test_batch['output_file_path'][i]
            output_path_enc = 'output_audios/multimodal/jun_25_patient/enc/'+test_batch['output_file_path'][i]
            file_label = test_batch['file_label'][i]
            # output_path = 'output_audios_closed/'+test_batch['output_file_path'][i]
            if not os.path.exists(output_path_dec):
                os.makedirs(output_path_dec)
                print(f"Directory '{output_path_dec}' created.")
            else:
                print(f"Directory '{output_path_dec}' already exists.")

            if not os.path.exists(output_path_enc):
                os.makedirs(output_path_enc)
                print(f"Directory '{output_path_enc}' created.")
            else:
                print(f"Directory '{output_path_enc}' already exists.")

            # output_path += f"/{eval_steps_per_epoch}_{i}_audio.wav"
            output_path_enc += f"/{file_label}_audio.wav"
            output_path_dec += f"/{file_label}_audio.wav"
            # audio = FA.resample(torch.Tensor(np.squeeze(gen_audio[0]).cpu().numpy()), 22050, 16000).cpu().numpy().astype(np.float32)
            output_audio_enc = gen_audio_enc[0].squeeze().cpu().numpy().astype('float32')
            output_audio_dec = gen_audio_dec[0].squeeze().cpu().numpy().astype('float32')

            sf.write(output_path_enc, output_audio_enc, 22050)
            sf.write(output_path_dec, output_audio_dec, 22050)

        
        # if eval_steps_per_epoch > config['steps_within_evaluation']:
        #     break
    return eval_steps_per_epoch

def eval_model(config, device, model, global_step, criterion, test_dataloader, total_eval_loss, eval_metrics,
               encoder_config=None):
    
    with torch.no_grad():
    
        for eval_steps_per_epoch, test_batch in enumerate(tqdm(test_dataloader), 1):

            # eval one step
            eval_step(config, criterion, device, model, total_eval_loss, test_batch, eval_metrics)
            torch.cuda.empty_cache()
            
            # if eval_steps_per_epoch > config['steps_within_evaluation']:
            #     break

        return 


def eval_step(config, criterion, device, model, total_eval_loss, test_batch, eval_metrics):
    """Evaluate model one step."""

    seq_len = 200

    # scaler = GradScaler()

    audio, _ = combine_fixed_length([t.to(device, non_blocking=True) for t in test_batch['audio']])
    x_emg, _ = combine_fixed_length([t.to(device, non_blocking=True) for t in test_batch['emg']])
    # x_vid, _ = combine_fixed_length_video([t.to(device, non_blocking=True) for t in test_batch['video_frames']])
    x_emg = x_emg.float()
    # x_vid = x_vid.float()
    # video_mask = video_mask.float()
    # x_mask = x_mask.float()
    # tensor_audio = torch.tensor(audio)
    # tensor_audio = tensor_audio.unsqueeze(0).cuda()
    audio = audio.float()

    hubert_encodings, _, units = gslm.encode(audio)
    del audio

    # print(x.shape, hubert_encodings.shape)

    y = units
    loss = 0.0

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    # with autocast():
    # y_dec, y_enc = model["model"](x, x_video, [], device, padding_mask = x_mask, video_mask = video_mask)
    y_dec, y_enc, _ = model['model'](x_emg, None, hubert_encodings, [], device, train=False)
    # print(y_.shape)
    del hubert_encodings

    # save losses
    loss = 0.0
    if config['use_ctc_loss']:

        # get sequences
        sequences = y.to(device=device, dtype=torch.int64)  # recall discrete units are 1-D so shape B x DU
        target_lengths = torch.full((sequences.shape[0],), sequences.shape[1]).to(device, dtype=torch.int64)
        blank = config['model_params']['num_outs'] - 1
        estimates = F.log_softmax(y_enc, dim=-1).permute((1, 0, 2))
        input_lengths = torch.full(size=(estimates.shape[1],), fill_value=estimates.shape[0], dtype=torch.int64).to(
            device)
        ctc_loss = criterion['ctc'](estimates, sequences, input_lengths, target_lengths, blank=blank,
                                    zero_infinity=True) * config['ctc_params']['loss_lambda'][0]


        y_dec = y_dec.permute(0, 2, 1)
        cross_entropy_loss = criterion['cel'](y_dec, sequences)

        alpha = 0.5
        loss = alpha*ctc_loss + (1-alpha)*cross_entropy_loss
        # loss = ctc_loss
        # loss = cross_entropy_loss

        # # compute ctc loss
        ctc_decoder = ctc_utils.Decoder(blank_index=blank, silent=[None], remove_rep=True)
        estimated_sequences = torch.argmax(estimates, dim=-1)

        #
        cer = ctc_decoder.phone_word_error(estimated_sequences.T, sequences)
        total_eval_loss["test/cer"] += cer

    total_eval_loss["test/loss"] += loss
    total_eval_loss["test/ctc_loss"] += ctc_loss.item()
    total_eval_loss["test/cel_loss"] += cross_entropy_loss

    del y
    del units
    del y_enc
    del y_dec
    del sequences
    del target_lengths
    del estimates
    del input_lengths
    del estimated_sequences
    del x_emg
    # del x_vid

    torch.cuda.empty_cache()
    return 


def test_model(config, testset, device):

    """Train the model"""

    # define models, criterion, scheduler, and optimizer
    model_class = getattr(
        neural_synthesis.models,
        config.get("model_type", "Model"),
    )
    model = {
        "model": model_class(
            **config["model_params"],
        ).to(device)
    }
    optimizer_class = getattr(
        neural_synthesis.optimizers,
        config.get("model_optimizer_type", "RAdam"),
    )
    optimizer = {
        "model": optimizer_class(
            model["model"].parameters(),
            **config["model_optimizer_params"],
        )
    }
    scheduler_class = getattr(
        torch.optim.lr_scheduler,
        config.get("model_scheduler_type", "StepLR"),
    )
    # scheduler_class = getattr(
    #     torch.optim.lr_scheduler,
    #     config.get("model_scheduler_type", "LROnPlateau"),
    # )
    scheduler = {
        "model": scheduler_class(
            optimizer=optimizer["model"],
            **config["model_scheduler_params"],
        )
    }  

    criterion = {}
    if (config['use_ctc_loss'], False):
        print("Using CTC Loss")
        criterion['ctc'] = F.ctc_loss
    else:
        config['use_ctc_loss'] = False

    dataloader = torch.utils.data.DataLoader(testset, pin_memory=(device=='cuda'), collate_fn=testset.collate_raw, num_workers=0, batch_sampler=SizeAwareSampler(testset, 8000))  ## min batch_length = 200sec with 800 sampling rate
    
    total_eval_loss = defaultdict(float)
    eval_metrics = defaultdict(float)

    # model = CnnRnnClassifier()
    model['model'].load_state_dict(torch.load('torch_models/ssi/patient_multimodal_jun_25_run1/model_125.pt'))
    model['model'].eval()
    model['model'].to(device)
    
    eval_test_model(config, device, model, 0, criterion, dataloader, total_eval_loss, eval_metrics,
               encoder_config=None)


def main():
    
    # choose device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # load config file
    with open(utils.get_config_file(FLAGS.config)) as f:
        experiment_config = yaml.load(f, Loader=yaml.Loader)
    with open(utils.get_config_file("default_config.yaml")) as f:
        default_config = yaml.load(f, Loader=yaml.Loader)
    config = utils.merge_yaml_configs(experiment_config, default_config)
    if FLAGS.train_data_fraction < 1.0:
        config["dataloader_params"]["train_data_fraction"] = FLAGS.train_data_fraction
    # config.update(vars(args))

    config.update({
        "model_type": FLAGS.model_type,
        "experiment_name": FLAGS.experiment_name,
        "run_name": FLAGS.run_name,
        "config": FLAGS.config,
        "root_dir": FLAGS.root_dir,
        "debug": FLAGS.debug,
        "chance": FLAGS.chance,
        "subject": FLAGS.subject,
        "train_data_fraction": FLAGS.train_data_fraction,
    })


    if config['debug']:
        config['steps_within_evaluation'] = 5
        config['steps_per_summary'] = 5
        config['outdir'] = 'debug'
        config['num_save_intermediate_results'] = 4
        device = torch.device('cpu')

    # initialize wandb
    # if config['use_wandb']:
    #     wandb.init(project=args.experiment_name, config=config, name=args.run_name)

    
    # load and build dataset
    print(config["dataloader_params"]['input_types'])
    # print(config['train_filename'])
    # print(config['test_filename'])
    print(config['subject'])
    trainset = EMGDataset(test=False, dev=False)
    testset = EMGDataset(test=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(config, trainset, testset, device)
    # test_model(config, testset, device)

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()


#             emg = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg_true']], seq_len*8)
#             audio = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['audio']], seq_len*128)
#             audio = audio.float()
#             feat_chunk, one_hot, unit = gslm.encode(audio)

#             _, gen_audio = gslm.decode(unit, True)

#             # print(emg.shape, audio.shape, gen_audio[0].shape)
#             # print(batch['audio_lengths'])
#             # break

#             # output_audio = gen_audio[5].squeeze().cpu().numpy().astype('float32')
#             # input_audio = audio[5].squeeze().cpu().numpy().astype('float32')

#             # sf.write('new_output{}.wav'.format(count), output_audio, 22050)
#             # sf.write('new_input{}.wav'.format(count), input_audio, 16000)

#             count+=1
#             if(count == 20):
#                 break
#         if(count == 20):
#                 break