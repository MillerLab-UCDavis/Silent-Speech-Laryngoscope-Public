import os
import sys
import torch
from read_emg import EMGDataset, SizeAwareSampler
from tqdm import tqdm
from neural_synthesis import ctc_utils
from neural_synthesis import utils
import librosa
from collections import defaultdict
import soundfile as sf
import argparse
import torch.nn.functional as F
import wandb
import yaml
import numpy as np
import neural_synthesis as ns
from neural_synthesis.models import GSLM, CnnRnnClassifier
from absl import flags
import neural_synthesis.models
from neural_synthesis import ctc_utils
from neural_synthesis import utils
import torchaudio.functional as FA
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length, combine_fixed_length_for_training
from tensorboardX import SummaryWriter
FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', 'CnnRnnClassifier', 'name of model to be used')
flags.DEFINE_string('experiment_name', 'SSI', 'experiment name')
flags.DEFINE_string('run_name', '1', 'run name')
flags.DEFINE_string('config', 'example_config.yaml', 'name of config file')
flags.DEFINE_string('root_dir', '/', 'path to root directory')
flags.DEFINE_bool('debug', False, 'Set to True to put in debug mode, which cycles through evaluation faster.')
flags.DEFINE_bool('chance', False, 'Set to True to train a model on noise inputs')
flags.DEFINE_string('subject', 'bravo3', 'subject')
flags.DEFINE_float('train_data_fraction', 1.0, 'Data fraction for training set')

os.environ["WANDB_SILENT"] = "true"

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

def train_model(config, trainset, devset, device):

    """Train the model"""

    # define models, criterion, scheduler, and optimizer
    model_class = getattr(
        neural_synthesis.models,
        config.get("model_type", "CnnRnnClassifier"),
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
    scheduler = {
        "model": scheduler_class(
            optimizer=optimizer["model"],
            **config["model_scheduler_params"],
        )
    }
    
    dataloader = torch.utils.data.DataLoader(trainset, pin_memory=(device=='cuda'), collate_fn=trainset.collate_raw, num_workers=0, batch_sampler=SizeAwareSampler(trainset, 160000))  ## min batch_length = 200sec with 800 sampling rate
    n_epochs = 10000
    criterion = {}
    if (config['use_ctc_loss'], False):
        print("Using CTC Loss")
        criterion['ctc'] = F.ctc_loss
    else:
        config['use_ctc_loss'] = False
    print('Created model.')
    # model = CnnRnnClassifier(260, 6, 3, 0.7, 101, True, 506, True, False)

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
    local_train_step = 0
    total_train_loss = defaultdict(float)
    total_eval_loss = defaultdict(float)
    eval_metrics = defaultdict(float)

    for epoch_idx in tqdm(range(n_epochs)):
        # losses = []
        for batch in dataloader:
            # x = batch['emg_true']
            audio, _ = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['audio']])
            x, x_mask = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg_true']])
            x = x.float()
            x_mask = x_mask.float()
            # print("emg and audio shape: ", x.shape, audio.shape)
            # tensor_audio = torch.tensor(audio)
            # tensor_audio = tensor_audio.unsqueeze(0).cuda()
            audio = audio.float()

            # if x.shape[0] != audio.shape[0]:
            #     if x.shape[0] > audio.shape[0]:
            #         diff = x.shape[0] - audio.shape[0]
            #         x = x[:-diff]
            #     else:
            #         diff = audio.shape[0] - x.shape[0]
            #         audio = audio[:-diff]


            _, _, units = gslm.encode(audio)
            
            y = units
            loss = 0.0
            # masks = batch['masks']

            y_ = model['model'](x, padding_mask = x_mask)

            # print("Prediction: ", y_[0], "Ground truth: ", y[0])

            # print(x.shape, audio.shape, y.shape, y_.shape, x[0].shape, audio[0].shape, units.shape)

            # get sequences
            sequences = y.to(device=device, dtype=torch.int64)  # recall discrete units are 1-D so shape B x DU
            target_lengths = torch.full((sequences.shape[0],), sequences.shape[1]).to(device, dtype=torch.int64)
            blank = config['model_params']['n_classes'] - 1
            estimates = F.log_softmax(y_, dim=-1).permute((1, 0, 2))
            input_lengths = torch.full(size=(estimates.shape[1],), fill_value=estimates.shape[0],
                                        dtype=torch.int64).to(device)
            # print("seq and est ", sequences.shape, estimates.shape)
            # compute ctc loss
            ctc_loss = criterion['ctc'](estimates, sequences, input_lengths, target_lengths, blank=blank,
                                        zero_infinity=True) * config['ctc_params']['loss_lambda'][0]
            loss += ctc_loss
            total_train_loss["train/pho_ctc_loss"] += ctc_loss.item()

            total_train_loss["train/loss"] += loss

            # Backpropogate and optimize
            optimizer["model"].zero_grad()
            loss.backward()
            if config["model_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model["model"].parameters(),
                    config["model_grad_norm"],
                )
            optimizer["model"].step()
            if config["model_scheduler_type"] == "ReduceLROnPlateau":
                scheduler["model"].step(loss)
            else:
                scheduler["model"].step()
            global_step += 1
            local_train_step += 1
            # print("EPOCH: ", epoch_idx, " STEP: ", global_step, " COMPLETE")

        print("EPOCH: ", epoch_idx, " COMPLETE")
        #######################
        #      Evaluation     #
        #######################
        # evaluate and save model
        # if global_step % config['steps_per_summary'] == 0:
        model['model'].eval()
        test_dataloader = torch.utils.data.DataLoader(devset, pin_memory=(device=='cuda'), collate_fn=devset.collate_raw, num_workers=0, batch_sampler=SizeAwareSampler(devset, 16000))  ## min batch_length = 200sec with 800 sampling rate
        eval_model(config, device, model, global_step, criterion, test_dataloader,
                                        total_eval_loss, eval_metrics)
        model['model'].train()
        # if config['use_wandb']:
        #     for loss_name, loss_val in total_train_loss.items():
        #         wandb.log({loss_name: loss_val}, step=global_step, commit=False)
        #     for loss_name, loss_val in total_eval_loss.items():
        #         wandb.log({loss_name: loss_val}, step=global_step, commit=False)
        #     for metric_name, metric_val in eval_metrics.items():
        #         wandb.log({metric_name: metric_val.mean()}, step=global_step, commit=False)
        print(f"Epoch : {epoch_idx} train_loss : {total_train_loss['train/loss']} eval_loss : {total_eval_loss['test/loss']} CER : {total_eval_loss['test/cer']}")
        writer.add_scalar('train_loss', total_train_loss["train/loss"], epoch_idx)
        writer.add_scalar('test_loss', total_eval_loss["test/loss"], epoch_idx)
        writer.add_scalar('CER', total_eval_loss['test/cer'], epoch_idx)
        total_train_loss = defaultdict(float)
        total_eval_loss = defaultdict(float)
        eval_metrics = defaultdict(float)
        # torch.save(model['model'].state_dict(), checkpointer(global_step))
        torch.save(model['model'].state_dict(), checkpointer(epoch_idx))
        local_train_step = 0

def eval_test_model(config, device, model, global_step, criterion, test_dataloader, total_eval_loss, eval_metrics,
               encoder_config=None):
    
    for eval_steps_per_epoch, test_batch in enumerate(tqdm(test_dataloader), 1):
        # eval one step
        seq_len = 200
        x, x_mask = combine_fixed_length([t.to(device, non_blocking=True) for t in test_batch['emg_true']])
        x = x.float()
        x_mask = x_mask.float()
        # print("x shape: ", x.shape)
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 2)

        y_ = model["model"](x, padding_mask = x_mask)
        estimates = F.log_softmax(y_, dim=-1)
        decoder = ctc_utils.BeamDecoder(None)
        # print("output from model : ", y_.shape, lengths)

        # ctc_decoder = ctc_utils.Decoder(blank_index=100, silent=[None], remove_rep=True)

        # estimated_sequences = torch.squeeze(torch.argmax(y_, dim=-1))
        # print(len(estimated_sequences.tolist()))
        # seq = [ctc_decoder.process_list(l) for l in estimated_sequences.detach().cpu().numpy().tolist()]

        # print(estimates.shape)
        # print("NEWWWWWW : ", len(seq), seq[0])

        seq_lengths = [estimates.shape[1] for _ in range(estimates.shape[0])]
        estimated_seq = decoder.decode(estimates, seq_lengths)
        print(len(estimated_seq), estimated_seq[0])
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
            curr_seq = torch.tensor([estimated_seq[i]])
            # _, audio = gslm.decode(torch.unsqueeze(torch.squeeze(seq),0).to(torch.int64), return_wave = True)
            _, gen_audio = gslm.decode(curr_seq, True)
            output_path = 'output_audios/18/'+test_batch['output_file_path'][i]
            file_label = test_batch['file_label'][i]
            # output_path = 'output_audios_closed/'+test_batch['output_file_path'][i]
            if not os.path.exists(output_path):
                # If it doesn't exist, create it
                os.makedirs(output_path)
                print(f"Directory '{output_path}' created.")
            else:
                print(f"Directory '{output_path}' already exists.")
            # output_path += f"/{eval_steps_per_epoch}_{i}_audio.wav"
            output_path += f"/{file_label}_audio.wav"
            # audio = FA.resample(torch.Tensor(np.squeeze(gen_audio[0]).cpu().numpy()), 22050, 16000).cpu().numpy().astype(np.float32)
            output_audio = gen_audio[0].squeeze().cpu().numpy().astype('float32')

            sf.write(output_path, output_audio, 22050)

        
        if eval_steps_per_epoch > config['steps_within_evaluation']:
            break
    return eval_steps_per_epoch

def eval_model(config, device, model, global_step, criterion, test_dataloader, total_eval_loss, eval_metrics,
               encoder_config=None):
    
    for eval_steps_per_epoch, test_batch in enumerate(tqdm(test_dataloader), 1):

        # eval one step
        estimated_seq = eval_step(config, criterion, device, model, total_eval_loss, test_batch, eval_metrics)

        
        if eval_steps_per_epoch > config['steps_within_evaluation']:
            break

    return estimated_seq


def eval_step(config, criterion, device, model, total_eval_loss, test_batch, eval_metrics):
    """Evaluate model one step."""

    # get data and run inference
    # x = utils.repackage_data(test_batch, config['dataloader_params']['input_types'], device, noise=config['chance'])
    # y = utils.repackage_data(test_batch, config['dataloader_params']['output_types'], device)
    # x = utils.tuple_to_obj(x)
    # y = utils.tuple_to_obj(y)
    seq_len = 200
    audio, _ = combine_fixed_length([t.to(device, non_blocking=True) for t in test_batch['audio']])
    x, x_mask = combine_fixed_length([t.to(device, non_blocking=True) for t in test_batch['emg_true']])
    x = x.float()
    x_mask = x_mask.float()
    # tensor_audio = torch.tensor(audio)
    # tensor_audio = tensor_audio.unsqueeze(0).cuda()
    audio = audio.float()

    # if x.shape[0] != audio.shape[0]:
    #     if x.shape[0] > audio.shape[0]:
    #         diff = x.shape[0] - audio.shape[0]
    #         x = x[:-diff]
    #     else:
    #         diff = audio.shape[0] - x.shape[0]
    #         audio = audio[:-diff]


    _, _, units = gslm.encode(audio)
    
    y = units
    loss = 0.0
    if len(x.shape) == 2:
        x = torch.unsqueeze(x, 2)
    y_ = model["model"](x, padding_mask = x_mask)
    # y_ = model(x)

    # save losses
    loss = 0.0
    if config['use_ctc_loss']:

        # get sequences
        sequences = y.to(device=device, dtype=torch.int64)  # recall discrete units are 1-D so shape B x DU
        target_lengths = torch.full((sequences.shape[0],), sequences.shape[1]).to(device, dtype=torch.int64)
        blank = config['model_params']['n_classes'] - 1
        estimates = F.log_softmax(y_, dim=-1).permute((1, 0, 2))
        input_lengths = torch.full(size=(estimates.shape[1],), fill_value=estimates.shape[0], dtype=torch.int64).to(
            device)
        ctc_loss = criterion['ctc'](estimates, sequences, input_lengths, target_lengths, blank=blank,
                                    zero_infinity=True) * config['ctc_params']['loss_lambda'][0]

        # compute ctc loss
        loss += ctc_loss
        total_eval_loss["test/pho_ctc_loss"] += ctc_loss.item()
        ctc_decoder = ctc_utils.Decoder(blank_index=blank, silent=[None], remove_rep=True)
        estimated_sequences = torch.argmax(estimates, dim=-1)
        # print("Prediction: ",estimated_sequences[0], "Ground_Truth: ",sequences[0])
        cer = ctc_decoder.phone_word_error(estimated_sequences.T, sequences)
        total_eval_loss["test/cer"] += cer
    total_eval_loss["test/loss"] += loss
    # print('Done')
    return estimated_sequences


def test_model(config, testset, device):

    """Train the model"""

    # define models, criterion, scheduler, and optimizer
    model_class = getattr(
        neural_synthesis.models,
        config.get("model_type", "CnnRnnClassifier"),
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

    dataloader = torch.utils.data.DataLoader(testset, pin_memory=(device=='cuda'), collate_fn=testset.collate_raw, num_workers=0, batch_sampler=SizeAwareSampler(testset, 80000))  ## min batch_length = 200sec with 800 sampling rate
    total_eval_loss = defaultdict(float)
    eval_metrics = defaultdict(float)

    # model = CnnRnnClassifier()
    model['model'].load_state_dict(torch.load('torch_models/ssi/18/model_97770.pt'))
    model['model'].eval()
    model['model'].to(device)
    
    eval_test_model(config, device, model, 0, criterion, dataloader, total_eval_loss, eval_metrics,
               encoder_config=None)


def main():
    # parse arguments provided
    # parser = argparse.ArgumentParser(
    #     description="Supervised ECoG-to-X training"
    # )

    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     required=True,
    #     help="name of model to be used",
    # )
    # parser.add_argument(
    #     "--experiment_name",
    #     type=str,
    #     required=True,
    #     help="experiment name",
    # )
    # parser.add_argument(
    #     "--run_name",
    #     type=str,
    #     required=True,
    #     help="run name",
    # )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     required=True,
    #     help="yaml format configuration file. only the name needed",
    # )
    # parser.add_argument(
    #     "--train_filename",
    #     type=str,
    #     required=True,
    #     help=".txt file used for training set. only the name needed",
    # )
    # parser.add_argument(
    #     "--test_filename",
    #     type=str,
    #     required=True,
    #     help=".txt file used for eval set. only the name needed",
    # )
    # parser.add_argument(
    #     "--root_dir",
    #     type=str,
    #     required=True,
    #     help="gimlet_data directory (e.g. userdata/username/gimlet_data)",
    # )
    # parser.add_argument(
    #     "--debug",
    #     type=bool,
    #     default=False,
    #     help="Set to True to put in debug mode, which cycles through evaluation faster.",
    # )
    # parser.add_argument(
    #     "--chance",
    #     type=bool,
    #     default=False,
    #     help="Set to True to train a model on noise inputs",
    # )
    # parser.add_argument(
    #     "--subject",
    #     type=str,
    #     default='bravo3',
    #     help="subject",
    # )
    # parser.add_argument(
    #     "--train_data_fraction",
    #     type=float,
    #     default=1.0,
    #     help="Data fraction for training set",
    # )
    # args = parser.parse_args()

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