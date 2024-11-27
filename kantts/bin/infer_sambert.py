import sys
import torch
import os
import numpy as np
import argparse
import yaml
import logging

import json
import re

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

try:
    from kantts.models import model_builder
    from kantts.utils.ling_unit.ling_unit import KanTtsLinguisticUnit
except ImportError:
    raise ImportError("Please install kantts.")

logging.basicConfig(
    #  filename=os.path.join(stage_dir, 'stdout.log'),
    format="%(asctime)s, %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def denorm_f0(mel, f0_threshold=30, uv_threshold=0.6, norm_type='mean_std', f0_feature=None):
    if norm_type == 'mean_std':
        f0_mvn = f0_feature

        f0 = mel[:, -2]
        uv = mel[:, -1]

        uv[uv < uv_threshold] = 0.0
        uv[uv >= uv_threshold] = 1.0

        f0 = f0 * f0_mvn[1:, :] + f0_mvn[0:1, :]
        f0[f0 < f0_threshold] = f0_threshold

        mel[:, -2] = f0
        mel[:, -1] = uv
    else: # global
        f0_global_max_min = f0_feature

        f0 = mel[:, -2]
        uv = mel[:, -1]

        uv[uv < uv_threshold] = 0.0
        uv[uv >= uv_threshold] = 1.0

        f0 = f0 * (f0_global_max_min[0] - f0_global_max_min[1]) + f0_global_max_min[1]
        f0[f0 < f0_threshold] = f0_threshold

        mel[:, -2] = f0
        mel[:, -1] = uv

    return mel

def am_synthesis(symbol_seq, fsnet, ling_unit, device, se=None):
    inputs_feat_lst = ling_unit.encode_symbol_sequence(symbol_seq)

    inputs_feat_index = 0
    if ling_unit.using_byte():
        inputs_byte_index = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_ling = torch.stack([inputs_byte_index], dim=-1).unsqueeze(0)
    else:
        inputs_sy = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_tone = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_syllable = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_ws = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_ling = torch.stack(
            [inputs_sy, inputs_tone, inputs_syllable, inputs_ws], dim=-1
        ).unsqueeze(0)

    inputs_feat_index = inputs_feat_index + 1
    inputs_emo = (
        torch.from_numpy(inputs_feat_lst[inputs_feat_index])
        .long()
        .to(device)
        .unsqueeze(0)
    )

    inputs_feat_index = inputs_feat_index + 1
    se_enable = False if se is None else True
    if se_enable:
        inputs_spk = (
            torch.from_numpy(se.repeat(len(inputs_feat_lst[inputs_feat_index]), axis=0))
            .float()
            .to(device)
            .unsqueeze(0)[:, :-1, :]
        )
    else:
        inputs_spk = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index])
            .long()
            .to(device)
            .unsqueeze(0)[:, :-1]
        )

    inputs_len = (
        torch.zeros(1).to(device).long() + inputs_emo.size(1) - 1
    )  # minus 1 for "~"


    res = fsnet(
        inputs_ling[:, :-1, :],
        inputs_emo[:, :-1],
        inputs_spk,
        inputs_len,
    )

    x_band_width = res["x_band_width"]
    h_band_width = res["h_band_width"]
    #  enc_slf_attn_lst = res["enc_slf_attn_lst"]
    #  pnca_x_attn_lst = res["pnca_x_attn_lst"]
    #  pnca_h_attn_lst = res["pnca_h_attn_lst"]
    dec_outputs = res["dec_outputs"]
    postnet_outputs = res["postnet_outputs"]
    LR_length_rounded = res["LR_length_rounded"]
    log_duration_predictions = res["log_duration_predictions"]
    pitch_predictions = res["pitch_predictions"]
    energy_predictions = res["energy_predictions"]

    valid_length = int(LR_length_rounded[0].item())
    dec_outputs = dec_outputs[0, :valid_length, :].cpu().numpy()
    postnet_outputs = postnet_outputs[0, :valid_length, :].cpu().numpy()
    duration_predictions = (
        (torch.exp(log_duration_predictions) - 1 + 0.5).long().squeeze().cpu().numpy()
    )
    pitch_predictions = pitch_predictions.squeeze().cpu().numpy()
    energy_predictions = energy_predictions.squeeze().cpu().numpy()

    logging.info("x_band_width:{}, h_band_width: {}".format(x_band_width, h_band_width))

    return (
        dec_outputs,
        postnet_outputs,
        duration_predictions,
        pitch_predictions,
        energy_predictions,
    )

def parse_symbol_sequence(symbol_sequence, ling_unit):
    """
    Parse the input symbol sequence to extract phoneme names (with tones) and their IDs.

    Args:
        symbol_sequence (str): Input symbol sequence string from the sentence file.
        ling_unit (KanTtsLinguisticUnit): Linguistic unit instance to fetch phoneme IDs.

    Returns:
        tuple: A tuple of phoneme names (with tones) and IDs.
    """
    # Define the pattern to match phoneme blocks
    pattern = r"{([^}]*)}"
    matches = re.findall(pattern, symbol_sequence)

    # List of phoneme names to ignore
    # ignore_phonemes = {"#1", "#2", "#3", "#4", "#_", "#~", "[MASK]"}

    phn_names_with_tone = []
    phn_ids = []

    for match in matches:
        components = match.split("$")
        phn_name = components[0]  # Extract phoneme name, e.g., 'y_c'
        tone = None

        # Skip ignored phonemes
        # if phn_name in ignore_phonemes:
            # continue

        # Extract tone information
        for component in components:
            if component.startswith("tone"):
                tone = component[4:]  # Get the tone number after 'tone'

        # Replace 'none' tone with 5
        if tone == "none":
            tone = "5"

        # Construct phoneme name with tone for output
        phn_name_with_tone = f"{phn_name}{tone}" if tone else phn_name
        phn_names_with_tone.append(phn_name_with_tone)

        # Fetch phoneme ID without tone for ID lookup
        phn_id = ling_unit.get_phoneme_id(f"@{phn_name}")
        phn_ids.append(phn_id)

    return phn_names_with_tone, phn_ids

def am_infer(sentence, ckpt, output_dir, se_file=None, config=None):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", 0)

    if config is not None:
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
    else:
        am_config_file = os.path.join(
            os.path.dirname(os.path.dirname(ckpt)), "config.yaml"
        )
        with open(am_config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

    ling_unit = KanTtsLinguisticUnit(config)
    ling_unit_size = ling_unit.get_unit_size()
    config["Model"]["KanTtsSAMBERT"]["params"].update(ling_unit_size)

    se_enable = config["Model"]["KanTtsSAMBERT"]["params"].get("SE", False) 
    se = np.load(se_file) if se_enable else None

    # nsf
    nsf_enable = config["Model"]["KanTtsSAMBERT"]["params"].get("NSF", False) 
    if nsf_enable:
        nsf_norm_type = config["Model"]["KanTtsSAMBERT"]["params"].get("nsf_norm_type", "mean_std")
        if nsf_norm_type == "mean_std":
            f0_mvn_file = os.path.join(
                os.path.dirname(os.path.dirname(ckpt)), "mvn.npy"
            )
            f0_feature = np.load(f0_mvn_file)   
        else: # global
            nsf_f0_global_minimum = config["Model"]["KanTtsSAMBERT"]["params"].get("nsf_f0_global_minimum", 30.0) 
            nsf_f0_global_maximum = config["Model"]["KanTtsSAMBERT"]["params"].get("nsf_f0_global_maximum", 730.0) 
            f0_feature = [nsf_f0_global_maximum, nsf_f0_global_minimum]

    model, _, _ = model_builder(config, device)

    fsnet = model["KanTtsSAMBERT"]

    logging.info("Loading checkpoint: {}".format(ckpt))
    state_dict = torch.load(ckpt)

    fsnet.load_state_dict(state_dict["model"], strict=False)

    results_dir = os.path.join(output_dir, "feat")
    os.makedirs(results_dir, exist_ok=True)
    fsnet.eval()

    # Initialize variables for managing JSON file creation
    current_main_id = None
    json_results = []  # This will accumulate all results for a single main_id

    # Ensure the output directory for JSON files exists
    res_wavs_dir = os.path.join(output_dir, "res_wavs")
    os.makedirs(res_wavs_dir, exist_ok=True)

    with open(sentence, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            logging.info("Inference sentence: {}".format(line[0]))

            # Extract the "main_id_sub_id" from the start of the sentence (e.g., "0_0")
            match = re.match(r"^([0-9]+)_([0-9]+)", line[0])
            if match:
                main_id = int(match.group(1))
                sub_id = int(match.group(2))

                # Check if the current sentence belongs to a new main_id
                if main_id != current_main_id:
                    # Save the current main_id's results to a JSON file
                    if json_results:
                        json_filename = os.path.join(res_wavs_dir, f"{current_main_id}.json")
                        with open(json_filename, "w", encoding="utf-8") as json_file:
                            json.dump(json_results[0], json_file, ensure_ascii=False, indent=4)
                    
                    # Reset for new main_id
                    json_results = []  # Start fresh for the new main_id
                    current_main_id = main_id

                mel_path = f"{results_dir}/{line[0]}_mel.npy"
                dur_path = f"{results_dir}/{line[0]}_dur.txt"
                f0_path = f"{results_dir}/{line[0]}_f0.txt"
                energy_path = f"{results_dir}/{line[0]}_energy.txt"

                with torch.no_grad():
                    mel, mel_post, dur, f0, energy = am_synthesis(
                        line[1], fsnet, ling_unit, device, se=se
                    )

                if nsf_enable:
                    mel_post = denorm_f0(mel_post, norm_type=nsf_norm_type, f0_feature=f0_feature) 

                # Parse symbol sequence for phoneme details
                phn_name_seq, phn_id_seq = parse_symbol_sequence(line[1], ling_unit)
                phn_dur_seq = dur.tolist()

                # Filter out phonemes with zero duration
                filtered_phn_name_seq = []
                filtered_phn_id_seq = []
                filtered_phn_dur_seq = []

                for phn_name, phn_id, duration in zip(phn_name_seq, phn_id_seq, phn_dur_seq):
                    if duration != 0:  # Skip phonemes with duration 0
                        filtered_phn_name_seq.append(phn_name)
                        filtered_phn_id_seq.append(phn_id)
                        filtered_phn_dur_seq.append(duration)

                phn_seq_len = len(filtered_phn_name_seq)

                # Create a new entry for the current segment
                result = {
                    "result": {
                        "audio": 0,
                        "audio_len": 0,
                        "is_end": False,  # Default value, will be updated later for the last entry
                        "phn_dur_seq": filtered_phn_dur_seq,
                        "phn_id_seq": filtered_phn_id_seq,
                        "phn_name_seq": filtered_phn_name_seq,
                        "phn_seq_len": phn_seq_len,
                    },
                    "status": 10000,
                    "status_msg": "Success"
                }

                # Append to the list for the current main_id
                json_results.append(result)

                np.save(mel_path, mel_post)
                np.savetxt(dur_path, dur)
                np.savetxt(f0_path, f0)
                np.savetxt(energy_path, energy)

    # After the loop ends, save the last JSON file
    if json_results:
        # Set "is_end" for the last segment
        json_results[-1]["result"]["is_end"] = True  
        json_filename = os.path.join(res_wavs_dir, f"{current_main_id}.json")
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(json_results[0], json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--se_file", type=str, required=False)

    args = parser.parse_args()

    am_infer(args.sentence, args.ckpt, args.output_dir, args.se_file)
