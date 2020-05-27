# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history,event,cluster,pid, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output,tokenizer,cluster, event, pid,  with_eos=False)
        cl_token_ids = torch.tensor(instance["cl_token_ids"], device=args.device).unsqueeze(0)
        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        cluster = torch.tensor(instance["cluster"], device=args.device).unsqueeze(0)
        
        logits = model(input_ids, token_type_ids=token_type_ids,clusters=cluster,cl_token_ids=cl_token_ids,with_cluster=True)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        cnt = 0
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids and cnt<10000:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                cnt +=1
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def get_test_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"test": defaultdict(list)}
    num_cluster = 25
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        for dialog in dataset:
            for utterance in dialog["utterances"]:
                cluster = utterance["cluster"] 
                persona = utterance["personality"]
                event = utterance["event"]
                pid = utterance['id']
                history = utterance["history"][-(2*args.max_history+1):]
                instance = build_input_from_segments(
                    persona, history, [], tokenizer, cluster, event,pid,lm_labels =  True)
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(
                        input_array)
                datasets[dataset_name]["n_cluster"] = num_cluster

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"test": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(
            dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            print(input_name,tensor.shape)
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    test_dataset = TensorDataset(*tensor_datasets["test"])
    test_sampler =  None
    test_loader = DataLoader(test_dataset, sampler=test_sampler,
                              batch_size=args.test_batch_size, shuffle=False)

    logger.info("test dataset (Batch, Seq length): {}".format(
        test_dataset.tensors[0].shape))
    return test_loader


def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
	
	
    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    logger.info("Sample a personality")
    dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    res_dialogs = []
    for dialog in tqdm( dataset):
        out_dialog = {}
        
        out_dialog["utterances"] = []
        for utterance in dialog["utterances"]:
            out_utterance = {}

            cluster = utterance["cluster"] 
            persona = utterance["personality"]
            event = utterance["event"]
            history = utterance["history"][-(2*args.max_history+1):]
            pid = utterance['id']     
            out_utterance['cluster']= utterance["cluster"] 
            out_utterance['event'] = tokenizer.decode(utterance["event"], skip_special_tokens=True)
            out_utterance['target']= tokenizer.decode(utterance["candidates"][-1], skip_special_tokens=True) 
            out_utterance['history'] =[tokenizer.decode(h, skip_special_tokens=True) for h in history] 
            out_utterance['pid'] = pid
            
            if pid not in out_dialog:
                out_dialog[pid] = [tokenizer.decode(per, skip_special_tokens=True) for per in utterance["personality"]] 
                
            with torch.no_grad():
                out_ids = sample_sequence(persona, history,event,cluster, pid,tokenizer, model, args)
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            out_utterance['pred']=out_text
            out_dialog["utterances"].append(out_utterance)
        res_dialogs.append(out_dialog)
    json.dump(res_dialogs,open("pred_with_switch_cluster_100.json",'w'))
                     


if __name__ == "__main__":
    run()
