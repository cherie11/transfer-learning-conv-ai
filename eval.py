# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_,pad_dataset
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                          GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME,GPT2LMHeadModel,)

from utils import get_dataset, make_logdir
MODEL_INPUTS = ["input_ids",
                "lm_labels", "token_type_ids","cluster","cl_token_ids"]



PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    scalar_t = torch.tensor(scalar, dtype=torch.float,device=args.device)
    return scalar_t.item()



def get_test_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = {'test':get_dataset(tokenizer, args.dataset_path, args.dataset_cache)}

    logger.info("Build inputs and labels")
    datasets = {"test": defaultdict(list)}
    num_cluster = 25
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        for dialog in dataset:
            # persona = dialog["personality"].copy()

            # for _ in range(args.personality_permutations):
            for utterance in dialog["utterances"]:
                cluster = utterance["cluster"] 
                persona = utterance["personality"].copy()
                event = utterance["event"]
                pid = utterance['id']
                history = utterance["history"][-(2*args.max_history+1):]
                candidate=utterance["candidates"][-1]
                instance = build_input_from_segments(
                    persona, history, candidate, tokenizer, cluster, event,pid,lm_labels =  True)
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
    return test_loader, test_sampler



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--test_batch_size", type=int,
                        default=50, help="Batch size for test")
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

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    
    
    
    logger.info("Prepare datasets")
    test_loader, test_sampler = get_test_loaders(args, tokenizer)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device)
                          for input_tensor in batch)
            input_ids,lm_labels, token_type_ids,cluster,cl_token_ids= batch
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
             # (loss), lm_logits, presents, (all hidden_states), (attentions)
                
            lm_logits,*_ = model(
                input_ids, token_type_ids=token_type_ids, clusters=cluster,cl_token_ids=cl_token_ids,with_cluster = True
            )
            lm_logits_flat_shifted = lm_logits[..., :-1,
                                               :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted
    evaluator = Engine(inference)
    
    # Prepare metrics - note how we compute distributed metrics
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100))}#, output_transform=lambda x: (x[0][0], x[1][0]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    pbar = ProgressBar(persist=True)
    pbar.attach(evaluator, metric_names=["loss"])
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message(
            "Validation: %s" % pformat(evaluator.state.metrics)))
    evaluator.run(test_loader)
    