#This script is taken from https://github.com/kimdanny/Fair-RAG

"""
Perform LM inference either with
a) 0 profile: to get the baseline perforamance
b) 1 profile: to get the utility-gain of one item (profile) for a specific generator
"""
from tqdm import tqdm
import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import argparse
from typing import List
from transformers import AutoTokenizer
from utils import models_info, trim_sentence_by_token_len
from data.lamp_handler import LaMPHandler
from generator.lm import PromptLM


def main(args):
    MODEL_NAME: str = args.model_name
    LAMP_NUM: int = args.lamp_num
    EXPERIMENT_BASELINE: bool = args.experiment_baseline
    TOKENIZER = AutoTokenizer.from_pretrained(models_info[MODEL_NAME]["model_id"], trust_remote_code=True)
    TOKENIZER_MAX_LEN = min(TOKENIZER.model_max_length, 8192)
    K = 0 if EXPERIMENT_BASELINE else args.k
    dataset_name = args.dataset
    if dataset_name == "":
        print("you must pass a valid value for --dataset arg.")
        return
    
    retriever_name = args.retriever
    if retriever_name == "":
        print("you must pass a valid value for --retriever arg.")
        return

    task = args.task
    if task == "":
        print("you must pass a valid value for --task arg.")
        return
        
    query_topic = args.query_topic
    if query_topic == "":
        print("you must pass a valid value for --query_topic arg.")
        return
        
    output_dir = os.path.join(
        CUR_DIR_PATH,
        "inference_results",
        dataset_name,
        task,
        query_topic,
        retriever_name,
        MODEL_NAME
    )   
    os.makedirs(output_dir, exist_ok=True)
    if EXPERIMENT_BASELINE:
        output_file = open(os.path.join(output_dir, "5_output_baseline.log"), "w")
    elif K==1:
        output_file = open(os.path.join(output_dir, "5_output_augment.log"), "w")
    elif K>1:
        output_file = open(os.path.join(output_dir, "5_output_rag.log"), "w")
        
    # qid: question ID
    # pid: profile ID
    col_names = ["qid", "pid", "answer", "target"]
    # print("\t".join(col_names), flush=True)
    output_file.write("\t".join(col_names) + "\n")
    lamp_handler = LaMPHandler(
        split_type=args.lamp_split_type,
        tokenizer_model_name=models_info[MODEL_NAME]["model_id"],
        retriever = retriever_name,
        dataset = dataset_name,
        task = task,
        k=K,
        query_topic=query_topic
    )
    qa_model = PromptLM(model_name=MODEL_NAME)
    aip_func = lamp_handler.get_aip_func(lamp_num=LAMP_NUM)

    inputs_file_iterator = lamp_handler.get_inputs_file_iterator(lamp_number=LAMP_NUM)
    outputs_file_iterator = lamp_handler.get_outputs_file_iterator(lamp_number=LAMP_NUM)

    for i, (input_entry, output_entry) in tqdm(enumerate(
        zip(inputs_file_iterator, outputs_file_iterator)
    )):
        # First 1000 queries
        # if i > 5:
        #     break

        assert input_entry["id"] == output_entry["id"]
        entry_id: str = input_entry["id"]
        entry_question: str = input_entry["input"]
        profiles: List[dict] = input_entry["profile"]
        # gold label
        entry_target = output_entry["output"]

        if EXPERIMENT_BASELINE:
            final_prompt=trim_sentence_by_token_len(
                    entry_question,
                    tokenizer=TOKENIZER,
                    max_tok_len=TOKENIZER_MAX_LEN,
                )
            answer = qa_model.answer_question(final_prompt)
            if answer.strip() == "":
                for j in range(5):
                    answer = qa_model.answer_question(final_prompt=final_prompt)
                    if answer.strip() != "":
                        break
            answer = answer.replace("\r", " ").replace("\n", " ")
            s = "\t".join([entry_id, "-1", answer, entry_target])
            # print(s, flush=True)
            output_file.write(s+"\n")
        else:
            if K>1:
                final_prompt = aip_func(question=entry_question, profiles=profiles)
                final_prompt = trim_sentence_by_token_len(
                    final_prompt,
                    tokenizer=TOKENIZER,
                    max_tok_len=TOKENIZER_MAX_LEN,
                )
                answer = qa_model.answer_question(final_prompt=final_prompt)
                if answer.strip() == "":
                    for j in range(5):
                        answer = qa_model.answer_question(final_prompt=final_prompt)
                        if answer.strip() != "":
                            break
                answer = answer.replace("\r", " ").replace("\n", " ")
                s = "\t".join([entry_id, "-1", answer, entry_target])
                # print(s, flush=True)
                output_file.write(s+"\n")
            else:
                for profile in profiles:
                    # augment with one profile one by one to test its relevancy (usefulness)
                    final_prompt = aip_func(question=entry_question, profiles=[profile])
                    final_prompt = trim_sentence_by_token_len(
                        final_prompt,
                        tokenizer=TOKENIZER,
                        max_tok_len=TOKENIZER_MAX_LEN,
                    )
                    answer = qa_model.answer_question(final_prompt=final_prompt)
                    if answer.strip() == "":
                        for j in range(5):
                            answer = qa_model.answer_question(final_prompt=final_prompt)
                            if answer.strip() != "":
                                break
                    answer = answer.replace("\r", " ").replace("\n", " ")
                    s = "\t".join([entry_id, profile["id"], answer, entry_target])
                    # print(s, flush=True)
                    output_file.write(s+"\n")
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="flanT5XXL",
        help="Model nickname of HF model",
    )

    parser.add_argument(
        "--lamp_split_type",
        type=str,
        default="user",
        help="data split type of LaMP: either 'user' or 'time'",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="user",
        help="The task you want to solve using RAG.",
    )

    parser.add_argument(
        "--lamp_num",
        type=int,
        help="LaMP number",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="number of k (how many to retrieve)",
    )

    parser.add_argument(
        "--experiment_baseline",
        action="store_true",
        help="Enable baseline experiment (no profile injection)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Help with nameing output dir creation",
    )

    parser.add_argument(
        "--retriever",
        type=str,
        default="",
        help="Help with reading input dataset from the correct directory",
    )

    parser.add_argument(
        "--query_topic",
        type=str,
        default="",
        help="Help with reading input dataset from the correct directory",
    )
    args = parser.parse_args()

    main(args)
