#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from FlashOAgents import OpenAIServerModel
from FlashOAgents import VisualInspectorTool, TextInspectorTool, AudioInspectorTool, get_zip_description, get_single_file_description
from base_agent import MMSearchAgent
from utils import read_jsonl, write_jsonl, write_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)
logger.info(f"Loaded configuration from: {env_path}")



def process_item(item, model, summary_interval, prompts_type, max_steps, visual_tool, text_tool, audio_tool):

    search_agent = MMSearchAgent(
        model,
        summary_interval=summary_interval,
        prompts_type=prompts_type,
        max_steps=max_steps
    )

    question = item["question"]
    golden_answer = item["answer"]

    if item.get("file_name"):
        # Use the file_name as-is (already contains correct path like ./mm/images/xxx.png)
        if ".zip" in item["file_name"]:
            question += "\n\nTo solve the task above, you will have to use these attached files:\n"
            question += get_zip_description(
                item["file_name"], item["question"], visual_tool, text_tool, audio_tool,
            )
        else:
            question += "\n\nTo solve the task above, you will have to use this attached file:"
            question += get_single_file_description(
                item["file_name"], item["question"], visual_tool, text_tool, audio_tool,
            )

    try:
        result = search_agent(question)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Exception occurred while calling search_agent for question: {question[:100]}...")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback:\n{error_trace}")

        # Return partial result with error info
        return {
            "question": question,
            "golden_answer": golden_answer,
            "error": str(e),
            "error_trace": error_trace,
            "status": "failed"
        }

    return {
        "question": question,
        "golden_answer": golden_answer,
        "status": "success",
        **result,
    }


def main(args):
    # Log API configuration
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    default_model = os.environ.get("DEFAULT_MODEL")

    logger.info(f"API Base: {api_base}")
    logger.info(f"API Key: {'*' * 20}{api_key[-4:] if api_key else 'NOT SET'}")
    logger.info(f"Model: {default_model}")

    if not api_key or api_key == "your-siliconflow-api-key-here":
        logger.error("❌ Please set your OPENAI_API_KEY in .env file!")
        logger.error(f"Edit {env_path} and replace 'your-siliconflow-api-key-here' with your actual API key")
        return

    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    model = OpenAIServerModel(
        default_model,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=32768,
        api_key=api_key,
        api_base=api_base,
    )

    visual_tool = VisualInspectorTool(model, 100000)
    text_tool = TextInspectorTool(model, 100000)
    audio_tool = AudioInspectorTool(model, 100000)

    if args.infile.lower().endswith('.json'):
        with open(args.infile, 'r') as f:
            data = json.load(f)
    else:
        data = read_jsonl(args.infile)

    if args.sample_num is not None:
        data = data[:args.sample_num]

    # Determine output format based on file extension
    is_json_output = args.outfile.lower().endswith('.json')

    try:
        if is_json_output:
            # For .json files, try to read existing data as JSON array
            with open(args.outfile, 'r') as f:
                out_data = json.load(f)
        else:
            # For .jsonl files, read line by line
            out_data = read_jsonl(args.outfile)
    except Exception:
        out_data = []

    done_questions = set([item.get("question") for item in out_data])
    data_to_run = [item for item in data if item.get("question") not in done_questions]
    logger.info(f"Total data: {len(data)}, Completed: {len(done_questions)}, Remaining: {len(data_to_run)}")

    results = []
    file_lock = threading.Lock()

    def safe_write(result):
        with file_lock:
            if is_json_output:
                # For JSON: read all, append, write all (atomic operation within lock)
                try:
                    with open(args.outfile, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing_data = []

                existing_data.append(result)

                # Write to temp file first, then rename (atomic on most systems)
                temp_file = args.outfile + '.tmp'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=4)

                # Atomic rename
                import shutil
                shutil.move(temp_file, args.outfile)
            else:
                # For JSONL: append line
                write_jsonl(args.outfile, [result], "a")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        summary_interval = random.randint(args.summary_interval - 1, args.summary_interval + 1)

        futures = [
            executor.submit(
                process_item, 
                item, 
                model, 
                summary_interval, 
                args.prompts_type, 
                args.max_steps, 
                visual_tool, 
                text_tool, 
                audio_tool
            ) for item in data_to_run
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                results.append(result)
                safe_write(result)

    logger.info(f"Processing completed. Newly added: {len(results)}, Total completed: {len(done_questions) + len(results)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal data generation')

    parser.add_argument('--infile', type=str, default="", help='input filename')
    parser.add_argument('--outfile', type=str, default="", help='output filename')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--summary_interval', type=int, default=8, help='Summary interval')
    parser.add_argument('--prompts_type', type=str, default="default", help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=15, help='Number of concurrency')
    parser.add_argument('--max_steps', type=int, default=6, help='Maximum number of steps')

    args = parser.parse_args()

    # Auto-join paths for mm folder
    if not os.path.isabs(args.infile) and not args.infile.startswith('./') and not args.infile.startswith('.\\'):
        args.infile = os.path.join('mm', args.infile)

    # Auto-join output path
    if not args.outfile:
        # Use input filename but change extension to .jsonl
        base_name = os.path.splitext(os.path.basename(args.infile))[0]
        args.outfile = os.path.join('output_for_analysis', 'mm', f'{base_name}.jsonl')
    elif not os.path.isabs(args.outfile) and not args.outfile.startswith('./') and not args.outfile.startswith('.\\'):
        # Check if path already contains output_for_analysis to avoid duplication
        normalized_path = args.outfile.replace('\\', '/')
        if not normalized_path.startswith('output_for_analysis/'):
            args.outfile = os.path.join('output_for_analysis', 'mm', args.outfile)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    logger.info(f"Input: {args.infile}")
    logger.info(f"Output: {args.outfile}")

    main(args)
    