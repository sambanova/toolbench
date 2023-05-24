"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import json

from action_generator import RagGenerator
import evaluator

EVALUATOR_MAP = {
    "home_search": evaluator.HomeSearchEvaluator,
    "booking": evaluator.BookingEvaluator,
    "virtual_home": evaluator.VirtualHomeEvaluator,
    "google_sheets": evaluator.GoogleSheetsEvaluator,
    "open_weather": evaluator.OpenWeatherEvaluator,
    "code_as_policies_tabletop": evaluator.TabletopEvaluator,
    "web_shop": evaluator.WebShopEvaluator,
    "the_cat_api": evaluator.TheCatAPIEvaluator,
}

N_RETRY = 3


def test(
    task,
    task_dir,
    evaluator,
    run_name,
    num_test_samples,
    out_dir,
):
    def write_log(f, res_dict):
        try:
            json.dump(res_dict, f)
        except:
            json.dump(res_dict["crashed"], f)
        f.write("\n")
        f.flush()

    os.makedirs(out_dir, exist_ok=True)
    test_examples_file = f"{task_dir}/test.jsonl"
    log_file = f"{out_dir}/{run_name}.jsonl"
    with open(test_examples_file) as f, open(log_file, "w") as f_log:
        lines = f.readlines()
        if num_test_samples > 0:
            lines = lines[:num_test_samples]

        if task == "code_as_policies_tabletop":
            from moviepy.editor import ImageSequenceClip
            from evaluator.code_as_policies_env.tester import TESTERS
            from PIL import Image

            def make_image_name():
                factors = [
                    f"{tester_id}",
                    f"{configs_id}",
                    f"crash_{results['crashed']}",
                    f"success_{results['success']}",
                    instruction.replace(" ", "_"),
                ]
                return "__".join(factors)

            render_dir = f"{out_dir}/{run_name}"
            os.makedirs(render_dir, exist_ok=True)
            cur_tester_id = ""
            for line in lines:
                jsonobj = json.loads(line)
                tester_id = jsonobj["tester_id"]
                configs = jsonobj["configs"]
                configs_id = jsonobj["configs_id"]
                if cur_tester_id != tester_id:
                    tester = TESTERS[tester_id]()
                print(tester_id, configs_id)

                instruction = tester.reset_env_from_configs(configs)
                instruction = configs["instruction"]
                evaluator.setup(tester.env)
                results = evaluator(instruction, tester)
                write_log(f_log, results)

                # render results
                image_name = make_image_name()
                if tester.env.cache_video:
                    rendered_clip = ImageSequenceClip(tester.env.cache_video, fps=35)
                    rendered_clip.write_gif(f"{render_dir}/{image_name}.gif")
                else:
                    camera_img = tester.env.get_camera_image()
                    img = Image.fromarray(camera_img, "RGB")
                    img.save(f"{render_dir}/{image_name}_initial_state.png")
        elif task == "web_shop":
            for line in lines:
                data = json.loads(line)
                results = evaluator(data["id"])
                write_log(f_log, results)
        else:
            for line in lines:
                data = json.loads(line)
                query = data["prompt"].strip()
                labels = data["completion"]
                if task == "google_sheets":
                    results = evaluator(
                        query,
                        labels,
                        data["question_sheet_name"],
                        data["check_format"],
                    )
                elif task == "the_cat_api":
                    results = evaluator(query, labels, data["compare_string_only"])
                else:
                    results = evaluator(query, labels)
                write_log(f_log, results)
        final_results = evaluator.aggregate_results()
        print(final_results)
        write_log(f_log, final_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--client_name", type=str)
    parser.add_argument("--client_connection", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--version", type=str, default="v0")
    parser.add_argument("--max_output_token", type=int, default=128)
    parser.add_argument("--stop_tokens", type=str, default=None)
    parser.add_argument("--num_test_samples", type=int, default=-1)
    parser.add_argument("--top_k_api", type=int, default=0)
    parser.add_argument("--top_k_example", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="out/")
    args = parser.parse_args()

    assert args.task in EVALUATOR_MAP, args.task
    if args.stop_tokens:
        args.stop_tokens = eval(args.stop_tokens)
        assert type(args.stop_tokens) == list

    # Create generator
    if args.task == "code_as_policies_tabletop":
        query_template = "{query}"
    else:
        query_template = """{api_docs}\n{examples}\nTask: {query}\nAction:\n"""
    task_dir = f"data/{args.task}/{args.version}"
    generator = RagGenerator(
        client_name=args.client_name,
        client_connection=args.client_connection,
        model_name=args.model_name,
        context_dir=task_dir,
        max_output_token=args.max_output_token,
        stop_tokens=args.stop_tokens,
        top_k_api=args.top_k_api,
        top_k_example=args.top_k_example,
        query_template=query_template,
    )

    # Create evaluator
    etor = EVALUATOR_MAP[args.task](generator)

    # Run tests for 3 times
    def make_name(
        task,
        client_name,
        model_name,
        max_output_token,
        version,
        top_k_api,
        top_k_example,
        run,
    ):
        entities = [f"{task}_{version}"]
        model_name = model_name.replace("/", "-")
        entities += [f"model_{client_name}_{model_name}"]
        entities += [f"len_{max_output_token}"]
        entities += [f"n_api_{top_k_api}"]
        entities += [f"n_example_{top_k_example}"]
        entities += [f"run_{run}"]
        return "__".join(entities)

    for i in range(3):
        run_name = make_name(
            args.task,
            args.client_name,
            args.model_name,
            args.max_output_token,
            args.version,
            args.top_k_api,
            args.top_k_example,
            i,
        )
        for _ in range(N_RETRY):
            try:
                test(
                    args.task,
                    task_dir=task_dir,
                    evaluator=etor,
                    run_name=run_name,
                    num_test_samples=args.num_test_samples,
                    out_dir=args.out_dir,
                )
                break
            except:
                pass
