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

from action_generator import RagGenerator
import evaluator


def test_booking_evaluator():
    label = """API.select_booking_type('trip tickets')
API.set_num_adults(3)
API.select_transportation('cruise')
origin = Loc('Fort Worth')
API.set_origin(origin)
destination = Loc('Ontario')
API.set_destination(destination)
departure_date = Date(5, 17, 2023)
API.set_departure_date(departure_date)
API.search()"""
    context_dir = f"data/booking/v0"
    generator = RagGenerator(
        client_name="openai",
        model_name="text-curie-001",
        context_dir=context_dir,
        max_output_token=256,
        top_k_api=10,
        top_k_example=3,
        query_template="Task: {query} (Answer in code only)\nActions:\n",
    )

    e = evaluator.BookingEvaluator(generator)
    results = e(
        "Buy 3 cruise tickets for adults, from Fort Worth to Ontario on 2023/05/17.",
        [label],
    )
    print(results)


def test_home_search_evaluator():
    label = """API.set_location("Cape Coral")
API.set_buy_or_rent("buy")
API.select_home_type(["Multi-family", "Townhouse", "Co-op"])
API.set_min_price(707000)
API.set_max_price(1457000)
API.search()"""
    context_dir = f"data/home_search/v0"
    generator = RagGenerator(
        client_name="openai",
        model_name="text-curie-001",
        context_dir=context_dir,
        max_output_token=256,
        top_k_api=10,
        top_k_example=3,
        query_template="Task: {query} (Answer in code only)\nActions:\n",
    )

    e = evaluator.HomeSearchEvaluator(generator)
    results = e(
        "Find a multi-family, townhouse or co-op in Cape Coral between $707000 and $1457000.",
        [label],
    )
    print(results)


def test_open_weather_evaluator():
    label = """curl -X GET 'https://api.openweathermap.org/data/2.5/weather?lat=44.34&lon=10.99&appid={API_KEY}'"""
    context_dir = f"data/open_weather/v0"
    generator = RagGenerator(
        client_name="openai",
        model_name="text-curie-001",
        context_dir=context_dir,
        max_output_token=256,
        top_k_api=3,
        top_k_example=3,
        query_template="Task: {query} (Answer in code only)\nActions:\n",
    )

    e = evaluator.OpenWeatherEvaluator(generator)
    results = e(
        "Show me the weather of latitude 44.34 and lontitude 10.99",
        [label],
    )
    print(results)


def test_the_cat_api_evaluator():
    label = """curl -X GET 'https://api.thecatapi.com/v0/favourites'"""
    context_dir = f"data/the_cat_api/v0"
    generator = RagGenerator(
        client_name="openai",
        model_name="text-curie-001",
        context_dir=context_dir,
        max_output_token=256,
        top_k_api=3,
        top_k_example=3,
        query_template="Task: {query} (Answer in code only)\nActions:\n",
    )

    e = evaluator.TheCatAPIEvaluator(generator)
    results = e(
        "List all my favorite cats",
        [label],
    )
    print(results)


def test_virtual_home_evaluator():
    label = """Agent.WalkTo(bedroom)
Agent.WalkTo(floor_lamp)
Agent.Find(floor_lamp)
Agent.SwitchOn(floor_lamp)
Agent.Find(novel)
Agent.Grab(novel)
Agent.Read(novel)"""
    context_dir = f"data/virtual_home/v0"
    generator = RagGenerator(
        client_name="openai",
        model_name="text-curie-001",
        context_dir=context_dir,
        max_output_token=256,
        top_k_api=3,
        top_k_example=3,
        query_template="Task: {query} (Answer in code only)\nActions:\n",
    )

    e = evaluator.VirtualHomeEvaluator(generator)
    results = e(
        "send an email",
        [label],
    )
    print(results)


def test_google_sheets_evaluator():
    label = """worksheet.update("D4", "asd")"""
    context_dir = f"data/google_sheets/v0"
    generator = RagGenerator(
        client_name="openai",
        model_name="text-curie-001",
        context_dir=context_dir,
        max_output_token=256,
        top_k_api=0,
        top_k_example=3,
        query_template="Task: {query} (Answer in code only)\nActions:\n",
    )

    e = evaluator.GoogleSheetsEvaluator(generator)
    results = e(
        "Update cell D4 to asd",
        [label],
        "Sheet3",
    )
    print(results)


def test_web_shop_evaluator():
    context_dir = f"data/web_shop/v1"
    generator = RagGenerator(
        client_name="openai",
        model_name="text-curie-001",
        context_dir=context_dir,
        max_output_token=256,
        top_k_api=0,
        top_k_example=1,
        query_template="Task: {query} (Answer in code only)\nActions:\n",
    )

    e = evaluator.WebShopEvaluator(generator)
    results = e(10)
    print(results)


def test_tabletop_evaluator():
    from evaluator.code_as_policies_env.tester import TESTERS

    context_dir = f"data/code_as_policies_tabletop/v0"
    generator = RagGenerator(
        client_name="openai",
        model_name="text-curie-001",
        context_dir=context_dir,
        max_output_token=256,
        top_k_api=0,
        top_k_example=0,
    )
    e = evaluator.TabletopEvaluator(generator)

    with open(f"data/code_as_policies_tabletop/v0/test.jsonl") as f:
        line = f.readline()

        jsonobj = eval(line)
        tester_id = jsonobj["tester_id"]
        configs = jsonobj["configs"]
        configs_id = jsonobj["configs_id"]
        tester = TESTERS[tester_id]()
        print(tester_id, configs_id)

        instruction = tester.reset_env_from_configs(configs)
        instruction = configs["instruction"]
        e.setup(tester.env)
        results = e(instruction, tester)
        print(results)
