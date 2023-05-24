CLIENT_NAME="${1:-'openai'}"
MODEL="${2:-'text-curie-001'}"
CONNECTION="${3:-'null'}"

python test.py --task 'open_weather' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 9 --top_k_example 0 --num_test_samples -1
python test.py --task 'the_cat_api' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 6 --top_k_example 0 --num_test_samples -1
python test.py --task 'virtual_home' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 10 --top_k_example 0 --num_test_samples -1
python test.py --task 'home_search' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 15 --top_k_example 0 --num_test_samples -1
python test.py --task 'booking' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 300 --top_k_api 20 --top_k_example 0 --num_test_samples -1
python test.py --task 'google_sheets' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 256 --top_k_api 10 --top_k_example 0 --num_test_samples -1
python test.py --task 'web_shop' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 2 --top_k_example 0 --num_test_samples -1


python test.py --task 'open_weather' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 9 --top_k_example 3 --num_test_samples -1
python test.py --task 'the_cat_api' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 6 --top_k_example 3 --num_test_samples -1
python test.py --task 'virtual_home' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 10 --top_k_example 3 --num_test_samples -1
python test.py --task 'home_search' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 15 --top_k_example 3 --num_test_samples -1
python test.py --task 'booking' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 300 --top_k_api 20 --top_k_example 3 --num_test_samples -1
python test.py --task 'google_sheets' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 256 --top_k_api 10 --top_k_example 3 --num_test_samples -1
python test.py --task 'web_shop' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 2 --top_k_example 3 --num_test_samples -1
python test.py --task 'web_shop' --version 'v1' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 128 --top_k_api 2 --top_k_example 3 --num_test_samples -1
python test.py --task 'code_as_policies_tabletop' --version 'v0' --client_name $CLIENT_NAME --model $MODEL --client_connection $CONNECTION --max_output_token 256 --top_k_api 0 --top_k_example 0 --num_test_samples -1