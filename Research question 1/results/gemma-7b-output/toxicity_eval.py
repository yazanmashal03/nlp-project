# import json
# import requests
#
# API URL 和密钥
api_key = "AIzaSyDoI9B1r2mvfP0Lj021i1X5c6iDjwdh2nc"
url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
# # 示例：指定 JSONL 文件路径
# jsonl_file_path = 'llama3.1_filtered_output/toxic_results_processed.jsonl'
#
#
# # 处理 JSONL 文件的主逻辑
# def process_jsonl_file(file_path):
#     data_points = []
#
#     # 读取 JSONL 文件
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 data_point = json.loads(line.strip())
#                 data_points.append(data_point)
#     except Exception as e:
#         print(f"Error while reading JSONL file: {e}")
#         return
#
#     print(f"Successfully loaded {len(data_points)} data points.")
#
#     # 清理每个 data_point 的 continuation 文本
#     for idx, data in enumerate(data_points):
#         print(f"Processing data point {idx + 1}...")
#
#         target_texts = [entry["text"] for entry in data["continuation"]]
#
#         cleaned_texts = []
#         for target_text in target_texts:
#             print(target_text + '\n')
#             # 请求数据
#             data = {
#                 "comment": {
#                     "text": target_text },
#                 "languages": ["en"],
#                 "requestedAttributes": {"TOXICITY": {},
#                                         "SEVERE_TOXICITY": {},
#                                         "INSULT": {},
#                                         "PROFANITY": {},
#                                         "IDENTITY_ATTACK": {},
#                                         "THREAT": {},
#                                         "SEXUALLY_EXPLICIT": {},
#                                         "FLIRTATION": {}
#                                         }
#             }
#
#             # 发送请求
#             response = requests.post(url, json=data)
#
#             # 打印响应
#             print(response.json())
#
# # 调用主函数
# process_jsonl_file(jsonl_file_path)
#
import json
import requests
import time

# Example: JSONL file path
jsonl_file_path = 'llama3.1_instruct_filtered_output/toxic_results_processed.jsonl'


# Main logic to process the JSONL file
def process_jsonl_file(file_path):
    data_points = []

    # Read JSONL file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data_point = json.loads(line.strip())
                data_points.append(data_point)
    except Exception as e:
        print(f"Error while reading JSONL file: {e}")
        return

    print(f"Successfully loaded {len(data_points)} data points.")

    # Process each data point's continuation text
    for idx, data in enumerate(data_points):
        print(f"Processing data point {idx + 1}...")

        target_texts = [entry["text"] for entry in data["continuation"]]

        for i, target_text in enumerate(target_texts):
            print(target_text + '\n')
            # Request data
            request_data = {
                "comment": {"text": target_text},
                "languages": ["en"],
                "requestedAttributes": {
                    "TOXICITY": {},
                    "SEVERE_TOXICITY": {},
                    "INSULT": {},
                    "PROFANITY": {},
                    "IDENTITY_ATTACK": {},
                    "THREAT": {},
                    "SEXUALLY_EXPLICIT": {},
                    "FLIRTATION": {}
                }
            }

            # Send request
            response = requests.post(url, json=request_data)
            response_data = response.json()

            # Extract and update toxicity analysis
            try:
                attribute_scores = response_data.get('attributeScores', {})
                toxicity_analysis = {
                    'SEXUALLY_EXPLICIT': round(
                        attribute_scores.get('SEXUALLY_EXPLICIT', {}).get('summaryScore', {}).get('value', -1), 3),
                    'TOXICITY': round(attribute_scores.get('TOXICITY', {}).get('summaryScore', {}).get('value', -1), 3),
                    'PROFANITY': round(attribute_scores.get('PROFANITY', {}).get('summaryScore', {}).get('value', -1),
                                       3),
                    'FLIRTATION': round(attribute_scores.get('FLIRTATION', {}).get('summaryScore', {}).get('value', -1),
                                        3),
                    'SEVERE_TOXICITY': round(
                        attribute_scores.get('SEVERE_TOXICITY', {}).get('summaryScore', {}).get('value', -1), 3),
                    'INSULT': round(attribute_scores.get('INSULT', {}).get('summaryScore', {}).get('value', -1), 3),
                    'IDENTITY_ATTACK': round(
                        attribute_scores.get('IDENTITY_ATTACK', {}).get('summaryScore', {}).get('value', -1), 3),
                    'THREAT': round(attribute_scores.get('THREAT', {}).get('summaryScore', {}).get('value', -1), 3)
                }

                # print(f"Orignal toxicity analysis: {data['continuation'][i]['toxicity_analysis']}")
                # Write the extracted values back to the data point
                data['continuation'][i]['toxicity_analysis'] = toxicity_analysis
                print(f"Updated toxicity analysis: {data['continuation'][i]['toxicity_analysis']}")

            except Exception as e:
                print(f"Error processing response for data point {idx + 1}, continuation {i + 1}: {e}")

            # Limit requests to one per second
            time.sleep(1)

    # Save updated data back to a new JSONL file
    output_file_path = file_path.replace('.jsonl', '_analyzed.jsonl')
    try:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for data_point in data_points:
                output_file.write(json.dumps(data_point) + '\n')
        print(f"Successfully saved updated data to {output_file_path}")
    except Exception as e:
        print(f"Error while saving updated JSONL file: {e}")


# Call main function
process_jsonl_file(jsonl_file_path)
