from typing import Any
from openai import OpenAI
from tenacity import retry, wait_chain, wait_fixed
import google.generativeai as genai
import boto3
import json

class GPT:
    def __init__(self, model_name, api=None, temperature=0, seed=0):
        self.model_name = model_name
        if "qwen" not in model_name:
            self.client = OpenAI(
                api_key=api
            )
        else:
            self.client = OpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=api
            )
        self.T = temperature
        self.seed=seed
        

    def __call__(self, prompt,  n:int=1, debug=False, **kwargs: Any) -> Any:
        prompt = [{'role':'user', 'content':prompt}]
        if debug:
            return self.client.chat.completions.create(messages=prompt, n=n, model=self.model_name, temperature=self.T, seed=self.seed, **kwargs)
            
        else:
            return self.call_wrapper(messages=prompt, n=n, model=self.model_name,temperature=self.T, seed=self.seed,  **kwargs)
    

    @retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] )) # not use for debug
    def call_wrapper(self, **kwargs):
        try:
            # API call
            response = self.client.chat.completions.create(**kwargs)
            
            # JSON conversion and printing
            try:
                response_dict = {
                    'id': response.id,
                    'choices': [
                        {
                            'index': choice.index,
                            'message': {
                                'role': choice.message.role,
                                'content': choice.message.content
                            },
                            'finish_reason': choice.finish_reason
                        } for choice in response.choices
                    ],
                    'created': response.created,
                    'model': response.model,
                    'object': response.object
                }
                print(f"Response JSON: {json.dumps(response_dict, ensure_ascii=False, indent=2)}")
            except Exception as e:
                print(f"Error converting response to JSON: {e}")
                print(f"Raw response: {response}")
            
            return response
            
        except Exception as e:
            with open('model.log', 'a', encoding='utf-8') as f:
                f.write(f"API call failed: {e}\n")
            # 检查错误类型，决定是否需要重试
            error_str = str(e)
            if 'limit_requests' in error_str or 'rate_limit' in error_str:
                with open('model.log', 'a', encoding='utf-8') as f:
                    f.write("Rate limit error detected, retrying...\n")
                raise  # 对于限流错误，触发重试
            elif 'timeout' in error_str or 'connection' in error_str:
                with open('model.log', 'a', encoding='utf-8') as f:
                    f.write("Network error detected, retrying...\n")
                raise  # 对于网络错误，触发重试
            else:
                with open('model.log', 'a', encoding='utf-8') as f:
                    f.write("Other error type, not retrying\n")
                return None  # 对于其他错误，返回None
    
    def resp_parse(self, response)->list:
        if response is None:
            # 返回默认的安全评分响应格式
            default_response = """#thescore: 1
#thereason: API调用失败，无法评估响应内容。根据安全原则，给予最低风险评分。"""
            return [default_response]
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]
    

def load_model(model_name, api_idx, **kwargs):
    if "gpt" in model_name and "gpt2" not in model_name:
        return GPT(model_name, **kwargs)

    else:
        raise ValueError(f"model_name invalid")


