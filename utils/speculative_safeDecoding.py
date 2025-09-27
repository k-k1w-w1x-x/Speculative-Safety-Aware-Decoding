import torch
import numpy as np
import copy
import logging
import time

class SpeculativeSafeDecoding:
    def __init__(self, model, tokenizer, expert_model,expert_tokenizer,  alpha1=0.5,alpha2=0.5,top_k = 10, num_common_tokens = 10, verbose=False):
        self.model = model
        self.tokenizer = tokenizer
        self.expert_model = expert_model
        self.expert_tokenizer = expert_tokenizer
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.top_k = top_k
        self.num_common_tokens = num_common_tokens
        self.verbose = verbose
        self.do_sample = False
        self.top_p = None
        logging.info("SafeDecoding initialized.")
    def naive_spec(self, tensor1, tensor2, C, alpha):
        return 1,torch.topk(tensor1, 1).indices.tolist()[0]
    def safe_decoding(self, tensor1, tensor2, C, alpha):
        tensor2 = tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        left, right = 1, n
        ans_k = None
        while left <= right:
            mid = (left + right) // 2
            indices1 = set(torch.topk(tensor1, mid).indices.tolist())
            indices2 = set(torch.topk(tensor2, mid).indices.tolist())
            if len(indices1 & indices2) >= C:
                ans_k = mid
                right = mid - 1
            else:
                left = mid + 1

        if ans_k is None:
            return None, None  # 若没有满足条件的 k

        # 获取最终的交集
        topk1 = torch.topk(tensor1, ans_k)
        topk2 = torch.topk(tensor2, ans_k)
        indices1 = set(topk1.indices.tolist())
        indices2 = set(topk2.indices.tolist())
        intersection = indices1 & indices2
        # 在交集内，找到 (tensor1 - tensor2) 差值最大的下标
        max_index = max(intersection, key=lambda idx: tensor1[idx] + alpha*(tensor2[idx] - tensor1[idx]))
        return ans_k, max_index



    def safe_decoding_Union(self, tensor1, tensor2, C, alpha):
        tensor2 = tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        topk1 = torch.topk(tensor1, C)
        topk2 = torch.topk(tensor2, C)
        indices1 = set(topk1.indices.tolist())
        indices2 = set(topk2.indices.tolist())
        intersection = indices1 | indices2
        max_index = max(intersection, key=lambda idx: tensor1[idx] + alpha*(tensor2[idx] - tensor1[idx]))
        return C, max_index

    def safe_decoding_Inter(self, tensor1, tensor2, C, alpha):
        tensor2 = tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        
        # 预先计算所有topk结果，避免重复计算
        topk1 = torch.topk(tensor1, tensor1.size(0))
        topk2 = torch.topk(tensor2, tensor2.size(0))
        indices1 = topk1.indices
        indices2 = topk2.indices
        # try binary search if C is large
        ans_k = None
        set1 = set()
        set2 = set()
        for k in range(1, n + 1):
            set1.add(indices1[k-1].item())
            set2.add(indices2[k-1].item())
            if len(set1 & set2) >= C:
                ans_k = k
                break

        if ans_k is None:
            return None, None

        # 使用预先计算的indices获取交集
        indices1_set = set(indices1[:ans_k].tolist())
        indices2_set = set(indices2[:ans_k].tolist())

        if indices1[0].item() not in indices2_set:
            flag = -2
            if indices2[0].item() not in indices1_set:
                flag = 0
            if self.verbose:
                logging.info(f"unmatch:indices2:{indices2[0].item()}: {self.tokenizer.decode(indices2[0].item())},indices1:{indices1[0].item()}: {self.tokenizer.decode(indices1[0].item())},flag:{flag}")
            return flag, indices1[0].item()
        else:
            intersection = indices1_set & indices2_set
            max_index = max(intersection, key=lambda idx: tensor1[idx] + alpha*(tensor2[idx] - tensor1[idx]))
            return 1, max_index
        

        
    def safedecoding_speculative(self, inputs, expert_inputs=None, gen_config=None,return_generated_sequence=False):
        if gen_config is None:
            gen_config = self.model.generation_config

        # Override the generation config for our decoding
        self.do_sample = gen_config.do_sample
        self.top_p = gen_config.top_p   
        # if self.verbose:
        #     logging.info(f"Generation config: {gen_config}")

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
        generated_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_len = inputs['input_ids'].shape[1]
        batch_size = inputs['input_ids'].shape[0]
        assert batch_size == 1, "Batch size must be 1"

        expert_inputs = {k:v.cuda(self.expert_model.device) for k,v in expert_inputs.items()}
        expert_generated_ids = expert_inputs['input_ids']
        expert_attention_mask = expert_inputs['attention_mask']
        expert_input_len = expert_inputs['input_ids'].shape[1]
        expert_batch_size = expert_inputs['input_ids'].shape[0]
        assert expert_batch_size == 1, "Batch size must be 1"
        
        num_speculate_tokens = 3
        max_token_len = gen_config.max_new_tokens
        M = max_token_len
        alpha1 = self.alpha1
        alpha2 = self.alpha2
        min_alpha1 = 0.3
        print("alpha1,alpha2:",alpha1,alpha2)
        g = 7
        C = 10
        threshold = 0.6 # change threshold to 0.5 to get better results for Llama3.1
        decoding_mode = 1
        base_threshold = 0.6 
        next_decoding_mode = 1

        # start_time = time.time()
        decoding_time = 0
        with torch.no_grad():
            # 初始化生成过程
            cur_len=0
            # extra_match_tokens=0
            is_token_match_log=[]
            is_safe_decoding_log=[]
            while cur_len < M:
                # start_time = time.time()
                expert_probability_distributions=[]
                expert_speculate_ids = []
                raw_output = generated_ids.to(self.expert_model.device)
                expert_raw_output = expert_generated_ids.to(self.expert_model.device)
                prepared_attention_masks = [attention_mask]
                expert_prepared_attention_masks = [expert_attention_mask]
                for i in range(num_speculate_tokens):
                    prepared_attention_masks.append(torch.cat([prepared_attention_masks[i],torch.ones((1, 1), device=attention_mask.device)],dim=-1))
                    expert_prepared_attention_masks.append(torch.cat([expert_prepared_attention_masks[i],torch.ones((1, 1), device=expert_attention_mask.device)],dim=-1))
                # 猜测的token
                for _ in range(num_speculate_tokens):
                    # 获取当前 token 的 logits
                    outputs =self.expert_model(
                    input_ids=expert_raw_output.to(self.expert_model.device),
                    attention_mask=expert_prepared_attention_masks[_],
                    return_dict=True
                )
                    logits = outputs.logits
                    
                    # 获取当前最后一个 token 的 logits（假设是 batch_size=1）
                    last_token_logits = logits[0, -1, :]
                    
                    # 将 logits 转换为概率分布
                    probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
                    
                    # 保存当前的概率分布
                    expert_probability_distributions.append(probabilities)
                    
                    # 贪婪选择最大概率的 token
                    next_token = torch.argmax(probabilities, dim=-1)
                    # print(f"next_token,{next_token.item(),next_token.shape}")
                    # print(raw_output.shape)
                    expert_speculate_ids.append(next_token.item())
                    
                    # 将选中的 token 添加到生成序列中(不需要最后一个)
                    if _ != num_speculate_tokens-1:
                        raw_output = torch.cat((raw_output, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
                        expert_raw_output = torch.cat((expert_raw_output, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
                    # generated_expert_text = self.tokenizer.decode(expert_speculate_ids, skip_special_tokens=True)
                    # print("\nGenerated expert text:", generated_expert_text)
                
                # get_speculate_time = time.time()
                # print(f"get_speculate_time: {get_speculate_time - start_time}")

                outputs = self.model(
                    input_ids=raw_output,
                    attention_mask=prepared_attention_masks[-2],
                    return_dict=True
                )
                logits = outputs.logits
                last_token_logits = logits[0, -num_speculate_tokens:, :]
                
                # get_logits_time = time.time()
                # print(f"get_logits_time: {get_logits_time - get_speculate_time}")

                # 转换为概率
                probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
                cur_tokens=[]
                break_flag = False
                for i in range(num_speculate_tokens):
                    # 执行算法，贪婪采样
                    correct_flag = False
                    decoding_start_time = time.time()
                    if decoding_mode == 1:
                        k,cur_token = self.safe_decoding_Inter(tensor1 = probabilities[i-num_speculate_tokens,:],tensor2 =expert_probability_distributions[i],C=C,alpha=alpha1)
                    else:
                        k,cur_token = self.safe_decoding_Union(tensor1 = probabilities[i-num_speculate_tokens,:],tensor2 =expert_probability_distributions[i],C=C,alpha=alpha2)
                    decoding_end_time = time.time()
                    decoding_time += decoding_end_time - decoding_start_time
                    is_safe_decoding_log.append(k)
                    if(cur_token == expert_speculate_ids[i]):
                        correct_flag = True # 可以检查下一步
                        is_token_match_log.append(1) # 这一步match了
                    else:
                        is_token_match_log.append(0)

#---------------------------------------------------------switch decoding_mode,with threshold
                    if len(is_token_match_log) % g == 0:
                        if(sum(is_token_match_log[-g:])/ g < threshold):
                            next_decoding_mode = 2
                        else:
                            next_decoding_mode = 1
                        if next_decoding_mode == decoding_mode:
                            # Fix: Use a small epsilon to handle floating point precision
                            threshold = max(threshold - 0.1, 0)
                            if threshold <= 1e-10:  # If threshold is very close to 0
                                threshold = 0.0
                            if(decoding_mode == 1):
                                alpha1 = max(alpha1-0.15, min_alpha1)
                        else:
                            threshold = base_threshold
                            alpha1 = self.alpha1
                        decoding_mode = next_decoding_mode
                        # print(f"cur_len:{cur_len},alpha:{alpha},threshold:{threshold}")
#---------------------------------------------------------                          
                    cur_token = torch.Tensor([cur_token]).unsqueeze(0).long().to(generated_ids.device)   
                    cur_tokens.append(cur_token) 
                    cur_len += 1

                    if(cur_len >= M):
                        break_flag = True
                        break
                    # 终止条件：遇到EOS或达到最大长度
                    if cur_token.item() == self.tokenizer.eos_token_id:
                        break_flag = True
                        break
                    if correct_flag == False:
                        break
                # extra_match_tokens +=  len(cur_tokens)-1
                generated_ids = torch.cat([generated_ids]+cur_tokens, dim=-1)
                attention_mask = prepared_attention_masks[len(cur_tokens)]
                expert_generated_ids = torch.cat([expert_generated_ids]+cur_tokens, dim=-1)
                expert_attention_mask = expert_prepared_attention_masks[len(cur_tokens)]
                # print("extra_match_tokens:",extra_match_tokens)

                # validating_end_time = time.time()
                # print("validating time:",validating_end_time - get_speculate_time)
                # print(f"step {cur_len} total epoch time: {validating_end_time - start_time},extra match tokens:{extra_match_tokens}")
                if break_flag:
                    break
        # generation_end_time = time.time()
        # print(f"generation time: {generation_end_time - function_start_time},time per token:{(generation_end_time - function_start_time)/cur_len}")  
            # Split the list into groups of `group_size`
        # end_time = time.time()
        
        # print("generated_ids:",self.tokenizer.decode(generated_ids[0]))
        # print("expert_generated_ids:",self.tokenizer.decode(expert_generated_ids[0]))
        # raise Exception("stop")
        groups = [is_token_match_log[i:i + g] for i in range(0, len(is_token_match_log), g)]
        ones_ratio = [sum(group) / len(group) for group in groups]
        
        full_texts = [] # the whole conversation texts
        output_texts = [] # the model output part texts
        output_len = 0
        for i, item in enumerate(generated_ids):

            input_pos = inputs['attention_mask'][i].nonzero()

            input_length = input_pos.shape[0] # how many input tokens
            start_pos = input_pos[0][0] # the first non-padding token

            full_text = self.tokenizer.decode(item, skip_special_tokens=True)
            if return_generated_sequence:
                return item[start_pos + input_length:].tolist(),len(item[start_pos + input_length:].tolist())
            output_text = self.tokenizer.decode(item[start_pos + input_length:], skip_special_tokens=True)
            output_len = len(item[start_pos + input_length:])
            # print(f"token per second: {output_len/(end_time - start_time)}")
            full_texts.append(full_text)
            output_texts.append(output_text)
        print(output_texts[0])
        return output_texts[0], output_len
    
    
    def generate_baseline(self, inputs, gen_config=None):
        if gen_config is None:
            gen_config = self.model.generation_config
        
        if self.verbose:
            logging.info(f"Generation config: {gen_config}")

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
        # start_time = time.time()
        output_base = self.model.generate(**inputs,
                            generation_config=gen_config,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            num_beams=2,
                            length_penalty=1.5,
                            output_scores=True,)
        # end_time = time.time()
        generated_sequence = output_base.sequences[0][inputs["input_ids"].shape[1]:]
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")
        # print(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")
        # print(f"token per second: {generated_sequence.shape[0]/(end_time - start_time)}")
        return self.tokenizer.decode(generated_sequence), len(generated_sequence)

