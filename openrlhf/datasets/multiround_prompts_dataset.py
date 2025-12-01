from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Union, Callable
import json
import time
import uuid

import ray.actor
from torch.utils.data import Dataset
from tqdm import tqdm
import ray
from copy import deepcopy


def preprocess_data(data, input_template=None, input_key="input", label_key=None, apply_chat_template=None, sys_prompt=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
            if sys_prompt:
                chat.insert(0, {"role": "system", "content": sys_prompt})
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


# PATTERN_MAP = {
#     'qwen': "<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n?"
# }

# model_type = 'qwen'
# reg = re.compile(PATTERN_MAP[model_type], re.DOTALL)

# str_list = []
# msg_list = []

# def remove_chat_template(string: str) -> list[dict[str, str]]:
#     split_str = reg.findall(string)
#     msg = [{'role': s[0], 'content': s[1]} for s in split_str]
#     # print('str: ', string, json.dumps(msg, indent=2))
#     str_list.append(string)
#     with open(f'/data/works_xysui/RL/runs/tempjson/rct_str_list{len(str_list)}.json', 'a', encoding='utf-8') as f:
#         f.write('#############################\n\n\n' + '#############################\n\n\n'.join(str_list))

#     msg_list.append(msg)
#     with open(f'/data/works_xysui/RL/runs/tempjson/rct_msg_list{len(msg_list)}.json', 'w', encoding='utf-8') as f:
#         json.dump(msg_list, f, ensure_ascii=False, indent=2)
#     return msg

# print(remove_chat_template('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nfuck<|im_end|>\n'))




class ReadandDropDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        # self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        

        self.multiround_dataset = ray.get_actor("multiround_dataset")

        ray.get([self.multiround_dataset.set_init_dataset.remote(dataset, input_key, label_key, tokenizer)])
        print(tokenizer)

        # print(f'dataset: {self}')
        # ref = ray.put(self)
        # print(f'{ref=}')

        # ray.get(multiround_dataset.regist_dataset.remote(ref))

    def __len__(self):
        length = ray.get(self.multiround_dataset.get_length.remote())
        print('leng!!!!!!!!!!', length)
        return length

    def __getitem__(self, idx):
        item = ray.get(self.multiround_dataset.get_item.remote())
        return item


    

@ray.remote
class MultiRoundDataset:

    def __init__(self, world_size, n_sample, max_round, remote_user_url, sys_prompt, round_batch_size: int):
        print('MultiRoundDataset init!!!')

        self.world_size = world_size
        self.n_sample = n_sample
        self.max_round = max_round
        self.round_batch_size = round_batch_size if round_batch_size > 0 else 128
        self.by_round = (round_batch_size < 0)
        self.generate_user_round: Callable[[list[dict[str, str]], str], str] = None

        self.get_user_fn(remote_user_url)

        self.conversation_map: dict[str, list[map[str, Union[float, str]]]] = {}
        self.labels = []
        self.prompts = []
        self.tokenizer = None
        self.dataset = None
        self.input_key = None
        self.label_key = None
        self.user_tmp = []

        self.sys_prompt = None
        print(sys_prompt)
        if sys_prompt:
            with open(sys_prompt) as f:
                self.sys_prompt = f.read()
                print(self.sys_prompt)


        self.length = 0

        self.next_idx = 0

    def get_user_fn(self, remote_user_url):
        if isinstance(remote_user_url, list):
            remote_user_url = remote_user_url[0]
        assert remote_user_url.endswith(".py")

        print(f"Loading custom user_func(history_list, slot)` from {remote_user_url}")
        import importlib.util

        spec = importlib.util.spec_from_file_location("user_func", remote_user_url)
        user_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_module)
        self.generate_user_round = user_module.user_func

    def get_item(self):
        if self.next_idx >= len(self.prompts):
            self.gen_user_batch()
        while True:
            try:
                item = (self.prompts[self.next_idx], self.labels[self.next_idx])
                # print('get!!!', f'{self.next_idx=}', f'{len(self.prompts)=}', item)
                self.next_idx += 1
                return item
            except IndexError:
                print('wait...')
                time.sleep(0.5)

    def get_n_sample(self):
        return self.n_sample
    
    def get_world_size(self):
        return self.world_size
    
    def get_max_round(self):
        return self.max_round
    
    def get_length(self):
        return self.length
    
    def _init_dataset(self):
        self.next_idx = 0

        self.conversation_map = {}
        self.prompts = []
        self.labels = []
        self.user_tmp = []
        for d in self.dataset:
            msg = [{"role": "user", "content": d[self.input_key]}]
            if self.sys_prompt is not None:
                msg.insert(0, {"role": "system", "content": self.sys_prompt})
            label = d[self.label_key] if self.label_key else ''
            uid = str(uuid.uuid4())

            self.conversation_map[uid] = {'message': msg, 'id': uid, 'responses': [], 'label': json.dumps({'dataset_label': label})}
            self.prompts.append(self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            self.labels.append(json.dumps({'message': msg, 'id': uid, 'label': json.dumps({'dataset_label': label})}, ensure_ascii=False))

    def set_init_dataset(self, dataset, input_key, label_key, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.input_key = input_key
        self.label_key = label_key
        
        self._init_dataset()

        self.length = len(self.prompts) * self.max_round

    def _gen_user(self, new_msg, choiced_label):
        user = self.generate_user_round(new_msg, choiced_label)
        new_msg += [{'role': 'user', 'content': user}]

        chat_msg = deepcopy(new_msg)
        chat_msg[-2]['content'] = f"<profile>\n{chat_msg[-2]['profile']}</profile>\n\n<response>\n{chat_msg[-2]['content']}</response>"

        new_uid = str(uuid.uuid4())
        self.conversation_map[new_uid] = {'message': new_msg, 'id': new_uid, 'responses': [], 'label': choiced_label}
        if self.by_round:
            self.prompts.append(self.tokenizer.apply_chat_template(chat_msg, tokenize=False, add_generation_prompt=True))
            self.labels.append(json.dumps({'message': new_msg, 'id': new_uid, 'label': choiced_label}, ensure_ascii=False))
        else:
            self.prompts.insert(self.next_idx, self.tokenizer.apply_chat_template(chat_msg, tokenize=False, add_generation_prompt=True))
            self.labels.insert(self.next_idx, json.dumps({'message': new_msg, 'id': new_uid, 'label': choiced_label}, ensure_ascii=False))

    def gen_user_batch(self):
        print('gen_user_batch!!!!!')
        print(self.user_tmp[0])
        print(len(self.user_tmp[0]['msg']))
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(self._gen_user, d['msg'], d['choiced_label']) for d in self.user_tmp]
            for future in tqdm(as_completed(futures), total=len(futures)):
                pass
        self.user_tmp = []

    def add_gen_result(self, uid, profile, response, reward: float, label: str):
        # print(self.conversation_map)
        msg = self.conversation_map[uid]['message']
        response_list = self.conversation_map[uid]['responses']
        response_list.append({'profile': profile, 'response': response, 'reward': reward, 'label': label})
        
        # print('add_gen_result', len(response_list), self.n_sample, len(msg) + 1, self.max_round, len(msg) + 1 < self.max_round * 2, msg)
        self.conversation_map[uid]['responses'] = response_list

        # print(json.dumps(self.conversation_map, ensure_ascii=False))
        
        if len(response_list) >= self.n_sample:
            self.conversation_map.pop(uid)
            # print(f'{len(response_list)=}')
            # print('pop!!!!!!!!!!!!')

            if len(msg) + 1 < self.max_round * 2:
                choiced_response = max(response_list, key=lambda x: x['reward'])
                choiced_label = choiced_response['label']
                choiced_profile = choiced_response['profile']
                choiced_response = choiced_response['response']
                # if len(msg) > 2:
                #     msg[-2]['content'] = re.search( r'<profile>\n(.*?)</profile>\n<response>\n(.*?)</response>', msg[-2]['content'], re.DOTALL).group(2)
                new_msg = msg + [{'role': 'assistant', 'content': choiced_response, 'profile': choiced_profile}]
                # f'<profile>\n{choiced_profile}</profile>\n<response>\n{choiced_response}</response>'

                self.user_tmp.append({'choiced_label': choiced_label, 'msg': new_msg})

                if len(self.user_tmp) >= self.round_batch_size:
                    self.gen_user_batch()
                

                # print('new!!!')
                # print(json.dumps(self.prompts, ensure_ascii=False))
                # print(json.dumps(new_msg, ensure_ascii=False))

    def next_episode(self):
        print('new episode!!!!!!!!!!')
        self._init_dataset()
    