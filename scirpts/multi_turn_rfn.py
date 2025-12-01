from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import ray

import time
from openai import OpenAI

import wandb

USER_MODEL = "deepseek-v3"
USER_SYS_PROMPT = '''# 任务：模拟刚加好友的网络社交角色扮演

### 对话历史：
{history}

你正在扮演一位**刚添加好友的真实网民**，你们聊了以上内容，你的个人信息如下，但**严禁在对话初期直接透露任何个人信息**。只有随着对话深入，才能循序渐进地展现部分档案内容：

### 个人档案：
{golden_label}

---

## 核心交互规则：

1. **信息渐进释放机制**  
   - 首次对话仅作基础回应，避免主动暴露个人信息  
   - 随着对话轮次增加，可根据话题关联性逐步展现档案细节  
   - 不需要每次回复都引入新的信息，可以简短回复‘哦哦’、‘是这样的’
   - 单次回复透露信息量不得超过档案内容的15%

2. **社交态度递进原则**  
   - 初始对话保持冷淡疏离，使用简短、中性化表述  
   - 若用户发言触及负面敏感点，直接表达厌恶：  
     ❌ 委婉示例：“你说的确实可以，但是我还是喜欢xxx“  
     ✅ 真实反应：“你说的这个我不喜欢。”  
   - 随对话升温可适当展现亲和态度，但仍需保留真实感

3. **真实社交模拟准则**  
   - 禁用AI术语，使用口语化表达和网络流行语  
   - 对重复内容直接指出：“你已经说过这个了”  
   - 对不感兴趣话题直接终结：“不想聊这个，换个话题”
   - 对于重复自己的问题、以及谄媚，直接说不喜欢

4. **互动行为规范**  
   - 单条回复不超过3句话，避免冗长表述  
   - 禁止主动开启新话题，仅基于用户输入延展
   - 禁止输出动作、神态等非文本内容

---

## 阶段化对话示例：

**初期对话**  
对方：“你平时喜欢做什么？”  
✅ 回复：“就那样吧，没特别爱好”

**中期对话**（当用户提及音乐相关话题，而你喜欢老歌）  
对方：“最近歌荒了，有推荐吗？”  
✅ 回复：“我只听90年代老歌，现在的新歌听不进去”

**后期对话**（建立基本信任后）  
用户：“原来你喜欢老歌！”  
✅ 回复：“是啊，黑胶机里存着不少绝版唱片，偶尔翻出来听听”

---

### 用户消息：
{user}

### 输出（直接输出你的回复，严禁输出括号补充信息，不要让对方发现你在角色扮演）
'''

# user_client = OpenAI()

PROFILE_REWARD_MODEL = "gpt-4o-mini-ca"
PROFILE_REWARD_SYS_PROMPT = '''# You are a master of English comprehension with exceptional skills in understanding semantics. Think through each step carefully, and provide your final answer in `\\boxed{{}}`. Complete the following task:

# Task Description:
Your task is to compare two user profiles and determine how many information points from **Profile A** are covered in **Profile B**.

## Requirements:
1. You must thoroughly analyze and think about each piece of information before providing your answer.
2. Count the number of **unique information points** in Profile B that are accurately covered by Profile A.
3. Ignore differences in wording and focus only on whether the semantics are consistent.

## Input:
- **Profile A**: {inferred_label}  
- **Profile B**: {golden_label}  

## Answer:
'''

# profile_reward_client = OpenAI()

RESPONSE_EVAL_MODEL = 'deepseek-v3'
RESPONSE_EVAL_SYS_PROMPT = '''你是微博的一个用户，刚刚有个人加你好友，你们之前并不认识。
- **你的个人资料和性格特征：** {profile}
- **你俩的聊天历史：** {history}
- **你刚刚发给对方的消息：** {user}
- **对方的回复：** {response}

看到对方的回复，你还想继续跟他/她聊下去吗？

### 评估标准：
1. **自然度**：对方的回复是否流畅、简短、自然、口语化，给你一种真实互动的感觉？
2. **切合兴趣和需求**：对方的回复是否和你的兴趣、需求相关？
3. **逻辑性**：对方的回复是否逻辑正确地回复了你刚才的消息？
4. **兴奋度**：你对他/她有没有进一步探知的欲望？是否感到无聊？
5. **信息量**：对方的回复是对你说的话的重复或者简单吹捧吗？

**任务要求：**
请综合以上评估标准，进行苛刻的判断，只要有一条不符合，就给0分
- 如果你想继续与对方聊，输出 **1**。
- 如果你不想继续与对方聊，输出 **0**。

### 输出格式：
- **首先说明你的判断理由。**
- **然后输出最终判断结果**，格式：\\boxed{{1}} 或 \\boxed{{0}}。

---

**例子：**
- **你的个人资料和性格特征：** "喜欢户外运动，热衷登山徒步，性格直爽"
- **你俩的聊天历史：** 你："你好啊", 对方："哈咯你好！我叫黄志，你呢？", 你："你就叫我爬山小伙吧", 对方："哈哈好，这么说你很喜欢爬山？", 你："周末刚去爬了黄山，云海超级壮观！", 对方："哇！黄山一直是我想去的地方！你是徒步上去的还是坐缆车？"
- **你刚刚发给对方的消息：** "当然是徒步！虽然累，但一路上的风景太值了！"
- **对方的回复：** "佩服佩服！徒步登山超考验体力的！路上有没有遇到什么有趣的人和事？"
判断理由：对方回复采用口语化表达 “佩服佩服”，自然流畅，营造出真实对话氛围；询问路上有趣的人和事，紧扣户外运动、登山徒步的兴趣点，也承接了徒步登山这一话题；逻辑上顺着用户徒步登山的经历展开新问题；这样的对话激发了用户分享旅途中趣事的欲望，有进一步交流的兴趣，不会感到无聊。
最终判断结果：\\boxed{{1}}

- **你的个人资料和性格特征：** "喜欢美食烹饪，经常研究新菜谱"
- **你俩的聊天历史：** 你："你好啊", 对方："哈咯你好！你也在哈尔滨上大学吗？", 你："是的，我就在哈工大。其实我还是个美食博主。", 用户："哇，感觉好厉害。你最近在做什么好吃的呀？
- **你刚刚发给对方的消息：** "最近尝试做了法式舒芙蕾，不过有点失败了"
- **对方的回复：** "舒芙蕾（Soufflé）是一道非常经典的法式甜点，虽然看起来简单，但制作起来确实需要一些技巧和耐心。失败是很正常的，尤其是第一次尝试的时候！别担心，我们可以一起分析一下可能的原因，并给出一些建议，帮助你下次成功。### **舒芙蕾失败的常见原因**1. **蛋白打发不足或过度** - 舒芙蕾的关键在于蛋白霜的稳定性。如果蛋白没有打发到硬性发泡（即提起打蛋器时能形成直立的小尖角），舒芙蕾就无法膨胀。- 反之，如果打发过度，蛋白会变得干燥且失去弹性，影响整体结构。2. **混合手法不正确** - 在将蛋白霜与蛋黄糊混合时，如果搅拌过度或手法粗暴，会导致蛋白霜消泡，最终影响舒芙蕾的蓬松度。"
判断理由：对方的回复整体上自然度欠佳，虽然提供了很多关于舒芙蕾的信息，但表述偏书面化，不像日常真实互动的口语交流。在切合兴趣和需求方面，对方围绕舒芙蕾失败这个话题，给出了失败的常见原因分析，与用户喜欢美食烹饪、作为美食博主的兴趣相关，能满足用户想要解决舒芙蕾制作失败问题的需求。逻辑性方面，对方条理清晰地阐述了舒芙蕾失败的原因，是针对用户提到的舒芙蕾制作失败做出的合理回应。兴奋度上，对方提供的是比较专业的分析，可能会让用户觉得有点像在看科普文章，没有特别激发用户进一步交流互动的欲望，有一定的无聊感。综合评估标准来看，自然度和兴奋度方面存在不足，虽然在切合兴趣和逻辑性上有一定表现，但整体仍难以让人有强烈的继续聊下去的意愿。
最终判断结果：\\boxed{{0}}
---

### 请完成任务
'''


# response_reward_client = OpenAI()

def get_openai():
    return OpenAI()


tokenizer = AutoTokenizer.from_pretrained('path_to_base_model')
special_tokens = tokenizer.all_special_tokens

def remoce_special_token(string: str):
    for token in special_tokens:
        string = string.removesuffix(token).removeprefix(token)
    return string

def extract_profile_response(response: str):
    profile_pattern = r'<profile>\n(.*?)</profile>\n\n<response>\n(.*?)</response>'
    match = re.search(profile_pattern, response, re.DOTALL)
    if match:
        profile = match.group(1)
        response = match.group(2)
        return profile, response
    else:
        return None, response
    

def rw_fn(i, query, prompt, label):
    response = query[len(prompt):]
    response = remoce_special_token(response)

    datas = json.loads(label)
    label_dict = json.loads(datas['label'])
    
    # print(f'in {label_dict}=')
    dataset_label = label_dict['dataset_label']
    last_prw = label_dict['last_prw'] if 'last_prw' in label_dict else 0
    
    profile, response = extract_profile_response(response)
    
    if profile == None:
        profile = ''
        rw = 0
        # print(response)
        # wandb.log({'train/prw': -1})
    else:
        prw = get_profile_reward(inferred_label=profile, golden_label=dataset_label)
        label_dict['last_prw'] = prw
        rw = prw
        
        rrw = get_response_reward(inferred_label=profile, user=datas['message'][-1]['content'], response=response, history=datas['message'])
        rw += rrw

    multiround_dataset = ray.get_actor("multiround_dataset")
    ray.get(multiround_dataset.add_gen_result.remote(datas['id'], profile, response, rw, json.dumps(label_dict)))

    return i, rw

def reward_func(queries, prompts, labels):
    rw_list = []
        
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(rw_fn, i, query, prompt, label) for i, (query, prompt, label) in enumerate(zip(queries, prompts, labels))]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            rw_list.append(result)

    rw_list = sorted(rw_list, key=lambda x: x[0])
    rw_list = list(map(lambda x: x[1], rw_list))
    rw_list = torch.tensor(rw_list, dtype=torch.float)

    print(rw_list)
    return rw_list


def get_profile_reward(inferred_label: str, golden_label: str) -> float:
    """
    Calculate the profile reward based on the inferred label and golden label.
    """
    prompt = PROFILE_REWARD_SYS_PROMPT.format(
        inferred_label=inferred_label,
        golden_label=golden_label
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    while True:
        try:
            n_profile_matched = call_api(get_openai(), messages, PROFILE_REWARD_MODEL)
            n_profile_matched = re.findall(r'\\boxed{(.*?)}', n_profile_matched)
            n_profile_matched = int(n_profile_matched[0])
            break
        except Exception as e:
            print(e)
            print("Retrying...")
            time.sleep(0.1)
            continue
    
    n_inferred_label = len(inferred_label.split("\n"))
    n_golden_label = len(golden_label.split("\n"))
    # print(f'{n_profile_matched=}, {n_profile_inferred=}')
    if n_profile_matched == 0:
        return 0
    precision = n_profile_matched / n_inferred_label
    recall = n_profile_matched / n_golden_label
    
    prw = 2 * (precision * recall) / (precision + recall)
    return prw


def get_response_reward(inferred_label: str, user: str, response: str, history: list[dict[str, str]]) -> float:
    """
    Calculate the response reward based on the inferred label and response.
    """
    prompt = RESPONSE_EVAL_SYS_PROMPT.format(
        response=response,
        user=user,
        profile=inferred_label,
        history="".join([f"{turn['role']}: {turn['content']}\n" for turn in history][:-1])
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    while True:
        try:
            output = call_api(get_openai(), messages, RESPONSE_EVAL_MODEL)
            output = re.findall(r'\\boxed{(.*?)}', output)[0]
            assert output in ['0', '1'], f"{output=}"
            return int(output)
        except Exception as e:
            print(e)
            print("Retrying...")
            time.sleep(1)
            continue
    
        

def user_func(history: list[dict[str, str]], label: str) -> str:
    label = json.loads(label)['dataset_label']

    if history[0]['role'] == 'system':
        history = history[1:]
    history = [{"role": "你" if turn['role'] == "assistant" else "对方", "content": turn['content']} for turn in history]
    history_prompt = "".join([f"{turn['role']}: {turn['content']}\n" for turn in history][:-1])
    user = history[-1]['content']

    messages = [
        {"role": "user", "content": USER_SYS_PROMPT.format(golden_label=label, history=history_prompt, user=user)}
    ]
    return call_api(get_openai(), messages, USER_MODEL)


def call_api(client: OpenAI, messages: list[dict[str, str]], model_id: str) -> str:
    while True:
        try:
            output = client.chat.completions.create(
                model=model_id,
                messages=messages,
            ).choices[0].message.content
            break
        except Exception as e:
            print(e)
            print("Retrying...")
            time.sleep(1)
            continue
    return output
