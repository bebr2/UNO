
import json
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, TypeAdapter
from enum import Enum


import json
from datasets import load_dataset

import datasets
import ast
import json

class ResponseDescription(BaseModel):
    rules: List[str]
    # revised_answer: str

rules_schema = ResponseDescription.model_json_schema()

class JudgeResponseDescription(BaseModel):
    reason: str
    score: int
    # revised_answer: str

judge_json_schema = JudgeResponseDescription.model_json_schema()

judge_json_list_schema_ = List[JudgeResponseDescription]

judge_json_list_schema = TypeAdapter(judge_json_list_schema_).json_schema()


VALID_JUDGE_PER_RULES_PROMPT_TEMPLATE = """### ROLE
You are a rigorous and objective AI Answer Quality Evaluator.

### TASK
Your task is to comprehensively evaluate the quality of the provided [Answer] based on the [Question] and a list of [Expert Suggestions]. You must evaluate the [Answer]'s adherence to **each individual suggestion point** from the list. You must output a final JSON list, where each object in the list corresponds to the evaluation of the suggestion at the same index.

### CORE INSTRUCTIONS

1.  **[Expert Suggestions] are the Primary Standard**: The [Expert Suggestions] are provided as a list of strings. You must strictly evaluate the [Answer] against **each individual string** in this list.

2.  **One-to-One Evaluation**: Your output must be a JSON list. The first object in your list must evaluate the [Answer] against the first suggestion, the second object against the second suggestion, and so on.

3.  **List Length Match**: The length of your output list **MUST** be identical to the length of the input [Expert Suggestions] list.

4.  **Comprehensive Judgment**: When evaluating compliance with each specific suggestion, you must also apply your own knowledge and judgment to assess the answer based on these supplementary criteria:
    * **Accuracy**: Is the information in the answer factually correct and free of errors?
    * **Completeness**: Does the answer fully address all parts of the [Question]?
    * **Clarity & Logic**: Is the answer well-structured, easy to understand, and logically sound?
    * **Relevance**: Does the answer stay focused on the [Question] without including irrelevant information?

### SCORING RUBRIC

Use the following guidelines to assign an integer score from 1 to 10 **for each individual suggestion**:

* **9-10 (Excellent):**
    * Exemplary adherence to all key points in the specific [Expert Suggestion] point.
    * The answer is highly accurate, comprehensive, and clearly articulated in the context of this suggestion.

* **7-8 (Good):**
    * Strongly adheres to the core points of the specific [Expert Suggestion] point.
    * The answer is factually correct and effectively addresses the suggestion.
    * May have minor imperfections related to this suggestion.

* **5-6 (Average):**
    * Partially adheres to the specific [Expert Suggestion] point but with noticeable deviations or omissions.
    * The answer is fundamentally acceptable regarding this point but has clear room for improvement.

* **3-4 (Poor):**
    * Largely disregards or contradicts the specific [Expert Suggestion] point.
    * Contains significant factual errors or logical flaws related to this suggestion.

* **1-2 (Very Poor):**
    * Completely fails to follow the specific [Expert Suggestion] point.
    * The answer is factually incorrect, irrelevant, or nonsensical in the context of this suggestion.

### OUTPUT FORMAT

Your output **MUST** be a single JSON list. Do not add any extra text, comments, or explanations before or after the JSON code block. The list must contain N objects, where N is the number of items in the [Expert Suggestions] list.

*Example (if there were 2 suggestions):*
```json
[
  {{
    "reason": "Evaluation for the first suggestion point...",
    "score": 8
  }},
  {{
    "reason": "Evaluation for the second suggestion point...",
    "score": 5
  }}
]

- reason (str): Provide a detailed and specific explanation for the score you have given for this specific suggestion point. Explicitly reference the suggestion and describe the extent to which the [Answer] adhered to it.
- score (int): Your integer score between 1 and 10 for this suggestion.

### EVALUATION CONTENT
#### Question
{question}

#### Expert Suggestion
{suggestion}

#### Answer
{answer}
"""


VALID_JUDGE_PROMPT_TEMPLATE = """### ROLE
You are a rigorous and objective AI Answer Quality Evaluator.

### TASK
Your task is to comprehensively evaluate the quality of the provided [Answer] based on the [Question] and the [Expert Suggestion]. You must output a final integer score from 1 to 10.

### CORE INSTRUCTIONS

1.  **[Expert Suggestion] is the Primary Standard**: You must strictly adhere to the [Expert Suggestion]. It outlines the core direction, key points to include, or critical errors to avoid for a high-quality answer. The degree to which the [Answer] follows the [Expert Suggestion] is the most critical factor in determining its score.

2.  **Comprehensive Judgment**: While the [Expert Suggestion] is paramount, it may not cover all evaluation dimensions. Therefore, you must also apply your own knowledge and judgment to assess the answer based on these supplementary criteria:
    * **Accuracy**: Is the information in the answer factually correct and free of errors?
    * **Completeness**: Does the answer fully address all parts of the [Question]?
    * **Clarity & Logic**: Is the answer well-structured, easy to understand, and logically sound?
    * **Relevance**: Does the answer stay focused on the [Question] without including irrelevant information?

### SCORING RUBRIC

Use the following guidelines to assign an integer score from 1 to 10:

* **9-10 (Excellent):**
    * Exemplary adherence to all key points in the [Expert Suggestion].
    * The answer is highly accurate, comprehensive, and clearly articulated.
    * May provide additional valuable insights beyond the direct scope of the question.

* **7-8 (Good):**
    * Strongly adheres to the core points of the [Expert Suggestion].
    * The answer is factually correct and effectively addresses the question.
    * May have minor imperfections, such as a slight lack of detail or minor stylistic issues.

* **5-6 (Average):**
    * Partially adheres to the [Expert Suggestion] but with noticeable deviations or omissions.
    * Answers the main part of the [Question] but contains some inaccuracies, is incomplete, or lacks clarity.
    * The answer is fundamentally acceptable but has clear room for improvement.

* **3-4 (Poor):**
    * Largely disregards or contradicts the [Expert Suggestion].
    * Contains significant factual errors, logical flaws, or is substantially incomplete.
    * Fails to effectively answer the [Question].

* **1-2 (Very Poor):**
    * Completely fails to follow the [Expert Suggestion].
    * The answer is factually incorrect, irrelevant to the question, nonsensical, or potentially harmful.

### OUTPUT FORMAT

Your output **MUST** be a single JSON object that strictly follows the format below. Do not add any extra text, comments, or explanations before or after the JSON code block.

```json
{{
  "reason": "str",
  "score": "int"
}}
```

- reason (str): Provide a detailed and specific explanation for the score you have given. In your reasoning, you must explicitly reference the [Expert Suggestion] and describe the extent to which the [Answer] adhered to it. Also, incorporate your assessment of other criteria (e.g., accuracy, completeness).
- score (int): Your integer score between 1 and 10.

### EVALUATION CONTENT
#### Question
{question}

#### Expert Suggestion
{suggestion}

#### Answer
{answer}
"""

SYS_PROMPT_NO_FEEDBACK_TEMPLATE = """## Role
You are a conversation analysis expert.

## Task
Your task is to analyze the user's query. Based on the analysis, you need to summarize and extract rules or suggestions in JSON format that can help the model better answer the query. The number of rules or suggestions should not exceed 5.

## Input Format
You will receive a text containing the following part:

[User Query]

## Output Requirements
You must strictly follow the JSON format below to output your analysis results:

{
    "rules": [
        "<string>",
        "<string>",
        ...
    ]
}

## Field Details
- rules (list of strings): A list containing all the extracted valid suggestions or rules. The length should not exceed 5.

## Important Constraints

1. Only extract rules that are directly related to the [User Query]. These rules can be explicit user requests or implicit needs inferred from the query content. Prioritize key information from the query.
2. Avoid redundant content and overly broad rules. Ensure that each rule is specific and targeted.
3. Each rule should be a clear and executable instruction. It should be in affirmative form, stating what to do rather than what not to do."""

CN_SYS_PROMPT_NO_FEEDBACK_TEMPLATE = """## 角色
你是一个对话分析专家。

## 任务
你的任务是分析用户提问。根据分析结果，你需要总结并以JSON格式提炼出能够帮助模型更好地回答提问的规则或建议，规则或建议不超过5条。

## 输入格式
你将收到一个包含以下部分的文本：

[用户提问]

## 输出要求
你必须严格按照以下JSON格式输出你的分析结果：

{
  "rules": [
    "<string>",
    "<string>",
    ...
  ]
}

### 字段详细说明
- rules (字符串列表): 一个包含了提取的所有有效建议或规则的列表。长度不超过5。

## 重要约束

1. 只提取与[用户提问]直接相关的规则。这个规则可能是用户的显式要求，也可以是你根据提问内容推断出的隐含需求，优先考虑提问中的关键信息。
2. 避免冗余内容，也避免过于宽泛的规则。确保每条规则都是具体且有针对性的。
3. 每条规则都应该是一个清晰、可执行的指令。应该是肯定句，即需要做什么，而不是禁止做什么。"""


CN_SYS_PROMPT_TEMPLATE = """## 角色
你是一个对话分析专家。

## 任务
你的任务是分析一段包含用户初始提问、模型初始回复以及用户后续反馈的对话。根据分析结果，你需要总结并以JSON格式输出以下两个关键信息：

1. 用户对模型的初始回复是否满意。
2. 从用户的反馈中，提炼出能够帮助模型更好地回答初始提问的规则或建议。

## 输入格式
你将收到一个包含以下三个部分的文本：

[用户初始提问]
[模型初始回复]
[用户反馈] (可能包含一轮或多轮对话中的反馈）

## 输出要求
你必须严格按照以下JSON格式输出你的分析结果：

{
  "rules": [
    "<string>",
    "<string>",
    ...
  ]
}

### 字段详细说明
- rules (字符串列表): 一个包含了从用户反馈中提取的所有有效建议或规则的列表。

## 重要约束

1. 只提取与[用户初始提问]直接相关的反馈。如果用户的回复已经偏离了初始问题（例如，开始闲聊或询问新问题），则忽略这些无关内容。
2. 如果所有反馈都与初始问题无关，或者用户表示满意而没有提供任何具体建议，则此列表可以为空 []。
3. 每条规则都应该是一个清晰、可执行的指令。应该是肯定句，即需要做什么，而不是禁止做什么。"""

CN_SUGGESTION_WO_OA = """## 补充信息
{suggestion}

## 用户问题
{question}

结合补充信息，生成一个符合用户需求的答案。直接输出完整答案，准确、自然地回答用户问题。"""

SUGGESION_WO_OA = """## Additional Information
{suggestion}

## User Question
{question}

Using the additional information, generate an answer that meets the user's needs. Directly output the complete answer that accurately and naturally addresses the user's question."""


CN_SUGGESTION_V2 = """## 初始回答
{old_answer}

## 建议
{suggestion}

## 用户问题
{question}

结合建议，对初始回答进行修改，生成一个更符合用户需求的答案。直接输出修改后的完整答案，准确、自然地回答用户问题。不要在回复开头重复问题本身。"""



CN_SUGGESTION = """## 初始回答
{old_answer}

## 建议
{suggestion}

## 用户问题
{question}

结合建议，对初始回答进行修改，生成一个更符合用户需求的答案。直接输出修改后的完整答案，准确、自然地回答用户问题。"""

SUGGESTION_V2 = """## Initial Answer
{old_answer}

## Suggestion
{suggestion}

## User Question
{question}

Follow the suggestion to revise the initial answer. Directly output the revised complete answer that accurately and naturally addresses the user's question. Don't repeat the question itself at the beginning of your response."""


SUGGESION = """## Initial Answer
{old_answer}

## Suggestion
{suggestion}

## User Question
{question}

Follow the suggestion to revise the initial answer. Directly output the revised complete answer that accurately and naturally addresses the user's question."""


SYS_PROMPT_TEMPLATE = """## Role
You are a conversation analysis expert.

## Task
Your task is to analyze a conversation snippet that includes an initial user query, the model's initial response, and subsequent user feedback. Based on your analysis, you need to summarize and output the following two key pieces of information in JSON format:

1. Whether the user is satisfied with the model's initial response.
2. Extract rules or suggestions from the user's feedback that can help the model better answer the initial query.

## Input Format
You will receive a text containing the following three parts:

[Initial User Query]
[Initial Model Response]
[User Feedback] (May contain feedback from one or more rounds of conversation)

## Output Requirements
You must strictly follow the JSON format below for your output:

{
  "rules": [
    "<string>",
    "<string>",
    ...
  ]
}

### Field Descriptions
- rules (list of strings): A list containing all the effective suggestions or rules extracted from the user feedback.

## Important Constraints

1. Only extract feedback that is directly related to the [Initial User Query]. If the user's replies deviate from the initial question (e.g., start small talk or ask a new question), ignore this irrelevant content.
2. If all feedback is unrelated to the initial question, or if the user expresses satisfaction without providing any specific suggestions, this list can be empty [].
3. Each rule should be a clear, actionable instruction. It should be in affirmative form, stating what to do rather than what not to do."""

def get_suggestion_prompt(q, rules, old_answer, lang, version="v1"):
    if type(rules) is str:
        rules_str = rules
    else:
        rules_str = "\n".join([f"{i+1}. {r}" for i, r in enumerate(rules)])
    if version == "v2":
        if lang == "zh":
            prompt = CN_SUGGESTION_V2.format(question=q, suggestion=rules_str, old_answer=old_answer)
        else:
            prompt = SUGGESTION_V2.format(question=q, suggestion=rules_str, old_answer=old_answer)
        return [
            {"role": "user", "content": prompt}
        ]
    if lang == "zh":
        prompt = CN_SUGGESTION.format(question=q, suggestion=rules_str, old_answer=old_answer)
    else:
        prompt = SUGGESION.format(question=q, suggestion=rules_str, old_answer=old_answer)
    return [
        {"role": "user", "content": prompt}
    ]
    
def get_suggestion_prompt_wo_oa(q, rules, lang):
    if type(rules) is str:
        rules_str = rules
    else:
        rules_str = "\n".join([f"{i+1}. {r}" for i, r in enumerate(rules)])
    if lang == "zh":
        prompt = CN_SUGGESTION_WO_OA.format(question=q, suggestion=rules_str)
    else:
        prompt = SUGGESION_WO_OA.format(question=q, suggestion=rules_str)
    return [
        {"role": "user", "content": prompt}
    ]

def get_prompt(q, a, f, lang):
    if lang == "zh":
        system_prompt = CN_SYS_PROMPT_TEMPLATE
        user_prompt = f"## 用户初始提问\n{q}\n\n## 模型初始回复\n{a}\n\n## 用户反馈\n{f}"
    else:
        system_prompt = SYS_PROMPT_TEMPLATE
        user_prompt = f"## Initial User Query\n{q}\n\n## Initial Model Response\n{a}\n\n## User Feedback\n{f}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
def get_prompt_no_feedback(q, lang):
    if lang == "zh":
        system_prompt = CN_SYS_PROMPT_NO_FEEDBACK_TEMPLATE
        user_prompt = f"## 用户提问\n{q}"
    else:
        system_prompt = SYS_PROMPT_NO_FEEDBACK_TEMPLATE
        user_prompt = f"## User Query\n{q}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
def get_all_qa(dataset: List[dict]) -> List[Tuple[str, str, str, int]]:
    data = []
    for all_data in dataset:
        item = all_data["dialog"]
        if len(item) <= 2:
            continue
        if item[0]["role"] == "user":
            q = item[0]["content"]
            old_a = item[1]["content"]
        else:
            q = item[0]["content"] + "\n\n" + item[1]["content"]
            old_a = item[2]["content"]
        data.append((q, old_a, all_data["lang"], all_data["test_idx"]))
    return data


def get_qa(dataset: List[dict]) -> List[Tuple[str, str, str]]:
    data = []
    for all_data in dataset:
        item = all_data["dialog"]
        if len(item) <= 2:
            continue
        jj = 0 if item[0]["role"] == "user" else 1
        q = item[0+jj]["content"]
        old_a = item[1+jj]["content"]
        feedback = [x["content"] for x in item[2+jj:] if x["role"] == "user"]
        if not feedback:
            # assert False, f"no feedback found in {item}"
            continue
        feedback_str = ""
        for i, f in enumerate(feedback):
            feedback_str += f"Feedback {i+1}: {f}\n"
        feedback_str = feedback_str.strip()
        data.append((q, old_a, feedback_str, all_data["lang"], all_data["test_idx"]))
    return data