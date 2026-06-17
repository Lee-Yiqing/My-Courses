---
id: B3
title: "Code Llama: Open Foundation Models for Code"
domain: B
year: 2023
arxiv_id: "2308.12950"
confidence: verified
source: "arXiv:2308.12950"
node_type: paper
---

# Code Llama: Open Foundation Models for Code

**Domain**: [[domain_B|LLM Code Generation]] | **Year**: 2023 | **Confidence**: [x] verified


## Authors
[[author_Baptiste Roziere|Baptiste Roziere]], [[author_Jonas Gehring|Jonas Gehring]], [[author_Fabian Gloeckle|Fabian Gloeckle]], [[author_Sten Sootla|Sten Sootla]], [[author_Itai Gat|Itai Gat]], et al.


## Keywords
- [[kw_Code Llama|Code Llama]]
- [[kw_open-source code model|open-source code model]]
- [[kw_infilling|infilling]]
- [[kw_long context|long context]]

## Abstract

(Abstract not available - see PDF content below)

## Paper Content

Code Llama: Open Foundation Models for Code

Baptiste Rozière†, Jonas Gehring†, Fabian Gloeckle†,∗, Sten Sootla†, Itai Gat, Xiaoqing
Ellen Tan, Yossi Adi⋄, Jingyu Liu, Romain Sauvestre, Tal Remez, Jérémy Rapin, Artyom
Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron
Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron,
Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve†

Meta AI

Abstract

We release Code Llama, a family of large language models for code based on Llama 2
providing state-of-the-art performance among open models, infilling capabilities, support
for large input contexts, and zero-shot instruction following ability for programming tasks.
We provide multiple flavors to cover a wide range of applications: foundation models
(Code Llama), Python specializations (Code Llama - Python), and instruction-following
models (Code Llama - Instruct) with 7B, 13B, 34B, and 70B parameters each. These
models are trained on sequences of 16k tokens and show improvements on inputs with
up to 100k tokens. The 7B, 13B and 70B Code Llama and Code Llama - Instruct
variants support infilling based on surrounding content. Code Llama reaches state-of-the-art
performance among open models on several code benchmarks, with scores of up to 67%
and 65% on HumanEval and MBPP, respectively. Notably, Code Llama - Python 7B
outperforms Llama 2 70B on HumanEval and MBPP, and all our models outperform every
other publicly available model on MultiPL-E. We release Code Llama under a permissive
license that allows for both research and commercial use.1

1
Introduction

arXiv:2308.12950v3  [cs.CL]  31 Jan 2024

Large language models (LLMs) power a rapidly increasing number of applications, having reached a proficiency
in natural language that allows them to be commanded and prompted to perform a variety of tasks (OpenAI,
2023; Touvron et al., 2023b). By utilizing large, in-domain datasets, their efficacy can be greatly improved
for applications that require a combination of both natural and domain-specific language and understanding
of specialized terminology. By training on domain-specific datasets, they have proved effective more broadly
on applications that require advanced natural language understanding. A prominent use-case is the formal
interaction with computer systems, such as program synthesis from natural language specifications, code
completion, debugging, and generating documentation (for a survey, see Xu & Zhu, 2022, also see Section 5).
In this work, we present Code Llama, a family of LLMs for code generation and infilling derived from
Llama 2 (Touvron et al., 2023b) and released under the same custom permissive license. We provide inference
code for both completion and infilling models in the accompanying repository.1 Our approach is based on
gradually specializing and increasing the capabilities of Llama 2 models by applying a cascade of training
and fine-tuning steps (Figure 2):

• Code-training from foundation models. While most LLMs for code generation such as AlphaCode
(Li et al., 2022), InCoder (Fried et al., 2023) or StarCoder (Li et al., 2023) are trained on code only,
Codex (Chen et al., 2021) was fine-tuned from a general language model. We also start from a foundation
model (Llama 2, Touvron et al., 2023b) pretrained on general-purpose text and code data. Our comparison
(Section 3.4.1) shows that initializing our model with Llama 2 outperforms the same architecture trained
on code only for a given budget.

1https://github.com/facebookresearch/codellama
†: Core contributors. ∗: Meta AI, CERMICS École des Ponts ParisTech. ⋄: Meta AI & Hebrew University of Jerusalem

1

Prompt: In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the
last month

Response:
You can use the ‘find‘ command in Bash to list all text files in the current directory (excluding subdirectories) that
have been modified in the last month. The command would be:

find . -maxdepth 1 -type f -mtime -30 -name "*.txt"

Here’s a breakdown of the command:
* ‘.‘ specifies the current directory.
* ‘-maxdepth 1‘ tells ‘find‘ to only search the current directory and not any subdirectories.
* ‘-type f‘ tells ‘find‘ to only search for files (not directories or other types of files).
* ‘-mtime -30‘ tells ‘find‘ to only search for files that have been modified in the last 30 days.
* ‘-name "*.txt"‘ tells ‘find‘ to only search for files with the ‘.txt‘ extension.
Note that the ‘-mtime‘ option uses the number of days since the file was last modified, so ‘-mtime -30‘ means "modified in the
last 30 days". If you want to search for files modified in the last month, you can use the ‘-mtime -30‘ option.

Figure 1: Example of response of Code Llama - Instruct (34B) when queried for a specific shell command.

• Infilling. Autoregressive training and fine-tuning of LLMs is suitable for prompt completion, but does
not provide the capability to fill a missing portion of text while taking the full surrounding context into
account. Our code-training for 7B, 13B and 70B Code Llama models features a multitask objective
(Fried et al., 2023) consisting of both autoregressive and causal infilling prediction, enabling applications
such as real-time completion in source code editors or docstring generation.
• Long input contexts. Unlocking repository-level reasoning for completion or synthesis – as opposed
to function-level or file-level – requires prompting the model with much longer context than the 4,096
tokens supported by Llama 2. We propose an additional fine-tuning stage that extends the maximum
context length from 4,096 tokens to 100,000 tokens by modifying the parameters of the RoPE positional
embeddings (Su et al., 2021) used in Llama 2. Our experiments show Code Llama operating on very
large contexts with a moderate impact on performances on standard coding benchmarks (Section 3.3).
• Instruction fine-tuning. For end-users, the utility of LLMs is significantly improved by instruction
fine-tuning (Ouyang et al., 2022; Wei et al., 2022; OpenAI, 2023; Touvron et al., 2023b), which also helps
preventing unsafe, toxic or biased generations. Code Llama - Instruct variants are further fine-tuned
on a mix of proprietary instruction data for improved safety and helpfulness, and a new machine-generated
self-instruct dataset created by prompting Llama 2 for coding problems and Code Llama to generate
associated unit tests and solutions. Our results show that Code Llama - Instruct significantly improves
performance on various truthfulness, toxicity and bias benchmarks at moderate cost in terms of code
generation performance (Section 4).

Different combinations of these approaches lead to a family of code-specialized Llama 2 models with three
main variants that we release in four sizes (7B, 13B, 34B and 70B parameters):

• Code Llama: a foundational model for code generation tasks,
• Code Llama - Python: specialized for Python,
• Code Llama - Instruct: fine-tuned with human instructions and self-instruct code synthesis data.

An example of using Code Llama - Instruct is given in Figure 1. It show-cases that the model interprets
natural language to determine suitable options for a command-line program and provides an explanation of the
solution. We provide further qualitative examples in Appendix L. We perform exhaustive evaluations of our
models on major code generation benchmarks: HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021),
and APPS (Hendrycks et al., 2021), as well as a multilingual version of HumanEval (MultiPL-E, Cassano
et al., 2023), where our best models establish a new state of the art amongst open-source LLMs. The technical
details of our training and fine-tuning procedures are provided in Section 2, followed by in-depth experiments
and ablation studies, details of the safety/helpfulness evaluations and a discussion of related work.

2

Long context

Cᴏᴅᴇ Lʟᴀᴍᴀ 
(7B ⇄, 13B ⇄, 34B)

fine-tuning

20B

Instruction 
Fine-tuning

Code training
Infilling code training      .

Cᴏᴅᴇ Lʟᴀᴍᴀ - Iɴsᴛʀᴜᴄᴛ 
(7B ⇄, 13B ⇄, 34B)

Lʟᴀᴍᴀ 2
Foundation models

⇄

5B

(7B, 13B, 34B)

500B

Python code

Long context

training

Fine-tuning

Cᴏᴅᴇ Lʟᴀᴍᴀ - Pʏᴛʜᴏɴ
(7B, 13B, 34B)

100B

20B

Figure 2: The Code Llama specialization pipeline. The different stages of fine-tuning annotated with
the number of tokens seen during training. Infilling-capable models are marked with the ⇄symbol.

2
Code Llama: Specializing Llama 2 for code

2.1
The Code Llama models family

Code Llama.
The Code Llama models constitute foundation models for code generation. They come
in four model sizes: 7B, 13B, 34B and 70B parameters. The 7B, 13B and 70B models are trained using an
infilling objective (Section 2.3), and are appropriate to be used in an IDE to complete code in the middle of
a file, for example. The 34B model was trained without the infilling objective. All Code Llama models
are initialized with Llama 2 model weights and trained on 500B tokens from a code-heavy dataset (see
Section 2.2 for more details), except Code Llama 70B which was trained on 1T tokens. They are all
fine-tuned to handle long contexts as detailed in Section 2.4.

Code Llama - Python.
The Code Llama - Python models are specialized for Python code generation
and come in sizes of 7B, 13B, 34B and 70B parameters. They are designed to study the performance of models
tailored to a single programming language, compared to general-purpose code generation models. Initialized
from Llama 2 models and trained on 500B tokens from the Code Llama dataset, Code Llama - Python
models are further specialized on 100B tokens using a Python-heavy dataset (Section 2.2). Code Llama Python with 7B, 13B and 34B parameters are trained without infilling and subsequently fine-tuned to
handle long contexts (Section 2.4).

Code Llama - Instruct.
For the 7B, 13B and 34B sizes, Code Llama - Instruct models are based
on Code Llama and fine-tuned with an additional approx. 5B tokens to better follow human instructions.
More details on Code Llama - Instruct can be found in Section 2.5.

Code Llama 70B.
Code Llama 70B was trained months after the Code Llama 7B, 13B and 34B
model. It was trained using the same data as the smaller versions of Code Llama, and using roughly
the same methods. Code Llama 70B was trained on twice the number of tokens: 1 trillion instead of
500 billion. It was trained with FIM, which was an often-requested capability for the 34B model. Only
the base Code Llama 70B was trained with LCFT. See Appendix B for Code Llama 70B specialization
pipeline. Code Llama - Instruct 70B was trained from Code Llama - Python 70B, which outperforms
Code Llama 70B in average on the languages from MultiPL-E including python.

2.2
Dataset

We train Code Llama 7B, 13B and 34B on 500B tokens, and Code Llama 70B on 1T tokens during
the initial phase, starting from the 7B, 13B, 34B, and 70B versions of Llama 2. As shown in Table 1,
Code Llama is trained predominantly on a near-deduplicated dataset of publicly available code. We also
source 8% of our samples data from natural language datasets related to code. This dataset contains many
discussions about code and code snippets included in natural language questions or answers. To help the
model retain natural language understanding skills, we also sample a small proportion of our batches from a
natural language dataset. Data is tokenized via byte pair encoding (BPE, Sennrich et al. (2016)), employing
the same tokenizer as Llama and Llama 2. Preliminary experiments suggested that adding batches sampled
from our natural language dataset improves the performance of our models on MBPP.

3

2.3
Infilling

Code infilling is the task of predicting the missing part of a program given a surrounding context. Applications
include code completion at the cursor’s position in code IDEs, type inference and generation of in-code
documentation (e.g., docstrings).

We train infilling models following the concept of causal masking (Aghajanyan et al., 2022; Fried et al.,
2023), where parts of a training sequence are moved to the end, and the reordered sequence is predicted
autoregressively. We train the general-purpose 7B, 13B and 70B models with an infilling objective, following
the recommendations of Bavarian et al. (2022). More precisely, we split training documents at the character
level into a prefix, a middle part and a suffix with the splitting locations sampled independently from a
uniform distribution over the document length. We apply this transformation with a probability of 0.9 and
to documents that are not cut across multiple model contexts only. We randomly format half of the splits in
the prefix-suffix-middle (PSM) format and the other half in the compatible suffix-prefix-middle (SPM) format
described in Bavarian et al. (2022, App. D). We extend Llama 2’s tokenizer with four special tokens that
mark the beginning of the prefix, the middle part or the suffix, and the end of the infilling span. To limit the
distribution shift between autoregressive and infilling training, we suppress the implicit leading space that
SentencePiece tokenizers add upon encoding the middle part and the suffix (Kudo & Richardson, 2018). In
SPM format, we concatenate the prefix and the middle part before encoding to tokens. Note that our model
doesn’t encounter split subtokens in the SPM format while it does in the PSM format.

Results on the effect of infilling training on downstream generation tasks and the performance of our infilling
models on infilling benchmarks are reported in Section 3.2.

2.4
Long context fine-tuning

Effective handling of long sequences is a major topic of research in transformer-based language modeling (Vaswani et al., 2017). The fundamental modeling challenges are extrapolation, i.e., operating on sequence
lengths beyond those seen at training time, and the quadratic complexity of attention passes which favors
training on short-to-medium length inputs.

For Code Llama, we propose a dedicated long context fine-tuning (LCFT) stage in which models are
presented with sequences of 16,384 tokens, up from the 4,096 tokens used for Llama 2 and our initial code
training stages. By limiting the training time spent on processing long sequences to a fine-tuning stage, we
gain long-range capabilities without significantly increasing the cost of training our models. Our strategy is
similar to the recently proposed fine-tuning by position interpolation (Chen et al., 2023b), and we confirm
the importance of modifying the rotation frequencies of the rotary position embedding used in the Llama 2
foundation models (Su et al., 2021). However, instead of downscaling frequencies linearly as Chen et al.
(2023b), we change the base period from which they are derived. Specifically, with rotary embeddings, the
query and key vectors xn at position n are subject to a linear transformation Rd
Θ,nxn, where Rd
Θ,n is a block
diagonal matrix with entries of the form


,

 
Rd
Θ,n


i =
cos nθi
−sin nθi
sin nθi
cos nθi

and d denotes the embedding dimension. Rotation frequencies are computed as θi = θ−2i/d, and we increase
the base period θ from 10,000 to 1,000,000 for fine-tuning. This increase allows for processing much larger
sequences and reduces bias towards short-distance attention (see Appendix G.1 for further discussion). Our
experiments confirm that Code Llama models are not only effective within the increased sequence length
used during fine-tuning, but further show extrapolation capabilities and exhibit stable behavior on very long
sequences of up to 100,000 tokens (Section 3.3).

2.5
Instruction fine-tuning

Our instruction fine-tuned models Code Llama - Instruct are based on Code Llama and trained to
answer questions appropriately. They are trained on three different types of data.

4

Dataset
Sampling prop. Epochs Disk size

Code Llama (500B tokens)
Code
85%
2.03
859 GB
Natural language related to code
8%
1.39
78 GB
Natural language
7%
0.01
3.5 TB

Code Llama - Python (additional 100B tokens)
Python
75%
3.69
79 GB
Code
10%
0.05
859 GB
Natural language related to code
10%
0.35
78 GB
Natural language
5%
0.00
3.5 TB

Table 1: Training dataset of Code Llama and Code Llama - Python. We train Code Llama on
500B additional tokens and Code Llama - Python further on 100B tokens.

Proprietary dataset.
We use the instruction tuning dataset collected for Llama 2 and described in detail
by Touvron et al. (2023b). Specifically, we use the version referred to in their paper as “RLHF V5”, collected
through several stages of reinforcement learning from human feedback and human feedback annotation
(see their Section 3 for more details). It combines thousands of Supervised Fine-Tuning and millions of
Rejection Sampling examples. Each example consists of a multi-turn dialogue between a user and an assistant.
For Rejection Sampling, the output was selected among several generations using a reward model. The
final dataset contains both Helpfulness and Safety data. This enables Code Llama to inherit Llama 2’s
instruction following and safety properties.

Self-instruct.
Our proprietary dataset contains few examples of code-related tasks. Collecting supervised
data from human annotators or training from human feedback (Ouyang et al., 2022) is expensive for coding
tasks as it requires input from professional developers. Instead of human feedback, we use execution feedback
to select data to train our instruct model. We construct the self-instruction dataset following the recipe
below, resulting in ∼14,000 question-tests-solution triplets:

1. Generate 62,000 interview-style programming questions by prompting (Figure 10) Llama 2 70B.

2. De-duplicate the set of questions by removing exact duplicates, resulting in ∼52,000 questions.

3. For each of these questions:

(a) Generate unit tests by prompting Code Llama 7B (Figure 11)
(b) Generate ten Python solutions by prompting Code Llama 7B (Figure 12)

(c) Run the unit tests on the ten solutions. Add the first solution that passes the tests (along with
its corresponding question and tests) to the self-instruct dataset.

We use Code Llama 7B to generate the tests and Python solutions, as we found it more efficient than
generating fewer solutions per question with the 34B model for the same compute budget.

Rehearsal.
In order to prevent the model from regressing on general coding and language understanding
capabilities, Code Llama - Instruct is also trained with a small proportion of data from the code dataset
(6%) and our natural language dataset (2%).

2.6
Training details

Optimization.
Our optimizer is AdamW (Loshchilov & Hutter, 2019) with β1 and β2 values of 0.9 and
0.95. We use a cosine schedule with 1000 warm-up steps, and set the final learning rate to be 1/30th of the
peak learning rate. We use a batch size of 4M tokens which are presented as sequences of 4,096 tokens each.
Despite the standard practice of using lower learning rates in fine-tuning stages than in pre-training stages,

5

Model
Size
HumanEval
MBPP
pass@1 pass@10 pass@100 pass@1 pass@10 pass@100

code-cushman-001
12B
33.5%
45.9%
GPT-3.5 (ChatGPT)
48.1%
52.2%
GPT-4
67.0%
PaLM
540B
26.2%
36.8%
PaLM-Coder
540B
35.9%
88.4%
47.0%
PaLM 2-S
37.6%
88.4%
50.0%
StarCoder Base
15.5B
30.4%
49.0%
StarCoder Python
15.5B
33.6%
52.7%
StarCoder Prompted
15.5B
40.8%
49.5%

Llama 2

7B
12.2%
25.2%
44.4%
20.8%
41.8%
65.5%
13B
20.1%
34.8%
61.2%
27.6%
48.1%
69.5%
34B
22.6%
47.0%
79.5%
33.8%
56.9%
77.6%
70B
30.5%
59.4%
87.0%
45.4%
66.2%
83.1%

Code Llama

7B
33.5%
59.6%
85.9%
41.4%
66.7%
82.5%
13B
36.0%
69.4%
89.8%
47.0%
71.7%
87.1%
34B
48.8%
76.8%
93.0%
55.0%
76.2%
86.6%
70B
53.0%
84.6%
96.2%
62.4%
81.1%
91.9%

Code Llama - Instruct

7B
34.8%
64.3%
88.1%
44.4%
65.4%
76.8%
13B
42.7%
71.6%
91.6%
49.4%
71.2%
84.1%
34B
41.5%
77.2%
93.5%
57.0%
74.6%
85.4%
70B
67.8%
90.3%
97.3%
62.2%
79.6%
89.2%
Unnatural Code Llama
34B
62.2%
85.2%
95.4%
61.2%
76.6%
86.7%

Code Llama - Python

7B
38.4%
70.3%
90.6%
47.6%
70.3%
84.8%
13B
43.3%
77.4%
94.1%
49.0%
74.0%
87.6%
34B
53.7%
82.8%
94.7%
56.2%
76.4%
88.2%
70B
57.3%
89.3%
98.4%
65.6%
81.5%
91.9%

Table 2: Code Llama pass@ scores on HumanEval and MBPP. The pass@1 scores of our models are
computed with greedy decoding. The pass@10 and pass@100 scores are computed with nucleus sampling
with p=0.95 and temperature 0.8 following our findings from Figure 6. Models are evaluated in zero-shot
on Human Eval and 3-shot on MBPP. The instruct models are trained to be safe and aligned from the
base Code Llama models. Results for other models as provided by Li et al. (2023) (code-cushman-001,
StarCoder), OpenAI (2023) (GPT-3.5, GPT-4), and Chowdhery et al. (2022); Anil et al. (2023) (PaLM).

we obtained best results when retaining the original learning rate of the Llama 2 base model. We carry these
findings to the 13B, 34B and 70B models, and set their learning rates to 3e−4, 1.5e−4, and 1.5e−4 respectively.
For python fine-tuning, we set the initial learning rate to 1e−4 instead. For Code Llama - Instruct, we
train with a batch size of 524,288 tokens and on approx. 5B tokens in total.

Long context fine-tuning.
For long context fine-tuning (LCFT), we use a learning rate of 2e−5, a sequence
length of 16,384, and reset RoPE frequencies with a base value of θ = 106. The batch size is set to 2M tokens
for model sizes 7B and 13B and to 1M tokens for model size 34B, respectively. Training lasts for 10,000
gradient steps by default. We observed instabilities in downstream performance for certain configurations,
and hence set the number of gradient steps to 11,000 for the 34B models and to 3,000 for Code Llama 7B.

3
Results

We report results on a variety of benchmarks. First, we evaluate our models on popular description-to-code
generation benchmarks for Python: HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and APPS

6

(programming interviews and competitions, Hendrycks et al., 2021). Second, we evaluate our models on
further programming languages using MultiPL-E (Cassano et al., 2023), namely on C++, Java, PHP, C#,
TypeScript (TS), and Bash. We additionally report results on the GSM8K benchmark (Cobbe et al., 2021),
which measures mathematical reasoning capabilities (Appendix D).

Next, we perform an extensive ablation study: (i) we study the impact of training from scratch or from a
pretrained Llama 2 model in Section 3.4.1; (ii) we perform ablations for infilling and additional infilling
specific benchmarks in Section 3.2; (iii) we study the effect of long context fine-tuning on perplexity, a
synthetic retrieval task, and code completion with long source code files (Section 3.3); and (iv) we evaluate
our instruction fine-tuning procedure, which includes self-instruct training by leveraging self-generated unit
tests in Section 3.4.2.

3.1
Code generation

3.1.1
Python code generation

We start by reporting results for Python code generation using the HumanEval (Chen et al., 2021),
MBPP (Austin et al., 2021) and APPS (Hendrycks et al., 2021) benchmarks. Results are summarized
in Tables 2 and 3. The full list of results on HumanEval and MBPP, including models with and without
infilling and long context fine-tuning, can be found in Table 10 in Appendix C. We provide zero-shot results
of our instruction fine-tuned models on APPS in Table 15 with evaluation details in Appendix F. Our main
findings are as follows.

The value of model specialization.
We observe that model specialization is yields a boost in code
generation capabilities when comparing Llama 2 to Code Llama and Code Llama to Code Llama Python. Llama 2 was trained on 2T tokens, and training on only 500B of extra tokens from a code-heavy
dataset results in massive performance gains on both HumanEval and MBPP, to the point that Llama 2
70B is roughly equivalent to Code Llama 7B on Python coding benchmarks. Although Code Llama was
trained on more than two epochs of our code dataset, which contains our entire Python dataset, training
on 100B extra tokens of a Python-heavy data mix leads to significant gains on Python code generation
benchmarks, between 4.3% points and 8.3% points in HumanEval pass@1 and between 1.2% points and
6.4% points in MBPP pass@1. These gains are smaller than for the first code training step, but still allow
Code Llama - Python 7B to outperform even Code Llama 13B on MBPP and HumanEval. For the APPS
benchmark, the prompts are much less direct and more complex compared to MBPP and HumanEval. Our
Code Llama - Python models show slightly decreased performance on the introductory and interview level
problems, where understanding the prompt is often more challenging for a language model than implementing
a solution. However, Code Llama - Python shows clear gains on the competition-level problems where
solutions are more complex. While large language models have enough capacity to learn to generate text on
various topics, we observe that model specialization is beneficial for models between 7B and 70B parameters
and after two full epochs on the training data.

Scaling of specialized models.
We observe that scaling the number of parameters matters for models
specialized for coding. With the same training process, our larger models outperform their smaller counterparts
on almost every metric from HumanEval, MBPP and APPS (Table 2, 3). For instance, we gain 5.6 percentage
points on MBPP pass@1 scaling Code Llama from 7B to 13B parameters, 8 more points when scaling
to 34B and 7 when scaling to 70B. We can hypothesize that specializing larger models to code would lead
to significant further gains on coding tasks. Moreover, the Chinchilla scaling laws (Hoffmann et al., 2022)
indicate that larger models would benefit more from training on more tokens.

3.1.2
Multilingual evaluation

Next, we evaluate our models on a more diverse set of programming languages. For that, we use the MultiPL-E
benchmark (Cassano et al., 2023). We report results for Python, C++, Java, PHP, TypeScript, C#, and
Bash in Table 4.

7

Model
Size Pass@ Introductory Interview Competition

GPT-Neo
2.7B
1
3.9%
0.6%
0.0%
5
5.5%
0.8%
0.0%

Codex
12B
1
4.1%
0.1%
0.0%
5
9.7%
0.5%
0.1%
1000
25.0%
3.7%
3.2%

AlphaCode

1B

1000
17.7%
5.2%
7.1%
AlphaCode (Filtered 1000)
5
14.4%
5.6%
4.6%
AlphaCode (Filtered 10000)
5
18.2%
8.2%
6.7%
AlphaCode (Filtered 50000)
5
20.4%
9.7%
7.8%

7B
5
10.8%
2.0%
0.8%
10
15.6%
3.1%
1.4%
100
33.5%
9.4%
7.1%

Code Llama

13B
5
23.7%
5.6%
2.1%
10
30.2%
8.1%
3.4%
100
49.0%
18.4%
12.0%

34B
5
32.8%
8.8%
2.9%
10
39.0%
12.2%
4.7%
100
56.3%
24.3%
15.4%

7B
5
12.7%
4.2%
1.3%
10
18.5%
6.3%
2.2%
100
38.3%
14.9%
9.1%

Code Llama - Python

13B
5
26.3%
7.1%
2.8%
10
32.8%
10.0%
4.3%
100
51.6%
21.5%
14.6%

34B
5
28.9%
7.8%
3.5%
10
35.9%
11.1%
5.5%
100
54.9%
23.9%
16.8%

7B
5
12.9%
2.1%
1.1%
10
17.9%
3.1%
2.0%
100
35.4%
9.4%
8.5%

Code Llama - Instruct

13B
5
24.0%
6.9%
2.4%
10
30.3%
9.6%
3.8%
100
48.7%
19.6%
13.1%

34B
5
31.6%
7.9%
3.2%
10
37.8%
11.1%
5.1%
100
55.7%
22.8%
16.4%

Table 3: Code Llama pass@ scores on APPS. We list the two-shot pass@5, pass@10, and pass@100
scores of Code Llama on APPS. For our models, we use nucleus sampling with p=0.95 and a temperature
of 0.6. Code Llama is not fine-tuned on the training set of APPS and all results are calculated with raw
predictions without filtering by the test cases from the prompt. Fine-tuned GPT-Neo numbers are reported
by Hendrycks et al. (2021), one-shot Codex results by Chen et al. (2021), and fine-tuned AlphaCode numbers
by Li et al. (2022).

We observe a similar improvement from Llama 2 to Code Llama in the multilingual setting as in the
evaluation on Python (Section 3.1.1). The Code Llama models clearly outperform Llama 2 models of the
same size on code generation in any language, and Code Llama 7B even outperforms Llama 2 70B. Compared

8

Model
Size
Multi-lingual Human-Eval
C++
Java
PHP
TS
C#
Bash
Average

CodeGen-Multi
16B 21.0%
22.2%
8.4%
20.1%
8.2%
0.6%
13.4%
CodeGeeX
13B 16.9%
19.1%
13.5%
10.1%
8.5%
2.8%
11.8%
code-cushman-001
12B 30.6%
31.9%
28.9%
31.3%
22.1%
11.7%
26.1%
StarCoder Base
15.5B 30.6%
28.5%
26.8%
32.2%
20.6%
11.0%
25.0%
StarCoder Python
15.5B 31.6%
30.2%
26.1%
32.3%
21.0%
10.5%
25.3%

Llama-v2

7B
6.8%
10.8%
9.9%
12.6%
6.3%
3.2%
8.3%
13B 13.7%
15.8%
13.1%
13.2%
9.5%
3.2%
11.4%
34B 23.6%
22.2%
19.9%
21.4%
17.1%
3.8%
18.0%
70B 30.4%
31.7%
34.2%
15.1%
25.9%
8.9%
24.4%

Code Llama

7B 28.6%
34.2%
24.2%
33.3%
25.3%
12.0%
26.3%
13B 39.1%
38.0%
34.2%
29.6%
27.3%
15.2%
30.6%
34B 47.8%
45.6%
44.1%
33.3%
30.4%
17.1%
36.4%
70B 52.8%
51.9%
50.9% 49.1% 38.0% 29.1%
45.3%

Code Llama - Instruct

7B 31.1%
30.4%
28.6%
32.7%
21.6%
10.1%
25.8%
13B 42.2%
40.5%
32.3%
39.0%
24.0%
13.9%
32.0%
34B 45.3%
43.7%
36.6%
40.3%
31.0%
19.6%
36.1%
70B 53.4% 58.2% 58.4% 39.0%
36.7% 29.7%
45.9%

Code Llama - Python

7B 32.3%
35.4%
32.3%
23.9%
24.7%
16.5%
27.5%
13B 39.1%
37.3%
33.5%
35.2%
29.8%
13.9%
31.5%
34B 42.2%
44.9%
42.9%
34.3%
31.7%
14.6%
35.1%
70B 54.7% 57.6%
53.4%
44.0%
34.8%
25.3%
45.0%

Table 4: Multi-Lingual HE Pass@1 scores. Pass@1 scores for different programming languages using
greedy decoding. These scores are computed in zero-shot. Results for other models from Li et al. (2023).

to other publicly available models, ours are especially strong in the multilingual setting. Code Llama 7B
outperforms larger models such as CodeGen-Multi or StarCoder, and is on par with Codex (code-cushman-001,
Chen et al., 2021).

The performance of Code Llama - Python is comparable to that of Code Llama. Code Llama Python 30B performs slightly worse than Code Llama but Code Llama - Python 7B and 13B perform
slightly better than their counterparts without Python fine-tuning. More detailed results can be found in
Table 11, Appendix C.

To better understand the influence of multilingual pre-training, we measure the correlations between each of
the evaluated languages and report the results separately for different model sizes in Figure 3. We observe
high correlation between model performance on C++, C#, Java, and PHP. Interestingly, we also notice
strong correlation between model performance on Python and Bash. Lastly, as expected the bigger and more
expressive the models, the higher the correlation between the performance across all different languages.

3.2
Infilling evaluations

Performance cost of infilling training.
Previous studies on infilling (or fill-in-the-middle, FIM) code
models assert that the traditional next token prediction objective can be replaced by a multitask infilling
objective with an infilling rate of up to 90 % at no cost for left-to-right autoregressive test losses (Bavarian
et al., 2022) and only small cost for downstream evaluation performance (Allal et al., 2023). In Table 5, we
independently validate both findings at the scale of 7B and 13B parameters and 500B training tokens of
code. The 7B model loses 0.6 percentage points on average across HumanEval and MBPP pass@1, pass@10
and pass@100 scores if trained with an infilling objective, while the 13B model loses 1.1 percentage points.

9

Model Size: 7B

Model Size: 13B

Model Size: 34B

1.0

1.0

1.0

1

1

1

0.8

0.8

0.8

0.32
1

0.86
1

0.83
1

0.46
0.95
1

0.84
1
1

0.92
0.98
1

0.6

0.6

0.6

0.43
0.96
0.92
1

0.81
0.98
0.98
1

0.9
0.94
0.97
1

0.4

0.4

0.4

0.63
0.84
0.8
0.7
1

0.93
0.97
0.96
0.91
1

0.84
0.9
0.89
0.77
1

0.43
0.94
0.99
0.87
0.84
1

0.84
0.93
0.93
0.98
0.87
1

0.95
0.96
0.99
0.95
0.92
1

0.2

0.2

0.2

0.71
0.75
0.81
0.85
0.59
0.73
1

0.75
0.96
0.97
0.99
0.89
0.96
1

0.81
0.97
0.94
0.85
0.97
0.94
1

Python
C++
Java
PHP
TS
C#
Bash

Python
C++
Java
PHP
TS
C#
Bash

Python
C++
Java
PHP
TS
C#
Bash

0.0

0.0

0.0

Python C++ Java PHP
TS
C#
Bash

Python C++ Java PHP
TS
C#
Bash

Python C++ Java PHP
TS
C#
Bash

Figure 3: Correlations between Languages. Correlation scores between the Python, C++, Java, PHP,
C#, TypeScript (TS), and Bash, reported for different model sizes. The code for this figure was generated by
Code Llama - Instruct, the prompt and code can be seen in Figure 22.

Because of this modest decline in performance and the wide applicability of models with infilling capability,
we decide to release Code Llama 7B, 13B and 70B in this configuration.

Code infilling benchmarks.
Our infilling models reach state-of-the-art performances in code infilling
benchmarks among models of their size. We evaluate on two related code infilling benchmarks based on the
HumanEval benchmark (Chen et al., 2021).

The HumanEval infilling benchmark (Fried et al., 2023) turns the reference solutions of the HumanEval
benchmark (Chen et al., 2021) into infilling problems by masking out either individual lines or blocks consisting
of multiple consecutive lines. It has been extended in Bavarian et al. (2022) with a random span infilling task
in which the masking is applied to a randomly selected substring at the character level. Predictions are scored
with a pass@1 score based on the test cases of the original HumanEval problems. According to the results in
Table 14, our models outperform all other infilling models of their size. Note, however, that the results in
random span infilling are significantly worse in suffix-prefix-middle (SPM) format than in prefix-suffix-middle
(PSM) format as it would require token healing (Microsoft, 2023), which we have not implemented for this
evaluation (see Appendix E for further discussion).

Allal et al. (2023) translates the HumanEval infilling benchmark to other programming languages using
MultiPL-E (Cassano et al., 2023). Single lines are masked and predictions are scored with an exact match
metric against the ground truth solution. Our models, including Code Llama 7B, outperform all open
infilling models across the three programming languages contained in the benchmark (Table 6). We observe a
further increase in performance when prompting the models in SPM format, like witnessed in Bavarian et al.
(2022).

3.3
Long context evaluations

We explore Code Llama’s ability to work with long sequences by measuring perplexity, key retrieval accuracy
and performance during generation on code completion tasks. These tasks, and our results are detailed below.
For full results and comparisons to alternative techniques of increasing the context length of LLMs, we refer
to Appendix G.

Perplexity during extrapolation.
In Figure 4a, perplexity is computed over 4M tokens from the code
dataset, using a subset of our validation data consisting of large source files (≥50kB). For all model sizes,
we observe a steady decrease in perplexity well beyond 16384 tokens, which is the sequence length we use
for long-context fine-tuning. After 100K tokens, the perplexity increases only slightly, in contrast to the
well-known instability phenomenon when testing transformer models on sequences larger than those seen
during training (Press et al., 2022).

10

Model
FIM Size
HumanEval
MBPP
Test loss
pass@1 pass@10 pass@100 pass@1 pass@10 pass@100

Code Llama (w/o LCFT)
✗
7B
33.2%
43.3%
49.9%
44.8%
52.5%
57.1%
0.408
13B
36.8%
49.2%
57.9%
48.2%
57.4%
61.6%
0.372

Code Llama (w/o LCFT)
✓
7B
33.6%
44.0%
48.8%
44.2%
51.4%
55.5%
0.407
13B
36.2%
48.3%
54.6%
48.0%
56.8%
60.8%
0.373

Absolute gap
✗- ✓
7B
−0.4%
−0.7%
1.1%
0.6%
1.1%
1.6%
0.001
13B
0.7%
0.9%
3.3%
0.2%
0.6%
0.8%
−0.001

Table 5: Comparison of models with and without FIM training. pass@1, pass@10 and pass@100
scores on HumanEval and MBPP evaluated at temperature 0.1 for models trained with and without infilling
(FIM) objective. Infilling training incurs no cost on autoregressive test set loss, but a small cost on HumanEval
and MBPP pass@k metrics that is aggravated at higher sample counts k. The models are compared prior to
long context fine-tuning (LCFT).

Model
Size
Python
Java
JavaScript
PSM
SPM
PSM
SPM
PSM
SPM

InCoder
6B
31.0%
49.0%
51.0%
SantaCoder
1.1B
44.0%
62.0%
60.0%
StarCoder
15.5B
62.0%
73.0%
74.0%

Code Llama
7B 67.6%
72.7%
74.3%
77.6%
80.2%
82.6%

13B 68.3% 74.5% 77.6% 80.0% 80.7% 85.0%

Table 6: Multilingual HumanEval single line infilling with MultiPL-E. Exact match rates on the line
infilling benchmark from Allal et al. (2023) with greedy decoding. Evaluated in both prefix-suffix-middle
(PSM) and suffix-prefix-middle (SPM) format. Numbers for InCoder, SantaCoder and StarCoder are reported
from Li et al. (2023).

Key retrieval.
In Figure 4b, we investigate key retrieval performance in synthetic task. The prompt
consists of a large amount of syntactically valid Python code, with a function returning a scalar inserted at a
specified position. The model is asked to complete an assert statement with the return value of the inserted
function. Liu et al. (2023b) showed that the inability to recall content placed in the middle of long prompts
is a common failure mode in LLMs; our retrieval task is analogous to their setup, albeit tailored to code
models which are not fine-tuned to follow instructions. All models exhibit strong retrieval performance on the
sequence length they were trained on, with the exception of the 7B model for test cases in which the function
is placed at the beginning of the prompt. We include OpenAI’s gpt-3.5-turbo-16k-0613 as a reference. We
query GPT with a system prompt of “Complete the following code.” and a temperature of 0. For sequences
beyond 16K tokens, i.e., when extrapolating, our models exhibit a decrease in performance (Appendix G.3).

Single line completion.
Finally, we test the benefits of the ability to handle long context sizes in a single
line code completion task. Our task is based on the Long Code Completion (LCC) benchmark (Guo et al.,
2023).2 The LCC test set is skewed towards shorter files and we hence sample a new set of examples from
LCC’s validation and test set with an equalized distribution over file size (Appendix G.2). In Table 7, we
compare the completion accuracy of the Code Llama models to their counterparts prior to long-context
fine-tuning. Non-LCFT models fail to generate meaningful completions on long sequences and we thus
truncate their prompts to the 4,000 tokens immediate preceding the line to complete. Across all metrics,
models fine-tuned to handle long contexts achieve significantly higher performance. This demonstrates that
long contexts are informative for code completion, and that with LCFT our models are able to leverage this
information to improve their generations. We note that the longest example’s prompt in this test consists

2Note that LCC data points are included in our code training data.

11

Large Source Files

Key Retrieval Accuracy (~16K tokens)

100

1.55

1.50

80

7B
13B
34B

1.45

60

PPL

1.40

Accuracy

40

1.35

20

1.30

7B
13B
34B
gpt-3.5-turbo-16k-0613

0

0
20
40
60
80
100
120
Context Length x10³

0.0
0.2
0.4
0.6
0.8
1.0
Relative Position of Key Function

(a)

(b)

Figure 4: Code Llama behavior on long sequences. (a) Perplexity on large source files (≥50 kB) from
the validation data from the code dataset. The dashed line marks the fine-tuning context length. Perplexity
decreases for up to 100K tokens for all Code Llama sizes. (b) Accuracy on a synthetic key retrieval task,
with a context of 16K tokens and comparison to gpt-3.5-turbo.

Model
EM
BLEU
EM
BLEU
EM
BLEU

Code Llama
7B
✗
36.86
60.16
47.82
69.20
46.29
67.75
Code Llama
7B
✓
39.23
61.84
51.94
71.89
50.20
70.22

Code Llama
13B
✗
37.96
61.33
50.49
69.99
49.22
69.87
Code Llama
13B
✓
41.06
62.76
52.67
72.29
52.15
71.00

Code Llama
34B
✗
42.52
63.74
54.13
72.38
52.34
71.36
Code Llama
34B
✓
44.89
65.99
56.80
73.79
53.71
72.69

Table 7: Average single line completion performance on LCC-balanced. Comparison of models
before and after long-context fine-tuning in terms of exact match (EM) and BLEU. For non-LCFT models,
context size limits are respected by truncating prompts to 4,000 tokens.

of 103K tokens, for which all Code Llama models generate syntactically correct completions, with the 7B
model producing an exact match.

Performance impact on short contexts.
While our models are effective on long sequences, we observe
that LCFT slightly hurts performance on standard code synthesis benchmarks consisting of short sequences.
In Table 10, we observe an average decrease of 0.52 percentage points on HumanEval pass@1 and 1.9 points
on MBPP for the pass@1 metric. Similarly, a breakdown of the code completion results in Table 7 by the
number of tokens in each example shows that for prompts shorter than 4k tokens, long context fine-tuning
induces a reduction of up to 2 BLEU points from base models after code training (Figure 9b). We observe
similar decreases in performance for infilling tasks (Table 14).

LCFT comes at a cost for short sequences, and slightly decreases our scores on standard coding benchmarks
such as HumanEval and MBPP. However, many real-world use cases are not captured by these benchmarks,
and we believe that this cost is more than offset by the potential of handling long sequences for real
downstream applications. Hence we opt to release all our Code Llama, Code Llama - Python and
Code Llama - Instruct models with long-context capabilities.

12

75.0

1.85

2.0

1.8

Scratch model 7B
Code Llama 7B

74.5

13B
34B
7B 13B
34B

Code Llama 7B
Code Llama 13B
Code Llama 34B

1.75

1.9

74.0

1.7

1.8

1.65

73.5

PPL

PPL

1.6

1.7

73.0

1.55

1.5

72.5

1.6

7B

General Helpfulness Abilities

Llama 2 - Chat
Code Llama Instruct

1.45

72.0

0
1
2
3
4
5
# Tokens
1e11

15
20
25
30
35
40
45
50
55
60
Coding Abilities (MBPP zero-shot)

0
1
2
3
4
5
# Tokens
1e11

(a)

(c)

(b)

Figure 5: (a) Training perplexity of Code Llama models. The continued decrease at 500B tokens
suggests further training would be beneficial. Results are presented without infilling for 7B and 13B models.
(b) Training losses of both Code Llama 7B versus an identical model trained from scratch (c) MBPP
(coding benchmark) vs. Helpfulness according to the helpfulness reward model from Llama 2 (Touvron
et al., 2023b).

3.4
Ablation studies

3.4.1
Fine tuning Llama 2 vs. training from scratch on code

Code Llama is based on the Llama 2 models, which are trained on 2T tokens of text, including only 80B
tokens of code. We tune these models on 500B extra tokens, consisting mostly of code (85%). Figure 5a
shows the training curves of Code Llama.

We compare the 7B parameters model to an identical model trained from scratch on the same data mix
(Figure 5b). At the end of training, the loss of the model trained from scratch is equal to the loss of
Code Llama 7B at about half of its training (with 240B less training tokens). Moreover, this gap becomes
larger over time.

3.4.2
Instruction fine-tuning

General helpfulness vs. coding ability
We evaluate Code Llama - Instruct and compare it to
Llama 2-Chat for coding tasks and helpfulness (Figure 5c). We observe that Code Llama improves its
coding abilities for each model sizes, while preserving the general helpfulness performance inherited from
Llama 2. The results on the helpfulness axis is an indication that Code Llama performs greatly on general
instructions following. But we emphasize that this result should be taken with a grain of salt, since we limited
our automatic evaluation to scoring the models answers with Llama 2 reward model.

The value of self-instruct data
We also perform ablations, showing the value of the self-instruct data
that we generate with our own model. To evaluate the capacity of the model to answer questions, we use a
zero-shot version of MBPP. We prompt the model to generate the code between [PYTHON] and [/PYTHON]
tags to make it easy to parse the result. Our exact prompt is shown in Figure 13 in the Appendix. Table 8
show the impact of training on data generated using our models and filtered with unit tests as described in
Section 2.5. The self-instruct data allows us to improve our scores on benchmarks such as HumanEval and
MBPP. It also makes the training more reliable. With self-instruct, the model easily learns to follow the
format requested for MBPP zero-shot while it sometimes fails without it.

Unnatural model.
For comparison purposes, we also finetuned Code Llama - Python 34B on 15,000
unnatural instructions similarly to Honovich et al. (2023) using the same prompts as for the self-instruct
dataset. We do not release this model, but we observe clear improvements on HumanEval and MBPP which
are indicative of the improvements that can be reached with a small set of high-quality coding data. The
results of the unnatural model are shown in Table 2.

13

Size SI HumanEval
MBPP
3-shot zero-shot

7B ✗
30.5% 43.4%
37.6%
✓
34.8% 44.4%
37.4%

13B ✗
40.9% 46.2%
20.4%
✓
42.7% 49.4%
40.2%

Table 8: Impact of self-instruct data. Impact of self-instruct data (SI) on the MBPP and HumanEval
scores of our self-instruct models. The scores are computed using greedy decoding. In MBPP zero-shot,
we prompt the model to generate the solution between [PYTHON][/PYTHON] tags. Removing SI results in
generally lower scores on HumanEval and MBPP, and makes learning to generate code with the right format
for MBPP zero shot much less reliable.

HumanEval Code Llama 7B

HumanEval Code Llama 13B

HumanEval Code Llama 34B

Pass@1
Pass@10
Pass@100

Pass@1
Pass@10
Pass@100

Pass@

Pass@

Pass@

Pass@1
Pass@10
Pass@100

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Temperature

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Temperature

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Temperature

MBPP Code Llama 7B

MBPP Code Llama 13B

MBPP Code Llama 34B

Pass@

Pass@

Pass@

Pass@1
Pass@10
Pass@100

Pass@1
Pass@10
Pass@100

Pass@1
Pass@10
Pass@100

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Temperature

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Temperature

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Temperature

Figure 6: Code Llama scores different temperature values. Results are presented for 7B, 13B, and
34B models on HumanEval and MBPP benchmarks. We report Pass@1, Pass@10, and Pass@100 for different
temperature values. We use nucleus sampling with p=0.95.

3.4.3
Pass@k evaluation

We study the effect of the sampling temperature on the pass@k performance. Specifically, we report pass@1,
10, and 100 using temperature ∈{0.1, 0.4, 0.6, 0.8} on both HumanEval and MBPP. Results are depicted
in Figure 6. As expected, as we increase the temperature, the pass@1 scores are getting worse while the
pass@10 and pass@100 improve.

4
Responsible AI and safety

Large language models have been shown to have the potential to produce known falsehoods due to misconceptions or false beliefs (Lin et al., 2022), generate toxic or offensive content (Hartvigsen et al., 2022) and
reproduce or even amplify the biases that are contained in the training data (Dhamala et al., 2021). As

14

mentioned in Section 2.5, we make Code Llama - Instruct safer by fine-tuning on outputs from Llama 2,
including adversarial prompts with safe responses, as well as prompts addressing code-specific risks.

In this section, we perform evaluations on three widely-used automatic safety benchmarks from the perspectives
of truthfulness, toxicity, and bias, respectively. Specifically, we assess the safety capabilities of both pretrained
Code Llama and fine-tuned Code Llama - Instruct with Falcon (Almazrouei et al., 2023), MPT
(MosaicML, 2023), and StarCoder (Li et al., 2023). Although we have chosen certain standard benchmarks
commonly used in the language model community to highlight some of the problems with these models, it’s
important to note that these evaluations alone do not provide a comprehensive understanding of the risks
associated with them. We complement the safety analysis of Code Llama - Instruct with additional red
teaming from various domain experts in offensive security, malware development, responsible AI and software
engineering, similar to Touvron et al. (2023b).

Truthfulness.
We use TruthfulQA (Lin et al., 2022) to gauge the factuality and common sense of our
models. The TruthfulQA benchmark comprises 817 questions spread across 38 categories, encompassing topics
such as health, finance, law, and politics (Lin et al., 2022). The questions are designed to be challenging, even
for humans, causing them to answer incorrectly due to unfounded beliefs or misconceptions. To evaluate the
generated outputs from LLMs, we utilize GPT-3-based metrics following Lin et al. (2022) to determine the
truthfulness and informativeness of the outputs. For the QA prompt, we use a few-shot prompt containing 6
random QA pairs, structured according to the InstructGPT format (Ouyang et al., 2022). The results are
reported as the percentage of generations that are both truthful and informative, as well as the percentage
that are either truthful or informative.

Toxicity.
We use ToxiGen (Hartvigsen et al., 2022) to quantify the extent of toxic language and hate
speech generation across various demographic groups. The ToxiGen dataset contains implicitly toxic and
benign sentences mentioning 13 minority groups. Following Touvron et al. (2023b), we utilize an improved
version of the dataset, which minimizes noise by removing prompts with disagreements among annotators
regarding the target demographic group. To measure the toxicity of the generated outputs from each of the
LLMs, we employ the default ToxiGen classifier, tuned on RoBERTa (Liu et al., 2019).

Bias.
We employ the Bias in Open-Ended Language Generation Dataset (BOLD) (Dhamala et al., 2021)
to investigate how the sentiment in the model’s outputs may differ based on demographic attributes. The
BOLD benchmark consists of a total of 23,679 English Wikipedia prompts that span five domains: race,
gender, religion, political ideology, and profession. These prompts cover 43 different subgroups. In our
analysis, we exclude prompts belonging to the religious ideology subgroups Hinduism and Atheism due to
their limited representation, consisting of only 12 and 29 prompts, respectively. To assess the sentiments
conveyed by the combination of the prompt prefix and model generation, we employ sentiment analysis using
the Valence Aware Dictionary and Sentiment Reasoner (VADER) (Hutto & Gilbert, 2014). The VADER
produces sentiment scores between -1 and 1, where a positive (negative) score indicates a positive (negative)
sentiment towards the population mentioned in the prompt. A score closer to 0 indicates a neutral sentiment.

Benchmark evaluation results.
Table 9 shows the evaluation results of the three safety benchmarks.
We follow the decoding setting as in Touvron et al. (2023b) where a temperature of 0.1 and top-p of 0.9 are
used. Regarding TruthfulQA, we provide the percentage of generations that are both truthful and informative,
where a higher percentage indicates better performance. Regarding ToxiGen, we present the percentage of
generations deemed toxic by the metric, with a lower percentage indicating better results. Regarding BOLD,
we present the average sentiment scores across demographic groups within the five domains in the BOLD
dataset. The fine-tuned Code Llama - Instruct exhibits significant improvements over the pretrained
Code Llama in terms of truthfulness (from 34.64 to 47.37 for 34B) and toxicity (from 17.62 to 0.00 for 34B).
The percentage of toxic generations drastically reduces to virtually 0% across all Code Llama sizes, making
it the least toxic among all the models compared. When compared to Falcon and MPT fine-tuned models, the
fine-tuned Code Llama demonstrates the second-best performance level in both toxicity and truthfulness,
right after Llama 2 Chat. Additionally, similar to Llama 2 Chat, the Code Llama - Instruct, after
fine-tuning, also tends to show an overall increase in positive sentiment for many demographic groups in
BOLD. More detailed results split by different demographic groups can be found in Appendix I.

15

Safety Reward Model Scores Distribution on Red Teaming Prompts

0.8

0.7

0.6

Responding Model
GPT 3.5 Turbo
Code Llama 13B Instruct
Code Llama 34B Instruct
Code Llama 7B Instruct

0.5

0.4

Density

0.3

0.2

0.1

0.0

0.2
0.0
0.2
0.4
0.6
0.8
1.0
1.2
Llama 2 70B Safety Reward Model Score

Figure 7: KDE plot of the risk score output by the Llama 2 safety reward model on prompts with clear
intent specific to code risk created by red teamers with background in cybersecurity and malware generation.

Red teaming.
It is important to also proactively identify risks with adversarial testing or red teaming.
We conducted 3 red teaming exercises with 25 Meta employees, including domain experts in responsible AI,
malware development, and offensive security engineering.

The red teamers provided a nuanced evaluation specifically on the risk from so called “dual intent prompts.”
Dual intent prompts are requests for help with writing code that could be used maliciously but the prompt
does not directly address the topic (example “Mosaic Prompts” Glukhov et al. (2023)). For example, the model
rightfully refuses to provide support with writing ransomware code but it complies when asked to provide a
script to encrypt all files in the user’s home directory since such a script could be used for benign purposes.

After conducting red team exercises, we asked participants (who had also participated in Llama 2 Chat
exercises) to also provide qualitative assessment of safety capabilities of the model. Some participants who
had expertise in offensive security and malware development questioned the ultimate risk posed by “malicious
code generation” through LLMs with current capabilities.

One red teamer remarked, “While LLMs being able to iteratively improve on produced source code is a risk,
producing source code isn’t the actual gap. That said, LLMs may be risky because they can inform low-skill
adversaries in production of scripts through iteration that perform some malicious behavior.”

According to another red teamer, “[v]arious scripts, program code, and compiled binaries are readily available
on mainstream public websites, hacking forums or on ‘the dark web.’ Advanced malware development is
beyond the current capabilities of available LLMs, and even an advanced LLM paired with an expert malware
developer is not particularly useful- as the barrier is not typically writing the malware code itself. That said,
these LLMs may produce code which will get easily caught if used directly.”

In addition to red teaming sessions, we ran a quantitative evaluation on risk from generating malicious
code by scoring Code Llama’s responses to ChatGPT’s (GPT3.5 Turbo) with LLAMAv2 70B’s safety
reward model. For this second quantitative evaluation, we selected prompts that the red teamers generated
specifically attempting to solicit malicious code (even though the red teaming included consideration of a
broad set of safety risks). These prompts were a mix of clear intent and slightly obfuscated intentions (see
some examples in Figure 16. We show a KDE plot of the distribution of the safety score for all models in
Figure 7). We observe that Code Llama tends to answer with safer responses; the distribution of safety
scores for Code Llama has more weight in the safer part of the range.

False refusals.
LLMs that are too safe can have a tendency to over-refuse valid claims similar to what was
reported after the release of Llama 2. We specifically asked red teamers to test for this behavior. They
found some limited evidence of false refusals (when not using a system preprompt). False refusals could also

16

TruthfulQA ↑
ToxiGen ↓
BOLD

Pretrained models

Falcon 7B
25.95
14.53
0.283
MPT 7B
29.13
22.32
0.322
StarCoder (Python) 15.5B
22.77
10.36
0.310
Llama 2 7B
33.29
21.25
0.304
Llama 2 13B
41.86
26.10
0.330
Llama 2 34B
43.45
21.19
0.318
Code Llama 7B
26.19
22.64
0.230
Code Llama 13B
33.29
22.45
0.176
Code Llama 34B
34.64
17.62
0.255

Instruct (aligned)

Falcon-instruct 7B
28.03
7.89
0.332
MPT-instruct 7B
29.99
16.33
0.302
Llama 2 Chat 7B
57.04
0.00
0.482
Llama 2 Chat 13B
62.18
0.00
0.471
Llama 2 Chat 34B
67.20
0.02
0.461
Code Llama - Instruct 7B
31.46
0.04
0.503
Code Llama - Instruct 13B
36.84
0.01
0.365
Code Llama - Instruct 34B
47.37
0.00
0.452

Table 9: Evaluations on safety datasets for both pretrained (base) models and aligned (instruct) models.
For TruthfulQA, we present the percentage of generations that are both truthful and informative (the higher,
the better). For ToxiGen, we present the percentage of toxic generations (the smaller, the better). For BOLD,
we present the average sentiment scores across demographic groups. A score closer to 0 indicates a neutral
sentiment, while a positive (negative) score indicates a positive (negative) sentiment towards the population
mentioned in the prompt.

be solved by rephrasing the prompt e.g. “Can you tell me how to kill a process?” rephrased to “How do
I kill a process?”. We show some examples in Appendix Table 15. This behavior is something we plan to
investigate in more details in the future.

Safety and coding performance.
As our instruction finetuning set prioritizes safety, longer finetunings
tend to degrade coding performance. We trained our models to reach high coding performances, while not
compromising on safety. As shown in Figure 7, our Code Llama - Instruct models are safer than ChatGPT.

5
Related work

Early observations with LLMs such as GPT-Neo (Black et al., 2021) or GPT-J (Wang & Komatsuzaki, 2021)
showed that adding code in the training data makes program synthesis possible even with medium size LLMs.
Code from open-source software is now a standard part of the training data for general-purpose LLMs such
as PaLM (Chowdhery et al., 2022), Chinchilla (Hoffmann et al., 2022), Gopher (Rae et al., 2021), GPT-4
(OpenAI, 2023), and Llama (Touvron et al., 2023a;b). In parallel, models specifically trained or fine-tuned for
code understanding and program synthesis from natural language prompts emerged with LLMs such as Codex
(Chen et al., 2021), CodeT5 (Wang et al., 2021), InCoder (Fried et al., 2023), AlphaCode (Li et al., 2022),
CodeGen (Nijkamp et al., 2023b) and CodeGen 2 (Nijkamp et al., 2023a), GPT-NeoX (Black et al., 2022),
SantaCoder (Allal et al., 2023), StarCoder (Li et al., 2023) and phi-1 (Gunasekar et al., 2023), consistently
demonstrating better performance on code benchmarks than general-purpose LLMs of comparable or even
larger size. This paper follows this line, by fine-tuning the recent general-purpose language model Llama 2
on code data.

Closed-source vs open-source models.
The landscape of LLMs is marked by whether the technology is
free and the code is available for research or commercial use. ChatGPT and GPT-4 (OpenAI, 2023), PaLM

17

(Chowdhery et al., 2022) and Chinchilla (Hoffmann et al., 2022) are closed source, while BLOOM (Scao et al.,
2022), OPT (Zhang et al., 2022b), and the seminal work of Llama are public (Touvron et al., 2023a). The
more recent Llama 2 has been released under a custom licence for commercial use (Touvron et al., 2023b). A
similar dichotomy exists for code models, with Codex/copilot (Chen et al., 2021), AlphaCode (Li et al., 2022),
GPT-4 or phi-1 (Gunasekar et al., 2023) being closed source, whereas the recent SantaCoder (Allal et al., 2023)
and StarCoder (Li et al., 2023) have been released open-source and allow for commercial use. In this work, we
allow for commercial use of the models under the same terms as Llama 2. Moreover, our largest model, with
its 70B parameters, is significantly larger than previous open-source models – GPT-NeoX-20B (Black et al.,
2022) and StarCoder with 15.5B parameters – which allows it to achieve state-of-the-art performances on
HumanEval, MBPP and MultiPL-E among open-source models. – GPT-NeoX-20B (Black et al., 2022) and
StarCoder with 15.5B parameters – which allows it to achieve state-of-the-art performances on HumanEval,
MBPP and MultiPL-E among open-source models.

Data.
It is well-known that data quality is critical in the training and responsible development of LLMs
(e.g., Hoffmann et al., 2022; Penedo et al., 2023), and this is also true for code as discussed by Allal et al.
(2023). Modern models are trained on publicly available, open-source code. In addition, Allamanis (2019)
and Allal et al. (2023) discuss the impact of effective deduplication and of selecting code from repositories
based on the number of GitHub stars (as a proxy for popularity), while Li et al. (2023) augment their data
with GitHub issues and commits collected from BigQuery. Gunasekar et al. (2023) filter data up to only
containing “textbook”-quality code and add synthetic problems collected using GPT-3.5, following Jung
et al. (2023), in order to obtain good performance on simple benchmarks such as HumanEval and MBPP. We
follow the approach of learning from publicly available code only, without additional meta-level or temporal
information such as issues or commits. We also do not train our foundation models on additional synthetic
exercises, since we did not want to take the risk of reducing the scope of our models to simple coding exercises
similar to those contained in HumanEval and MBPP.

Code understanding and synthesis tasks.
In addition to program synthesis from natural language
prompts or infilling (Fried et al., 2023; Bavarian et al., 2022; Li et al., 2023; Nguyen et al., 2023), many
tasks related to code understanding or synthesis have been addressed since the early 2020s with NLP models
adapted for code (Raffel et al., 2020; Feng et al., 2020; Guo et al., 2021; Wang et al., 2021; Ahmad et al., 2021),
also see the survey by Xu & Zhu (2022). These tasks include code summarization, refinement, translation
(Rozière et al., 2020; 2021; Szafraniec et al., 2023) fixing bugs
(Yasunaga & Liang, 2021; Zhang et al.,
2022a; Prenner et al., 2022), fixing build errors (Tarlow et al., 2020) or generating unit tests (Tufano et al.,
2020; Li et al., 2022; Chen et al., 2023a), as well as solving math problems as demonstrated by PaLM
(Chowdhery et al., 2022) or Codex (Chen et al., 2021). 14 code understanding tasks are represented in the
CodeXGlue benchmark (Lu et al., 2021). Here we focused on the main problem of program synthesis, as
well as infilling/completion for our 7B and 13B models where the ability comes with little impact on the
generation performance as previously observed by Bavarian et al. (2022).

Additional modifications to LLM training and inference.
A number of works proposed to incorporate within the training objective structural knowledge of programs, with specialized objectives for code
deobfuscation (Lachaux et al., 2021), contrastive learning through semantic-preserving code transformations
(Jain et al., 2021), leveraging Abstract Syntax Trees to learn tree-aware positional encodings (Shiv & Quirk,
2019; Peng et al., 2021). A recent stream of work takes into account program execution or unit tests to filter,
cluster, or improve the correctness of programs when few candidates must be submitted (Li et al., 2022; Chen
et al., 2023a; Le et al., 2022; Zhang et al., 2023), or unit tests them within a reinforcement learning objective
to enrich the training signal (Le et al., 2022; Liu et al., 2023a). We focused here on improving the base model
rather than tweaking the inference scheme, since we believe this is where most of the long-term progress
comes from; it is nonetheless an interesting direction to experiment with more elaborated inference schemes
on top of Code Llama.

Long sequences in LLMs.
Scaling Transformers and LLMs to long input sequences has attracted much
recent interest (Dai et al., 2019; Beltagy et al., 2020; Yu et al., 2023; Ding et al., 2023). The context lengths
supported by available models and APIs has seen a steady increase, with StarCoder being trained on 8K

18

token sequences ((Li et al., 2023), up from the 4K of Allal et al. (2023)), recent GPT versions supporting
16K (gpt-3.5-turbo-16k) and 32K tokens (gpt-4-32k), MPT-7b fine-tuned on 65K tokens (MosaicML, 2023),
and Claude featuring 100K context windows (Anthropic, 2023). Previous research focuses on alleviating the
O(n2) space and time complexity of self-attention (Vaswani et al., 2017) by introducing sparsity patterns,
as well as by encoding positional information in such a way that models can leverage input sizes larger
than those presented at training time (length extrapolation). In our work, we do not rely on hand-crafted
sparsity patterns such as those proposed for code input by Guo et al. (2023), who operate on sequences
of up to 4,096 tokens, as to not curtail the model’s expressivity, and modify the encoding of positions
instead. Starting from pretrained Llama 2 models that utilize RoPE (Su et al., 2021), Chen et al. (2023b)
propose additional fine-tuning for long sequence handling, an approach we pursue as well. However, we tailor
our hyper-parameter modifications to allow for extrapolation at inference time. Our modification of the
RoPE hyper-parameters (Su et al., 2021) is a simple modification which does not require any architectural
changes or restrictions and can be readily applied to existing implementations.3 Press et al. (2022) propose
a linear bias for attacking extrapolation; in contrast, our approach seeks to reduce existing bias towards
shot-range attention. Recent work suggests that causal models do not require an explicit encoding of position
information (Haviv et al., 2022; Kazemnejad et al., 2023), a hypothesis we did not test in this work as we
demonstrated that starting from pretrained Llama 2 models is significantly more efficient than training from
scratch.

6
Discussion

We release a family of code-specialized Llama 2 models called Code Llama, with three main variants that
we release with four sizes (7B, 13B, 34B, and 70B parameters): Code Llama, Code Llama - Python,
Code Llama - Instruct. With real-world applications in mind, we trained our 7B, 13B, and 70B models
to support infilling, and all our models to leverage large contexts. We tested their stability in inference up to
100K tokens (Figure 4a). Large context fine-tuning and infilling come at a cost on standard benchmarks
left-to-right code generation benchmarks (Table 10), that are all based on short sequences (i.e. function level).
Still, our 70B model is state-of-the-art among public models on standard python completion benchmarks, and
our other models are competitive compared to models with similar numbers of parameters. On multilingual
benchmarks, even our smallest model (Code Llama 7B) outperforms every other public model.

The Code Llama - Instruct models are trained to provide zero-shot instruction ability to Code Llama.
In this further fine-tuning, where we somewhat distillate Llama 2-Chat, we focused not only on being more
directly helpful (Figure 5c) but also sought to provide a safer model to use and deploy (Section 4). Following
instruction and being overly safe can cost some points on evaluations (e.g. on HumanEval for the 34B model
in Table 2), as exemplified in Figure 15. Further work is needed for LLMs to understand context and nuance
in their instructions.

3Concurrently to our work, the approach of increasing the rotation frequency base value has been proposed by user “bloc97” in
the “LocalLLaMA” subreddit (https://redd.it/14lz7j5), where it was applied to LLaMA models without further fine-tuning.

19

References

Armen Aghajanyan, Bernie Huang, Candace Ross, Vladimir Karpukhin, Hu Xu, Naman Goyal, Dmytro
Okhonko, Mandar Joshi, Gargi Ghosh, Mike Lewis, and Luke Zettlemoyer. CM3: A causal masked
multimodal model of the internet. arXiv:abs/2201.07520, 2022.

Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. Unified pre-training for
program understanding and generation. In NAACL-HLT, pp. 2655–2668. Association for Computational
Linguistics, 2021.

Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Mu~noz Ferrandis,
Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, Logesh Kumar Umapathi, Carolyn Jane
Anderson, Yangtian Zi, Joel Lamy-Poirier, Hailey Schoelkopf, Sergey Troshin, Dmitry Abulkhanov, Manuel
Romero, Michael Lappert, Francesco De Toni, Bernardo García del Río, Qian Liu, Shamik Bose, Urvashi
Bhattacharyya, Terry Yue Zhuo, Ian Yu, Paulo Villegas, Marco Zocca, Sourab Mangrulkar, David Lansky,
Huu Nguyen, Danish Contractor, Luis Villa, Jia Li, Dzmitry Bahdanau, Yacine Jernite, Sean Hughes,
Daniel Fried, Arjun Guha, Harm de Vries, and Leandro von Werra. SantaCoder: Don’t reach for the stars!
arXiv:abs/2301.03988, 2023.

Miltiadis Allamanis. The adverse effects of code duplication in machine learning models of code. In Onward!,
pp. 143–153. ACM, 2019.

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru,
Merouane Debbah, Etienne Goffinet, Daniel Heslow, Julien Launay, Quentin Malartic, Badreddine Noune,
Baptiste Pannier, and Guilherme Penedo. Falcon-40B: An open large language model with state-of-the-art
performance, 2023.

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak
Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey,
Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson,
Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan
Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele
Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément
Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer,
Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann,
Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey
Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy,
Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei
Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni,
Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric
Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan
Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar
Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha
Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John
Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven
Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. PaLM 2 Technical Report.
arXiv:abs/2305.10403, 2023.

Anthropic.
Introducing 100K Context Windows, 2023.
URL https://www.anthropic.com/index/
100k-context-windows.

Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen
Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language
models. arXiv:abs/2108.07732, 2021.

Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry Tworek, and
Mark Chen. Efficient training of language models to fill in the middle. arXiv:abs/2207.14255, 2022.

20

---

*Source: arXiv:2308.12950*
