---
id: B1
title: "Evaluating Large Language Models Trained on Code"
domain: B
year: 2021
arxiv_id: "2107.03374"
confidence: verified
source: "arXiv:2107.03374"
node_type: paper
---

# Evaluating Large Language Models Trained on Code

**Domain**: [[domain_B|LLM Code Generation]] | **Year**: 2021 | **Confidence**: [x] verified


## Authors
[[author_Mark Chen|Mark Chen]], [[author_Jerry Tworek|Jerry Tworek]], [[author_Heewoo Jun|Heewoo Jun]], [[author_Qiming Yuan|Qiming Yuan]], [[author_Henrique Ponde de Oliveira Pinto|Henrique Ponde de Oliveira Pinto]], [[author_Jared Kaplan|Jared Kaplan]], [[author_Harri Edwards|Harri Edwards]], [[author_Yuri Burda|Yuri Burda]], [[author_Nicholas Joseph|Nicholas Joseph]], [[author_Greg Brockman|Greg Brockman]], [[author_Alex Ray|Alex Ray]], [[author_Raul Puri|Raul Puri]], [[author_Gretchen Krueger|Gretchen Krueger]], [[author_Michael Petrov|Michael Petrov]], [[author_Heidy Khlaaf|Heidy Khlaaf]], [[author_Girish Sastry|Girish Sastry]], [[author_Pamela Mishkin|Pamela Mishkin]], [[author_Brooke Chan|Brooke Chan]], [[author_Scott Gray|Scott Gray]], [[author_Nick Ryder|Nick Ryder]], [[author_Mikhail Pavlov|Mikhail Pavlov]], [[author_Alethea Power|Alethea Power]], [[author_Lukasz Kaiser|Lukasz Kaiser]], [[author_Mohammad Bavarian|Mohammad Bavarian]], [[author_Clemens Winter|Clemens Winter]], [[author_Philippe Tillet|Philippe Tillet]], [[author_Felipe Petroski Such|Felipe Petroski Such]], [[author_Dave Cummings|Dave Cummings]], [[author_Matthias Plappert|Matthias Plappert]], [[author_Fotios Chantzis|Fotios Chantzis]], [[author_Elizabeth Barnes|Elizabeth Barnes]], [[author_Ariel Herbert-Voss|Ariel Herbert-Voss]], [[author_William Hebgen Guss|William Hebgen Guss]], [[author_Alex Nichol|Alex Nichol]], [[author_Alex Paino|Alex Paino]], [[author_Nikolas Tezak|Nikolas Tezak]], [[author_Jie Tang|Jie Tang]], [[author_Igor Babuschkin|Igor Babuschkin]], [[author_Suchir Balaji|Suchir Balaji]], [[author_Shantanu Jain|Shantanu Jain]], [[author_William Saunders|William Saunders]], [[author_Christopher Hesse|Christopher Hesse]], [[author_Andrew N. Carr|Andrew N. Carr]], [[author_Jan Leike|Jan Leike]], [[author_Josh Achiam|Josh Achiam]], [[author_Vedant Misra|Vedant Misra]], [[author_Evan Morikawa|Evan Morikawa]], [[author_Alec Radford|Alec Radford]], [[author_Matthew Knight|Matthew Knight]], [[author_Miles Brundage|Miles Brundage]], [[author_Mira Murati|Mira Murati]], [[author_Katie Mayer|Katie Mayer]], [[author_Peter Welinder|Peter Welinder]], [[author_Bob McGrew|Bob McGrew]], [[author_Dario Amodei|Dario Amodei]], [[author_Sam McCandlish|Sam McCandlish]], [[author_Ilya Sutskever|Ilya Sutskever]], [[author_Wojciech Zaremba|Wojciech Zaremba]]


## Keywords
- [[kw_HumanEval|HumanEval]]
- [[kw_code evaluation|code evaluation]]
- [[kw_benchmark|benchmark]]
- [[kw_Codex|Codex]]
- [[kw_pass@k|pass@k]]
- [[kw_LG|LG]]

## Abstract

We introduce Codex, a GPT language model fine-tuned on publicly available code from GitHub, and study its Python code-writing capabilities. A distinct production version of Codex powers GitHub Copilot. On HumanEval, a new evaluation set we release to measure functional correctness for synthesizing programs from docstrings, our model solves 28.8% of the problems, while GPT-3 solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling from the model is a surprisingly effective strategy for producing working solutions to difficult prompts. Using this method, we solve 70.2% of our problems with 100 samples per problem. Careful investigation of our model reveals its limitations, including difficulty with docstrings describing long chains of operations and with binding operations to variables. Finally, we discuss the potential broader impacts of deploying powerful code generation technologies, covering safety, security, and economics.

## Paper Content

Evaluating Large Language Models Trained on Code

Mark Chen * 1 Jerry Tworek * 1 Heewoo Jun * 1 Qiming Yuan * 1 Henrique Ponde de Oliveira Pinto * 1

Jared Kaplan * 2 Harri Edwards 1 Yuri Burda 1 Nicholas Joseph 2 Greg Brockman 1 Alex Ray 1 Raul Puri 1

Gretchen Krueger 1 Michael Petrov 1 Heidy Khlaaf 3 Girish Sastry 1 Pamela Mishkin 1 Brooke Chan 1

Scott Gray 1 Nick Ryder 1 Mikhail Pavlov 1 Alethea Power 1 Lukasz Kaiser 1 Mohammad Bavarian 1

Clemens Winter 1 Philippe Tillet 1 Felipe Petroski Such 1 Dave Cummings 1 Matthias Plappert 1

Fotios Chantzis 1 Elizabeth Barnes 1 Ariel Herbert-Voss 1 William Hebgen Guss 1 Alex Nichol 1 Alex Paino 1

Nikolas Tezak 1 Jie Tang 1 Igor Babuschkin 1 Suchir Balaji 1 Shantanu Jain 1 William Saunders 1

Christopher Hesse 1 Andrew N. Carr 1 Jan Leike 1 Josh Achiam 1 Vedant Misra 1 Evan Morikawa 1

Alec Radford 1 Matthew Knight 1 Miles Brundage 1 Mira Murati 1 Katie Mayer 1 Peter Welinder 1

Bob McGrew 1 Dario Amodei 2 Sam McCandlish 2 Ilya Sutskever 1 Wojciech Zaremba 1

Abstract

1. Introduction

arXiv:2107.03374v2  [cs.LG]  14 Jul 2021

We introduce Codex, a GPT language model ﬁnetuned on publicly available code from GitHub,
and study its Python code-writing capabilities.
A distinct production version of Codex powers
GitHub Copilot. On HumanEval, a new evaluation set we release to measure functional correctness for synthesizing programs from docstrings,
our model solves 28.8% of the problems, while
GPT-3 solves 0% and GPT-J solves 11.4%. Furthermore, we ﬁnd that repeated sampling from the
model is a surprisingly effective strategy for producing working solutions to difﬁcult prompts. Using this method, we solve 70.2% of our problems
with 100 samples per problem. Careful investigation of our model reveals its limitations, including
difﬁculty with docstrings describing long chains
of operations and with binding operations to variables. Finally, we discuss the potential broader
impacts of deploying powerful code generation
technologies, covering safety, security, and economics.

Scalable sequence prediction models (Graves, 2014;
Vaswani et al., 2017; Child et al., 2019) have become a
general-purpose method for generation and representation
learning in many domains, including natural language processing (Mikolov et al., 2013; Sutskever et al., 2014; Dai &
Le, 2015; Peters et al., 2018; Radford et al., 2018; Devlin
et al., 2018), computer vision (Van Oord et al., 2016; Menick
& Kalchbrenner, 2018; Chen et al., 2020; Bao et al., 2021),
audio and speech processing (Oord et al., 2016; 2018; Dhariwal et al., 2020; Baevski et al., 2020), biology (Alley et al.,
2019; Rives et al., 2021), and even across multiple modalities (Das et al., 2017; Lu et al., 2019; Ramesh et al., 2021;
Zellers et al., 2021). More recently, language models have
also fueled progress towards the longstanding challenge
of program synthesis (Simon, 1963; Manna & Waldinger,
1971), spurred by the presence of code in large datasets
(Husain et al., 2019; Gao et al., 2020) and the resulting programming capabilities of language models trained on these
datasets (Wang & Komatsuzaki, 2021). Popular language
modeling objectives like masked language modeling (Devlin
et al., 2018) and span prediction (Raffel et al., 2020) have
also been adapted to train their programming counterparts
CodeBERT (Feng et al., 2020) and PyMT5 (Clement et al.,
2020).

*Equal contribution
1OpenAI, San Francisco, California, USA.
2Anthropic AI, San Francisco, California, USA. Work performed while at OpenAI.
3Zipline, South San Francisco, California, USA. Work performed while at OpenAI.
Correspondence to:
Mark Chen <mark@openai.com>,
Jerry
Tworek
<jt@openai.com>,
Heewoo
Jun
<heewoo@openai.com>, Qiming Yuan <qiming@openai.com>.

Similarly, our early investigation of GPT-3 (Brown et al.,
2020) revealed that it could generate simple programs from
Python docstrings. While rudimentary, this capability was
exciting because GPT-3 was not explicitly trained for code
generation. Given the considerable success of large language models in other modalities and the abundance of
publicly available code, we hypothesized that a specialized
GPT model, called Codex, could excel at a variety of coding
tasks. This paper describes several early Codex models,
whose descendants power GitHub Copilot and the Codex
models in the OpenAI API.

Evaluating Large Language Models Trained on Code

generate at least one correct function for 77.5% of the problems. This result suggests that accurate code samples can
be selected via heuristic ranking instead of fully evaluating
each sample, the latter of which may not be possible or practical in deployment. Indeed, we ﬁnd that the sample with
highest mean log-probability passes unit tests for 44.5% of
the problems.

We conclude by discussing the limitations and potential
broader impacts of these Codex models and of increasingly
powerful code generating models more generally.

2. Evaluation Framework

In this section, we discuss the details of our evaluation
framework. We begin by deﬁning the pass@k metric, and
explain its advantages over standard match-based metrics.
Next, we describe the dataset of hand-written problems,
called “HumanEval,” which we created in order to benchmark our models. Finally, we discuss the sandbox environment we used to safely execute model-generated code.

2.1. Functional Correctness

Figure 1. Pass rates of our models on the HumanEval dataset as a
function of model size. When a single sample is generated for each
problem, GPT-12B solves no problems, but Codex (ﬁne-tuned
on code) solves 28.8% of the problems, and Codex-S (further
ﬁne-tuned on correctly implemented standalone functions) solves
37.7% of the problems. From here, further gains can be realized by
generating 100 samples per problem and selecting the sample with
the highest mean log-probability (44.5% solved) or by selecting
the sample that passes the unit tests (77.5% solved). All samples
are generated with temperature 0.8.

Generative models for code are predominantly benchmarked
by matching samples against a reference solution, where
the match can be exact or fuzzy (as in BLEU score). However, recent work has surfaced deﬁciencies in match-based
metrics for code. For instance, Ren et al. (2020) ﬁnds that
BLEU has problems capturing semantic features speciﬁc
to code, and suggests several semantic modiﬁcations to the
score.

In this work, we focus on the task of generating standalone Python functions from docstrings, and evaluate the
correctness of code samples automatically through unit
tests. This is in contrast to natural language generation,
where samples are typically evaluated by heuristics or by
human evaluators. To accurately benchmark our model,
we create a dataset of 164 original programming problems
with unit tests. These problems assess language comprehension, algorithms, and simple mathematics, with some
comparable to simple software interview questions. We
release this data along with an evaluation framework at
https://www.github.com/openai/human-eval.

More fundamentally, match-based metrics are unable to account for the large and complex space of programs functionally equivalent to a reference solution. As a consequence,
recent works in unsupervised code translation (Lachaux
et al., 2020) and pseudocode-to-code translation (Kulal et al.,
2019) have turned to functional correctness instead, where
a sample is considered correct if it passes a set of unit tests.
We argue that this metric should be applied to docstringconditional code generation as well.

To solve a problem in our test set, we generate multiple
samples from the models, and check if any of them pass the
unit tests. With just a single sample, a 12B parameter Codex
solves 28.8% of these problems, and a 300M parameter
Codex solves 13.2% of these problems. In contrast, the 6B
parameter GPT-J (Wang & Komatsuzaki, 2021) achieves
11.4% on the same dataset, while all GPT models achieve
near 0%. To improve our model’s performance at the task of
function synthesis from docstrings, we ﬁne-tune Codex on
standalone, correctly implemented functions. The resulting
model, Codex-S, solves 37.7% of problems with a single
sample. Figure 2 showcases problems of varying difﬁculty
in our dataset, along with correct model generated solutions.

Perhaps the most convincing reason to evaluate functional
correctness is that it is used by human developers to judge
code. A framework known as test-driven development dictates that software requirements be converted into test cases
before any implementation begins, and success is deﬁned
by a program that passes these tests. While few organizations employ full test-driven development, integration of
new code is usually dependent on creating and passing unit
tests.

Kulal et al. (2019) evaluate functional correctness using
the pass@k metric, where k code samples are generated
per problem, a problem is considered solved if any sample

Real-world programming tasks often involve iterations of
approaches and bug ﬁxes, which is approximated by generating many samples from our models and selecting one that
passes all unit tests. Within 100 samples, Codex-S is able to

Evaluating Large Language Models Trained on Code

Figure 2. Three example problems from the HumanEval dataset, where the probabilities that a single sample from Codex-12B passes unit
tests are 0.9, 0.17, and 0.005. The prompt provided to the model is shown with a white background, and a successful model-generated
completion is shown in a yellow background. Though not a guarantee for problem novelty, all problems were hand-written and not
programmatically copied from existing sources. Random problems and samples can be found in Appendix B.

passes the unit tests, and the total fraction of problems
solved is reported. However, computing pass@k in this
way can have high variance. Instead, to evaluate pass@k,
we generate n ≥k samples per task (in this paper, we
use n = 200 and k ≤100), count the number of correct
samples c ≤n which pass unit tests, and calculate the
unbiased estimator

"

#

def pass_at_k(n, c, k):
"""
:param n: total number of samples
:param c: number of correct samples
:param k: k in pass@$k$
"""
if n - c < k: return 1.0
return 1.0 - np.prod(1.0 - k /
np.arange(n - c + 1, n + 1))
Figure 3. A numerically stable script for calculating an unbiased
estimate of pass@k.

1 −

(1)

pass@k :=
E
Problems

 n−c
k

 n
k


Calculating this estimator directly results in very large numbers and numerical instability. In Figure 3, we include a
numerically stable numpy implementation that simpliﬁes
the expression and evaluates the product term-by-term. One
may be tempted to estimate pass@k with 1−(1−ˆp)k where
ˆp is the empirical estimate of pass@1, but we show that it is
biased in Appendix A.

Later, we provide evidence that BLEU score may not be
a reliable indicator of functional correctness by showing
that functionally inequivalent programs generated by our
model (which are guaranteed to disagree with the reference
solution on some input) often have higher BLEU scores than
functionally equivalent ones.

Evaluating Large Language Models Trained on Code

2.2. HumanEval: Hand-Written Evaluation Set

problem, and pick one that passes unit tests. When limited to
a budget of one evaluation per problem, producing multiple
samples with Codex and choosing the one with the highest
mean log-probability provides signiﬁcant gains.

3.1. Data Collection

We evaluate functional correctness on a set of 164 handwritten programming problems, which we call the HumanEval dataset. Each problem includes a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem. It is important for these
tasks to be hand-written, since our models are trained on a
large fraction of GitHub, which already contains solutions
to problems from a variety of sources. For example, there
are more than ten public repositories containing solutions to
Codeforces problems, which make up part of the recently
proposed APPS dataset (Hendrycks et al., 2021).

Our training dataset was collected in May 2020 from 54 million public software repositories hosted on GitHub, containing 179 GB of unique Python ﬁles under 1 MB. We ﬁltered
out ﬁles which were likely auto-generated, had average line
length greater than 100, had maximum line length greater
than 1000, or contained a small percentage of alphanumeric
characters. After ﬁltering, our ﬁnal dataset totaled 159 GB.

3.2. Methods

Programming tasks in the HumanEval dataset assess language comprehension, reasoning, algorithms, and simple
mathematics. We release the HumanEval dataset so that
others can evaluate functional correctness and measure the
problem-solving capabilities of their models. The dataset
can be found at https://www.github.com/openai/human-eval.

2.3. Sandbox for Executing Generated Programs

Since Codex is evaluated on natural language prompts, we
hypothesized that it would be beneﬁcial to ﬁne-tune from
the GPT-3 (Brown et al., 2020) model family, which already
contains strong natural language representations. Surprisingly, we did not observe improvements when starting from
a pre-trained language model, possibly because the ﬁnetuning dataset is so large. Nevertheless, models ﬁne-tuned
from GPT converge more quickly, so we apply this strategy
for all subsequent experiments.

Since publicly available programs have unknown intent and
generated programs are often incorrect, executing these
programs poses a security risk. Indeed, GitHub is known
to contain malicious programs that alter or change their
environments (Rokon et al., 2020).

We train Codex using the same learning rate as the corresponding GPT model, with a 175 step linear warmup and
cosine learning rate decay. We train for a total of 100 billion
tokens, using the Adam optimizer with β1 = 0.9, β2 = 0.95,
ϵ = 10−8, and a weight decay coefﬁcient of 0.1.

Therefore, we developed a sandbox environment to safely
run untrusted programs against unit tests. Our goals were to
prevent these programs from modifying, gaining persistence
on, accessing sensitive resources on, or exﬁltrating data from
a host or network. Since OpenAI’s training infrastructure
is built on Kubernetes and cloud services, we designed our
sandbox to address the limitations of these environments
while remaining idiomatic with their patterns of use.

In order to maximally leverage text representations from
GPT, we base our code lexer on the GPT-3 text tokenizer.
Since the distribution of words in GitHub code differs from
that of natural text, this tokenizer is not very effective for
representing code. The largest source of inefﬁciency arises
from encoding whitespace, so we add an additional set of
tokens for representing whitespace runs of different lengths.
This allows us to represent code using approximately 30%
fewer tokens.

We selected the gVisor container runtime (Lacasse, 2018)
as the main host protection component. Since container
runtimes like Docker can share host resources with containers, a malicious container could potentially compromise a
host. gVisor protects the host by emulating its resources to
introduce a security boundary between the host and its containers. Network-adjacent hosts and services are protected
by eBPF-based ﬁrewall rules that prevent inbound and outbound connections except for those required for experiment
control.

3. Code Fine-Tuning

To compute pass@k, we assemble each HumanEval problem into a prompt consisting of a header, a signature, and
a docstring, which is illustrated in Figure 2. We sample
tokens from Codex until we encounter one of the following
stop sequences: ‘\nclass’, ‘\ndef’, ‘\n#’, ‘\nif’, or
‘\nprint’, since the model will continue generating additional functions or statements otherwise. We use nucleus
sampling (Holtzman et al., 2020) with top p = 0.95 for all
sampling evaluation in this work.

3.3. Results

In Figure 4, we plot test loss on a held-out validation set
against Codex model size. We ﬁnd that just as language

We ﬁne-tune GPT models containing up to 12B parameters
on code to produce Codex. In contrast with GPT, Codex
displays non-trivial performance on the HumanEval dataset.
In fact, Codex is able to solve the majority of the problems
in HumanEval if we generate and evaluate 100 samples per

Evaluating Large Language Models Trained on Code

Figure 4. Model cross-entropy test loss measured on a held-out
split of our Python GitHub code corpus. The smooth power law
scaling of performance with model size observed in GPT-3 appears
to hold even after code ﬁne-tuning.

model test loss follows a power law in model size (Kaplan
et al., 2020), test loss after code ﬁne-tuning follows a similar
power law with functional form (
N
5.92×107 )−0.13 where N
is the number of non-embedding parameters in the model.

When evaluating pass@k, it is important to optimize sampling temperature for the particular value of k. In Figure 5,
we plot pass@k against the number of samples k and the
sampling temperature. We ﬁnd that higher temperatures are
optimal for larger k, because the resulting set of samples
has higher diversity, and the metric rewards only whether
the model generates any correct solution.

Figure 5. In the top panel, we plot pass@k against the number of
samples (k) for various temperature settings. Higher temperatures
are better when the number of samples is large, likely due to the
increased sample diversity. In the bottom panel, we plot the best
temperature setting for each k, obtained by taking the upper hull
of the top panel.

In particular, for a 679M parameter model, the optimal temperature for pass@1 is T ∗= 0.2 and the optimal temperature for pass@100 is T ∗= 0.8. With these temperatures,
we ﬁnd that pass@1 and pass@100 scale smoothly as a
function of model size (Figure 6).

Pass@k can also be interpreted as the result of evaluating
the best out of k samples, where the best sample is picked
by an oracle with prior knowledge of the unit tests. From
a practical perspective, we are also interested in the setting where we must select a single sample from k samples
without having access to an oracle. For instance, when the
model is used as an autocomplete tool where a user provides
a prompt, we do not have unit tests, but would like to return
only a single completion to the user for evaluation so as to
not overwhelm them.

Figure 6. Using the optimal temperatures 0.2 and 0.8 for pass@1
and pass@100, we plot these two metrics as a function of model
size. Performance appears to scale smoothly as a sigmoid in logparameters.

Inspired by similar work in language modeling, we ﬁnd
that choosing the sample with the highest mean token log
probability outperforms evaluating a random sample, while
choosing the sample based on sum log probability can perform slightly worse than picking randomly. Figure 7 demonstrates the beneﬁts of applying these heuristics to samples
(at temperature 0.8) from Codex-12B.

Evaluating Large Language Models Trained on Code

Figure 7. Model performance in the setting where we can generate
multiple samples, but only evaluate one. We can do better than randomly selecting a sample by choosing the solution with the highest
mean log-probability (red) or with the highest back-translation
score (orange) described in Sec. 5. The blue line represents the
theoretical best performance obtained using an oracle with prior
knowledge of the unit tests.

Figure 8. BLEU score probability densities for correct (blue) and
wrong (green) solutions from Codex-12B for 4 random tasks from
HumanEval. Note that the distributions are not cleanly separable,
suggesting that optimizing for BLEU score is not equivalent to
optimizing for functional correctness.

uating at temperatures 0.2, 0.4, and 0.8 for GPT-Neo, and
from temperatures 0.2 and 0.8 for GPT-J. Detailed results
across multiple model sizes can be found in Table 1.

Finally, we compute BLEU scores for all Codex-12B HumanEval samples (at temperature 0.8) against their reference
solutions. For each problem, when we plot the distributions
of BLEU scores for correct and incorrect solutions, we
notice signiﬁcant overlap (Figure 8). Since an incorrect
solution is guaranteed to be functionally inequivalent to
the reference solution, we conclude that improvements in
BLEU score may not indicate improved rates of functional
correctness in practice.

Finally, we benchmark Codex against the largest free model
from Tabnine, a leading code autocomplete system, which
achieves 2.6% pass@1 (at T = 0.4) and 7.6% pass@100
(at T = 0.8). This is roughly equivalent to Codex-12M, one
of the smallest models in our suite.

3.5. Results on the APPS Dataset

3.4. Comparative Analysis of Related Models and
Systems

Two recent works similar in spirit to Codex are GPT-Neo
(Black et al., 2021) and GPT-J (Wang & Komatsuzaki,
2021), which are trained on The Pile (Gao et al., 2020),
a dataset containing text from a variety of sources as well
as 8% GitHub code. The broader research community has
found that these models outperform existing GPT systems
in qualitative programming evaluations (Woolf, 2021).

Recently, Hendrycks et al. (2021) introduced the APPS
dataset to measure the coding challenge competence of language models. The APPS dataset consists of 5000 training
and 5000 test examples of coding problems, each with a set
of unit tests and, for the training data, a set of correct solutions. Most of the APPS tests problems are not formulated
as single-function synthesis tasks, but rather as full-program
synthesis, reading input from stdin and printing output to
stdout, in contrast to the main Codex training data.

In the paper that introduces APPS, the authors benchmark a
few language models and report two metrics: the percentage
of problems where the model ﬁnds a correct solution (called
the “strict accuracy”) and the percentage of unit tests passed,
even if the solution is incorrect. The latter measure is reported only so as to reduce variance of the measurements,
because the results on the ﬁrst metric were so low. We avoid
this metric and only focus on “strict accuracy”, and - as in

We conﬁrm these ﬁndings using the HumanEval dataset,
showing that GPT-Neo achieves 6.4% pass@1 and 21.3%
pass@100, while GPT models of comparable sizes achieve
near 0% on both metrics. We see a remarkable progression
in capabilities, with GPT-Neo-2.7B roughly equivalent to
Codex-85M (30× fewer parameters). Similarly, GPT-J-6B
achieves 11.6% pass@1 and 27.7% pass@100, which is
roughly equivalent to Codex-300M (20× fewer parameters).
Pass rates are obtained by taking the best result from eval
Evaluating Large Language Models Trained on Code

4. Supervised Fine-Tuning

Table 1. Codex, GPT-Neo, & TabNine evaluations for HumanEval.
We ﬁnd that GPT-J pass@1 is between Codex-85M and Codex300M performance.

PASS@k
k = 1
k = 10
k = 100

In addition to standalone functions, Python code found on
GitHub contains class implementations, conﬁguration ﬁles,
scripts, and even ﬁles used to store data. This code is seemingly unrelated to synthesizing functions from docstrings,
and we hypothesize that the distribution mismatch reduces
HumanEval performance.

GPT-NEO 125M
0.75%
1.88%
2.97%
GPT-NEO 1.3B
4.79%
7.47%
16.30%
GPT-NEO 2.7B
6.41%
11.27%
21.37%
GPT-J 6B
11.62%
15.74%
27.74%

TABNINE
2.58%
4.35%
7.59%

In order to adapt Codex to the distribution of the task of interest, we construct a set of training problems from correctly
implemented standalone functions, and use them for additional supervised ﬁne-tuning. We describe two approaches
for collecting these examples: from competitive programming websites and from repositories with continuous integration. We call the supervised ﬁne-tuned models Codex-S,
and show that they produce consistent gains across model
size.

CODEX-12M
2.00%
3.62%
8.58%
CODEX-25M
3.21%
7.1%
12.89%
CODEX-42M
5.06%
8.8%
15.55%
CODEX-85M
8.22%
12.81%
22.4%
CODEX-300M
13.17%
20.37%
36.27%
CODEX-679M
16.22%
25.7%
40.95%
CODEX-2.5B
21.36%
35.42%
59.5%
CODEX-12B
28.81%
46.81%
72.31%

4.1. Problems from Competitive Programming

the previous sections - we report pass@k numbers for various k (Table 2). There are 2 additional factors, well-known
from coding competitions, that we take into account:

Programming contest and interview preparation websites
use hidden unit tests to automatically judge the functional correctness of submissions. These problems are selfcontained, come with well-written problem statements, and
generally have excellent test coverage. Additionally, these
problems test algorithmic reasoning over a broad range of
core skills and difﬁculties.

• In coding competitions and in the APPS datasets, tasks
are provided with 3 input/output examples included in
the task description. We utilize this by sampling 1000
solutions from the model and ﬁltering out only those
that pass these 3 unit tests (if such solutions exist). We
then calculate pass rates in this ﬁltered set, and call it
ﬁltered pass@k. Results without ﬁltering are presented
as raw pass@k.

We collected problem statements, function signatures, and
solutions from several popular programming contest and
interview preparation websites. We then assembled these
into programming tasks similar to HumanEval, using the
problem description as the docstring. Since complete test
suites are often hidden, we created unit tests from examples
found in the problem statements, or extracted additional test
cases through submitting incorrect solutions. In total, we
curated 10,000 problems in this way.

4.2. Problems from Continuous Integration

• It is often the case both in coding competitions and in
the results from Codex that a correct solution is found,
but it is not algorithmically efﬁcient enough to be considered passing. While this is not acceptable in the
competitions, we also report the number of solutions
that Codex produces that do not fail on any unit test,
but that do time-out on some of them. We use a timeout
of 3 seconds in our evaluation.

Next, we curated programming problems from open source
projects. Taking advantage of sys.setprofile, we
were able to trace and collect inputs and outputs for all
functions called during integration tests. This data could
then be used to create unit tests for the functions.

Projects that employ continuous integration (CI) are ideal
candidates for tracing. We follow the commands in the CI
conﬁguration ﬁles, which contain build and test commands,
to set up the virtual environments, install dependencies, and
run integration tests.

We considered GitHub repos using travis and tox as their CI
frameworks, as they are two of the most popular CI tools.
We additionally used publicly available source code from
pip packages found in the python package index (PyPI).

To compensate for the fact the Codex is not ﬁne-tuned on
APPS, we append a single input/output example from the
task description to the docstring as a formatting hint. We denote this setting as “1-shot” in Table 2, and ﬁnd that Codex12B evaluated 1-shot achieves comparable performance to a
GPT-Neo model ﬁne-tuned on APPS. Consistent with our
earlier ﬁndings, there are large beneﬁts from generating and
evaluating as many as 1000 samples per task, though for
more difﬁcult problems, solutions are often not efﬁcient
enough to pass the time limits. Finally, evaluating the ﬁrst
sample which passes the 3 public unit tests for each problem
yields higher performance than raw pass@100 samples.

Evaluating Large Language Models Trained on Code

Table 2. Finetuned GPT-Neo numbers from the APPS paper referenced above. For Codex-12B, the number of passing programs that
timeout on some test is in the bracket. We used temperature 0.6 for sampling to cover all k in pass@k, so raw pass@1 results could be
improved with lower temperature.

INTRODUCTORY
INTERVIEW
COMPETITION

GPT-NEO 2.7B RAW PASS@1
3.90%
0.57%
0.00%
GPT-NEO 2.7B RAW PASS@5
5.50%
0.80%
0.00%

1-SHOT CODEX RAW PASS@1
4.14% (4.33%)
0.14% (0.30%)
0.02% (0.03%)
1-SHOT CODEX RAW PASS@5
9.65% (10.05%)
0.51% (1.02%)
0.09% (0.16%)
1-SHOT CODEX RAW PASS@100
20.20% (21.57%)
2.04% (3.99%)
1.05% (1.73%)
1-SHOT CODEX RAW PASS@1000
25.02% (27.77%)
3.70% (7.94%)
3.23% (5.85%)

1-SHOT CODEX FILTERED PASS@1
22.78% (25.10%)
2.64% (5.78%)
3.04% (5.25%)
1-SHOT CODEX FILTERED PASS@5
24.52% (27.15%)
3.23% (7.13%)
3.08% (5.53%)

4.4. Methods

Because these projects contained untrusted code, it was important to run integration tests in the sandboxed environment
described above.

We ﬁne-tune Codex on these training problems to produce a
set of “supervised ﬁne-tuned” models, which we call CodexS. To produce examples from training problems, we assemble the problems into the format shown in Figure 2. If there
are prompts of varying length in a batch, we left-pad shorter
prompts to the length of the longest prompt, so that the ﬁrst
tokens in the reference solutions line up in context.

While there are millions of potential functions to curate
problems from, we only collected about 40,000 because
not all functions accept inputs and return outputs. Even
when they do, most objects captured at runtime cannot be
pickled and restored outside the sandbox unless the project
was installed.

We train to minimize negative log-likelihood of the reference
solution, and mask out loss for any tokens in the prompt.
We train using a learning rate 1/10 as large as used for
ﬁne-tuning Codex, but adhere to the same learning rate
schedule, and train until validation loss plateaus (less than
10B tokens).

4.5. Results

Since our tracing methodology produced inputs and outputs
for all invoked functions, even builtin and library calls imported by the project were turned into problems. For this
reason, functions from tracing tended to be the building
blocks of command-line utilities. To excel at these tasks,
the model does not need to know advanced algorithms and
data structures. Rather, it needs to be able to follow instructions to implement the functionality speciﬁed in the
docstring. Thus, tracing complements the puzzle nature of
coding competition problems and broadens the distribution
of tasks.

4.3. Filtering Problems

As with Codex, we ﬁrst compute the optimal temperature for
evaluating pass@k for 1 ≤k ≤100. We ﬁnd that Codex-S
prefers slightly higher temperatures for all k > 1, which
possibly reﬂects the fact that Codex-S captures a narrower
distribution than Codex. We use T ∗= 0 for computing
pass@1 and T ∗= 1 for computing pass@100.

Next, we compare Codex-S against Codex on pass@1 and
pass@100. Codex-S outperforms the corresponding Codex
by an average margin of 6.5 percentage points on pass@1
and by a larger average margin of 15.1 percentage points on
pass@100 across model size.

In the previous sections, we presented two methods we
used to automatically create training problems. However,
it is unclear how to control for quality. Some prompts
underspecify the function that is implemented, in which
case a perfectly valid solution may be wrongly penalized by
the unit test. Some problems are stateful, and subsequent
executions can result in different outcomes.

We also plot the performance of different sample selection
heuristics for Codex-S-12B against the same heuristics for
Codex-12B. When ranking between 1 and 100 samples
by mean log probability, the average beneﬁt over random
ranking is 11.6 percentage points, which is over 2 percentage
points higher than the corresponding beneﬁt for Codex.

To address these issues, we use Codex-12B to generate 100
samples per curated problem. If no samples pass the unit
tests, we consider the task to be either ambiguous or too
difﬁcult, and ﬁlter it out. We reran this veriﬁcation several
times to remove stateful or non-deterministic problems.

Evaluating Large Language Models Trained on Code

5. Docstring Generation

Generating code from docstrings is possible with Codex
because code typically follows after a docstring, but it is not
easy to induce Codex to generate docstrings from code. Nevertheless, we are motivated to produce a docstring writing
model for safety reasons, as such a model can be used to describe the intent behind generated code. Using the training
problems described in the previous section, we can easily create a training dataset for code-conditional docstring
generation.

Speciﬁcally, for each training problem, we assemble a training example by concatenating the function signature, the
reference solution, and then the docstring. Just as we train
Codex-S by minimizing negative log-likelihood of the reference solution, we train the docstring generating models
Codex-D by minimizing negative log-likelihood of the docstring.

Figure 9. Optimal sampling temperatures as a function of the number of samples generated for both Codex and Codex-S. Codex-S
generally requires a higher temperature for any particular value of
k, possibly to compensate for the fact that it models a narrower
distribution.

When we benchmark our code generation models, we measure pass@k on the HumanEval dataset, where correctness
is deﬁned by passing a set of unit tests. However, there is
no similar way to evaluate docstring samples automatically.
Therefore, we grade sample docstrings by hand, considering
a docstring correct if it uniquely and accurately speciﬁes
the code body. Due to the time consuming nature of this
process, we only grade 10 samples per problem, for a total
of 1640 problems, from Codex-D-12B at temperature 0.8.

Codex-D often generates incorrect unit tests along with a
docstring, but we ignore these during grading. However,
we do not consider the docstring correct when the model
simply copies the code body into the docstring. The most
common failure modes we observe are when the docstring
model leaves out an important detail (such as “an answer
must be to two decimal places”) or when it over-conditions
on the function name and invents a problem unrelated to the
function body.

As shown in Table 3, pass rates for Codex-D are lower but
comparable to the corresponding pass rates for Codex-S at
the same temperature. We do not have a strong hypothesis
for which direction should yield higher pass rates. While
generating docstrings may be more forgiving because natural language syntax is less strict than code syntax, docstrings
in our dataset may be lower quality because developers tend
to devote less time to writing docstrings. Indeed, our model
produces docstrings like “I just found this function online”
and “This test is not correctly written and it’s not my solution.”

Figure 10. Comparing Codex-S against Codex on the metrics proposed in Section 3. Codex-S is one or two orders of magnitude
more parameter efﬁcient on pass@1 and pass@100, and log-prob
sample ranking with Codex-S yields similar beneﬁts over random
sampling that Codex does.

Finally, with a docstring model, we have yet another way
to choose a single sample from a set of k samples. Instead of picking the sample with the best mean log probability as investigated in the previous two sections, we can
choose the sample that maximizes the back-translation ob
Evaluating Large Language Models Trained on Code

Table 3. Pass rates for our docstring generating model Codex-D,
which is evaluated by hand-grading 10 samples per task due to the
lack of a ground-truth automatic evaluation. We ﬁnd similar but
lower pass-rates compared to Codex-S.

list is described in Appendix C). We ﬁnd that as the number
of chained building blocks in the docstring increases, model
performance decreases exponentially. This behavior is uncharacteristic of a human programmer, who should be able
to correctly implement a program for a chain of arbitrary
length if they can do so for a chain of length two.

MODEL
PASS@1
PASS@10

CODEX-S-12B
32.2%
59.5%
CODEX-D-12B
20.3%
46.5%

jective P(ground truth docstring|generated sample) where
P is evaluated using Codex-D. Unfortunately, in Figure 7,
we show that ranking samples via back-translation underperforms mean log-probability ranking, though it outperforms random ranking. This heuristic also appears to overﬁt
quickly.

6. Limitations

While Codex is able to sample correct solutions for the
majority of HumanEval problems, we ﬁnd that it has a
number of limitations.

Figure 11. Pass rates of Codex-12B samples against the number of
chained components in the synthetically generated docstring. With
each additional component, pass rate drops by roughly a factor of
2-3.

First, Codex is not sample efﬁcient to train. Our training
dataset comprises a signiﬁcant fraction of publicly available
Python code on GitHub, totaling hundreds of millions of
lines of code. Even seasoned developers do not encounter
anywhere near this amount of code over their careers. Indeed, a strong student who completes an introductory computer science course is expected to be able to solve a larger
fraction of problems than Codex-12B.

Further, just as text-conditional generative models in other
modalities (Ramesh et al., 2021) have difﬁculty with binding attributes to objects, Codex can make mistakes binding
operations to variables, especially when the number of operations and variables in the docstring is large. For instance,
in the following prompt, Codex-12B does not decrement the
variable w and also fails to return the product of all numbers.

def do_work(x, y, z, w):
""" Add 3 to y, then subtract 4
from both x and w. Return the
product of the four numbers. """
t = y + 3
u = x - 4
v = z * w
return v

This understanding of Codex’s limited system-level synthesis capabilities helps inform our assessment of the potential
hazards of using it in a generative capacity, as well as the
broader societal impacts that such systems could have.

7. Broader Impacts and Hazard Analysis

Next, we explore prompts on which Codex is likely to fail
or display counter-intuitive behavior. While evaluating code
generation is well-studied (Xu et al., 2021; Helmuth & Spector, 2015; Pantridge et al., 2017), many existing metrics
measure performance in tightly speciﬁed, constrained problem instances (e.g., string manipulation in FlashFill (Gulwani, 2011)). Therefore, we developed a set of qualitative
metrics for measuring the capabilities of code generating
models while controlling for the complexity and abstraction level of the speciﬁcations (Appendix D). Applying this
framework, we ﬁnd that Codex can recommend syntactically incorrect or undeﬁned code, and can invoke functions,
variables, and attributes that are undeﬁned or outside the
scope of the codebase. Moreover, Codex struggles to parse
through increasingly long and higher-level or system-level
speciﬁcations.

Codex has the potential to be useful in a range of ways.
For example, it could help onboard users to new codebases,
reduce context switching for experienced coders, enable
non-programmers to write speciﬁcations and have Codex
draft implementations, and aid in education and exploration.
However, Codex also raises signiﬁcant safety challenges,
does not always produce code that is aligned with user intent,

To concretely illustrate model performance degradation as
docstring length increases, we create a dataset of synthetic
problems assembled from 13 basic building blocks, each of
which modiﬁes an input string in a deterministic way. Example building blocks are “convert the string to lowercase”
or “remove every third character from the string” (the full

Evaluating Large Language Models Trained on Code

and has the potential to be misused.

To better understand some of the hazards of using Codex
in a generative capacity, we conducted a hazard analysis
focused on identifying risk factors (Leveson, 2019) with
the potential to cause harm.1 We outline some of our key
ﬁndings across several risk areas below.

Figure 12. When the prompt includes subtle bugs, Codex tends to
produce worse code than it is capable of. This persists when the
prompt also includes instructions to write correct code. This gap
increases with model size.

While some of our ﬁndings about the potential societal
impacts of code generation systems were informed by work
towards responsible deployment of the production-oriented
Codex models (which descended from the research-oriented
Codex models described in this paper), this section is not
intended to provide a full account of any particular product’s
safety features. Unless otherwise speciﬁed, we anchor our
analysis in the speciﬁc properties of the models described
in this paper. We share this analysis in the belief that some
of it generalizes to the broader class of code generation
systems, and to encourage a norm of performing detailed
impact analysis as part of major machine learning research
projects.

forward to provide documentation to users reminding them
about model limitations, empirical investigation is necessary in order to identify how to reliably ensure vigilance in
practice across a range of user experience levels, UI designs,
and tasks. One challenge researchers should consider is that
as capabilities improve, it may become increasingly difﬁcult
to guard against “automation bias.”

7.2. Misalignment

Note that by focusing largely on risks in this section, we do
not mean to imply that we expect the impact of this class of
technologies to be net-negative; rather, risks merit particular
attention here because they may be subtle or require deliberate effort to address, whereas we expect the beneﬁts to be
more obvious and “automatic” from the perspective of most
users and affected stakeholders.

7.1. Over-reliance

As with other large language models trained on a next-token
prediction objective, Codex will generate code that is as similar as possible to its training distribution. One consequence
of this is that such models may do things that are unhelpful
for the user, despite having the capability to be more helpful
(see Figure 12). For example, if the user has some subtle
mistakes in their code, Codex may “deliberately” suggest
code that superﬁcially appears good but is incorrect.

One of the key risks associated with using code generation
models in practice is over-reliance on generated outputs.
Due to the limitations described above as well as alignment
issues described below, Codex may suggest solutions that
superﬁcially appear correct but do not actually perform the
task the user intended. This could particularly affect novice
programmers, and could have signiﬁcant safety implications
depending on the context. We discuss a related issue in
Appendix G, namely that code generation models can suggest insecure code. For these reasons, human oversight and
vigilance is required for safe use of code generation systems
like Codex.

This is an alignment failure - the model is not aligned with
the user’s intentions. Informally, a system is misaligned if
there’s some task X that we want it to do, and it is “capable”
of doing X but “chooses” not to. In contrast, if a system
fails to do X because it does not have the ability to do so,
then this system is not misaligned; it is just incompetent.
See Appendix E for more detail, including a more precise
deﬁnition of alignment.

We note several immediate ways to improve safety in the
subsection on risk mitigation below, though over-reliance
in particular is one that we believe merits further inquiry
in industry and academia. While it is conceptually straight
It is important to study misalignment because it is a problem
that is likely to become worse, not better, as the capabilities of our systems increase. For example, the model size
scaling trend for the example in Figure 12 indicates that
misalignment would likely persist and even get worse if
data, parameters, and training time were scaled up.

1We sought to include harms spanning geographic and temporal
scales. We also considered not only the severity and probability,
but also the distribution of harms. However, we note that the
analysis described here is only one milestone in what we hope will
be a larger cross-sectoral and cross-organizational effort to steer
code generation in a societally beneﬁcial direction. As we describe
our ﬁndings, we note various speciﬁc uncertainties and areas for
future work in different sections.

While we expect that misaligned behaviour like this is unlikely to cause signiﬁcant harm in current models, it is likely
to become more dangerous and harder to eliminate as model

Evaluating Large Language Models Trained on Code

7.5. Security implications

Codex could have various effects on the security landscape.
Because Codex can produce vulnerable or misaligned code,3

capabilities increase. A highly capable but sufﬁciently misaligned model trained on user approval might produce obfuscated code that looks good to the user even on careful
inspection, but in fact does something undesirable or even
harmful.

7.3. Bias and representation

qualiﬁed operators should review its generations before executing or trusting them, absent appropriate precautions.
Future code generation models may be able to be trained
to produce more secure code than the average developer,
though that is far from certain.

Codex could also be misused to aid cybercrime. Although
this is worthy of concern, based on our testing, we believe
that at their current level of capability, Codex models do
not materially lower the barrier to entry for malware development.4 We expect that more powerful code generation
models will lead to future advancements, and therefore further research into mitigations and continued study of model
capabilities are necessary.

Mirroring what has been found in the case of other language
models trained on Internet data (Bender et al., 2021; Blodgett et al., 2020; Abid et al., 2021; Brown et al., 2020), we
found that Codex can be prompted in ways that generate
racist, denigratory, and otherwise harmful outputs as code
comments, meriting interventions such as those discussed
in the subsection on risk mitigation below. We also found
that code generation models raise further bias and representation issues beyond problematic natural language: Codex
can generate code with structure that reﬂects stereotypes
about gender, race, emotion, class, the structure of names,
and other characteristics. Particularly in the context of users
who might over-rely on Codex or use it without ﬁrst thinking through project design, this issue could have signiﬁcant
safety implications, giving further motivation to discourage
over-reliance. We discuss bias and representation issues
further in Appendix F. Filtration or modulation of generated
outputs, documentation, and other interventions may help
to mitigate these risks.

7.4. Economic and labor market impacts

The non-deterministic nature of systems like Codex could
enable more advanced malware. This non-determinism
makes it easier to create diverse software that accomplish
the same tasks. While software diversity can sometimes
aid defenders,5 it presents unique challenges for traditional
malware detection and antivirus systems that rely on ﬁngerprinting and signature-matching against previously sampled
binaries. For example, a more capable code generation
model could conceivably advance techniques for generating
polymorphic malware.6 We believe that application security and model deployment strategies including rate-limiting
access and abuse monitoring can manage this threat in the
near term; however, the efﬁcacy of these mitigations may
scale sublinearly as more capable models are developed.

Similar to large language models, Codex models can learn
patterns present in their training data (Carlini et al., 2021).
Sensitive data present in source code are liable to be predicted by the model. Because Codex is trained on public
repositories, we consider any sensitive data present in the
training data to have already been compromised. Similarly,
the public data should generally be treated as untrusted, as
previous work (Goldblum et al., 2021; Schuster et al., 2020)
has found that attackers may be able to corrupt training data
to trigger speciﬁc model behaviors at runtime. We further
discuss security implications in Appendix G.

Code generation and associated capabilities have several
possible economic and labor market impacts. While Codex
at its current capability level may somewhat reduce the cost
of producing software by increasing programmer productivity, the size of this effect may be limited by the fact that
engineers don’t spend their full day writing code (O*NET,
2021). Other important tasks include conferring with colleagues, writing design speciﬁcations, and upgrading existing software stacks.2 We also found that Codex imports
packages at different rates, which could advantage some
package authors over others, particularly if programmers
and engineers come to rely on Codex’s suggestions. Over a
longer time horizon, the effects of this class of technologies
on software-related labor markets and on the economy more
generally could be more substantial as capabilities improve.
More study is needed both on the effects of code generation capabilities and on appropriate responses. We discuss
economic and labor market implications in more detail in
Appendix H.

3See Appendix G - Insecure Code for examples of Codex producing insecure code.
4For more on characterizing Codex’s capability limitations, see
the Limitations section and experiments in the security analysis in
Appendix G.
5For example, by helping to prevent certain types of memory
corruption vulnerabilities. See (Davis, 2018) for more.
6Polymorphic malware is malicious code that mutates its implementation while maintaining its function.

2Indeed, BLS classiﬁes computer programmers and software
developers separately, where developers are more highly paid than
programmers, have more tasks indirectly related to writing and
interacting with code, and, in the US, are already projected to see
greater demand over the next 10 years (Li et al., 2020; Bureau of
Labor Statistics, 2021a;b).

Evaluating Large Language Models Trained on Code

7.6. Environmental impacts

features that exist as features of other tools of authorship
(e.g., document editors), in the sense that the ﬁnished work
is still seen as the author’s.

Our commitment to responsible and safe AI includes continued attention to the broader intellectual property implications of code generation systems. We intend to remain
engaged with policymakers and experts on these issues so
that the users of such systems can ultimately deploy them
with conﬁdence.

7.8. Risk mitigation

Codex, like other large generative models, has an energy
footprint from both training and inference (Schwartz et al.,
2019; Bender et al., 2021; Patterson et al., 2021). The original training of GPT-3-12B consumed hundreds of petaﬂop/sdays of compute, while ﬁne-tuning it to create Codex-12B
consumed a similar amount of compute. This training was
performed on a platform (Azure) that purchases carbon
credits and sources signiﬁcant amounts of renewable energy,
reducing its carbon footprint.7 Compute consumption also
has costs in the wider supply chain that can be quite concentrated on certain regions.8 Looking more globally and
long-term, the compute demands of code generation could
grow to be much larger than Codex’s training if signiﬁcant
inference is used to tackle challenging problems.9

7.7. Legal implications

In closing, given the above, models like Codex should be
developed, used, and their capabilities explored carefully
with an eye towards maximizing their positive social impacts and minimizing intentional or unintentional harms that
their use might cause. A contextual approach is critical to
effective hazard analysis and mitigation, though a few broad
categories of mitigations are important to consider in any
deployment of code generation models.

There are several legal considerations related to generated
code. To begin with, the training of AI systems on Internet
data, such as public GitHub repositories, has previously
been identiﬁed as an instance of “fair use” (O’Keefe et al.,
2019).

Careful documentation and user interface design, code review requirements, and/or content controls (e.g., ﬁltering
of outputs) may help to reduce harms associated with overreliance as well as offensive content or insecure code generation. In the context of a model made available as a service
(e.g., via an API), policies such as user review, use case
restrictions, monitoring, and/or rate limiting may also help
to reduce harms associated with malicious use or prevent
its use in high-stakes domains for which the models are not
well suited.

Appendices E, F, G, and H provide further detail on the risks
described in this section and outline additional mitigation
and research opportunities.

Our preliminary research also ﬁnds that Codex models rarely
generate code that is identical to the contents of training
data. Such occurrences were < 0.1% in a study examining
the frequency of code generations that appear to match code
snippets in the training data (Ziegler, 2021). In these rare
instances, the generated code consisted of common expressions or conventions within the programming language that
appeared over and over again in the training data. We ﬁnd
that, to the extent the generated code appears identical to
the training data, it is due to the predictive weightings in the
model rather than retention and copying of speciﬁc code.

8. Related Work

Generated code is also responsive and customized to the
user’s input, and the user retains complete control over
editing and acceptance of the generated code. This can make
code generation similar to auto-suggest or auto-completion

The deep learning resurgence has led to strong advances in
the ﬁeld of program learning. Two popular approaches to
neural program learning are program induction and program
synthesis.

7Microsoft made a commitment in 2020 to shift to 100 percent renewable energy supply in its buildings and data centers
by 2025. https://blogs.microsoft.com/blog/2020/01/16/microsoftwill-be-carbon-negative-by-2030/ A full assessment of the environmental impact of compute use is impossible to conduct without
grounding in context and making comparison to the counterfactual
impacts of competing products or services. Such analysis is out of
scope for this paper.
8While data center energy usage has become much more efﬁcient in recent years (Masanet et al., 2020), the production, use,
and disposal of semiconductors still imposes environmental and
human costs. See, e.g., (Crawford, 2021)
9Given that code generation (and other forms of AI) might be
deployed widely throughout the economy as discussed above, these
considerations suggest additional urgency in adopting renewable
energy.

In program induction, a model generates program outputs
directly from a latent program representation. Learning to
Execute (Zaremba & Sutskever, 2014) demonstrated that
models could execute simple tasks like addition and memorization. Later attempts at program induction incorporated
inductive biases based on modern computing devices, such
as the Neural Turing Machine (Graves et al., 2014), memory
networks (Weston et al., 2015; Sukhbaatar et al., 2015), the
Neural GPU (Kaiser & Sutskever, 2015), and the differentiable neural computer (Graves et al., 2016). More recent
approaches like the Neural Program Interpreter (Reed &
de Freitas, 2016; Shin et al., 2018; Pierrot et al., 2021) and

Evaluating Large Language Models Trained on Code

Universal Transformer (Dehghani et al., 2019) found recurrence to be a useful component in program induction.

In program synthesis, a model explicitly generates a program, usually from a natural language speciﬁcation. One
of the most popular classical approaches used a probabilistic context free grammar (PCFG) to generate a program’s
abstract syntax tree (AST). Maddison & Tarlow (2014) improved on this setup by learning a state vector used to condition child node expansion. Later, Allamanis et al. (2015)
applied this idea in text-to-code retrieval and Yin & Neubig (2017) utilized it in text-conditional code generation.
Code2seq (Alon et al., 2018) found that ASTs could also be
leveraged for code-to-text generation.

ral programming systems were FlashFill (Gulwani, 2011;
Gulwani et al., 2012) and Hearthstone (Ling et al., 2016),
though the community has trended towards broader and
more difﬁcult datasets. Barone & Sennrich (2017) proposed
a large training and evaluation dataset consisting of Python
declarations, docstrings, and bodies scraped from GitHub.
The CodeSearchNet challenge (Husain et al., 2019) built
an even larger corpus from GitHub with data from multiple
popular programming languages. Recently, CodeXGLUE
(Lu et al., 2021) aggregated several programming benchmarks, making use of the recently proposed CodeBLEU
metric (Ren et al., 2020). Most relevant to our evaluation
work is the APPS (Hendrycks et al., 2021) benchmark for
measuring functional correctness based on problems from
the competitive programming website Codeforces.

Programs can also be synthesized without passing through
an AST representation. Hindle et al. (2012) investigated
n-gram language models of code, ﬁnding code to be more
predictable than natural language. Latent Predictor Networks (Ling et al., 2016) showed that character-level language models could generate working code for implementing Magic the Gathering cards in an online arena, when
aided with a latent mode that allows card attributes to be
copied into code. DeepCoder (Balog et al., 2017) trained
a model to predict the functions appearing in source code,
which could be used to guide program search.

Following the success of large natural language models (Devlin et al., 2018; Radford et al., 2019; Liu et al., 2019; Raffel
et al., 2020; Brown et al., 2020) large scale Transformers
have also been applied towards program synthesis. CodeBERT (Feng et al., 2020) trained the BERT objective on
docstrings paired with functions, and obtained strong results
on code search. PyMT5 (Clement et al., 2020) is similar in
spirit to our work, and used the T5 objective to train a system which can translate between non-overlapping subsets
of {signature, docstring, body}.

Finally, we note that coding is a broad activity which involves much more than synthesizing code from docstrings.
Tufano et al. (2020) use Transformers to generate unit tests
for code which outperformed commercial offerings. Aye
et al. (2021) built an internal auto-complete tool for Facebook, and found that training on accepted user completions
boosted system performance. Development also entails locating and ﬁxing bugs. Early works used static or dynamic
code analysis (Agrawal et al., 1995; Korel & Rilling, 1997),
learned association rules (Jeffrey et al., 2009), and genetic
programming (Goues et al., 2012) to debug faulty code.
These approaches relied on running against a test suite to
not only evaluate the correctness of suggestions but also
expose problems in execution trace or search for a solution.
More recent works (Tufano et al., 2019; Drain et al., 2021)
considered bug-ﬁxing as neural machine translation from
buggy to correct programs. However, these works used an
exact match against a reference instead of functional correctness, citing Qi et al. (2015)’s ﬁnding that most of the
proposed solutions by genetic search in (Goues et al., 2012)
passed through weak test suites by deleting functionality
that failed. Human developers often write test suites with
limited but targeted coverage, but this does not always work
well against an algorithm, highlighting the challenges of
evaluating correctness of programs.

9. Conclusion

We used functional correctness to benchmark our models,
and observed improvements on this metric with more sampling. SPoC (Kulal et al., 2019) considered the problem
of producing functionally correct code from pseudocode
with a ﬁxed budget of compilations, which is similar to our
pass@k metric. TransCoder (Lachaux et al., 2020) trained
a system to translate between programming languages in
an unsupervised manner, and also observed that functional
correctness better captured the capabilities of their model
than BLEU score. In fact, ContraCode (Jain et al., 2020)
leveraged the large space of functionally correct programs
to train a contrastive code model, which improved model
performance on tasks like type inference. Finally, RobustFill (Devlin et al., 2017) observed that the best way to ﬁnd
a program consistent with input examples was to synthesize
multiple samples through beam search.

Two early domain-speciﬁc datasets used to benchmark neu
We investigated whether it was possible to train large language models to produce functionally correct code bodies
from natural language docstrings. By ﬁne-tuning GPT on
code from GitHub, we found that our models displayed
strong performance on a dataset of human-written problems
with difﬁculty level comparable to easy interview problems.
Model performance could be improved by training on a
distribution more similar to the evaluation set, and also by
producing multiple samples from a model. We also found
that it was simple to train a model to complete the reverse

Evaluating Large Language Models Trained on Code

Alon, U., Brody, S., Levy, O., and Yahav, E. code2seq: Generating sequences from structured representations of code. In
International Conference on Learning Representations, 2018.

task of producing docstrings from code bodies, and that the
performance proﬁles of these models were similar. Finally,
we expanded on the broader impacts of code generating
models, and discussed model limitations, ﬁnding signiﬁcant
room for improvement.

Aye, G. A., Kim, S., and Li, H. Learning autocompletion from realworld datasets. 2021 IEEE/ACM 43rd International Conference
on Software Engineering: Software Engineering in Practice
(ICSE-SEIP), pp. 131–139, 2021.

Acknowledgements

Baevski, A., Zhou, H., Mohamed, A., and Auli, M. wav2vec 2.0:
A framework for self-supervised learning of speech representations. arXiv preprint arXiv:2006.11477, 2020.

Balog, M., Gaunt, A., Brockschmidt, M., Nowozin, S., and Tarlow,
D. Deepcoder: Learning to write programs. In 5th International
Conference on Learning Representations (ICLR), 2017.

Bao, H., Dong, L., and Wei, F. Beit: Bert pre-training of image
transformers. arXiv preprint arXiv:2106.08254, 2021.

Barone, A. V. M. and Sennrich, R. A parallel corpus of python
functions and documentation strings for automated code documentation and code generation. ArXiv, abs/1707.02275, 2017.

We thank Sandhini Agarwal, Casey Chu, Jeffrey Ding, Peter Eckersley, Gillian Hadﬁeld, Rich Harang, Jacob Jackson, Yunxin Jiao, Jade Leung, Andrew Lohn, Ryan Lowe,
Thomas McGuire, Margaret Mitchell, Florentine Eloundou
Nekoul, Cullen O’Keefe, Long Ouyang, Pranav Shyam,
Irene Solaiman, Aravind Srinivas, Helen Toner, Ashish
Vaswani, and Jeffrey Wu for helpful discussions and feedback on drafts of this work. We are also grateful to the Acceleration and Supercomputing teams at OpenAI for their work
on software and hardware infrastructure that this project
used. Finally, we thank GitHub for partnering to build
GitHub Copilot and Microsoft Azure for supporting model
training with infrastructure management.

Barrington, I. M. and Maciel, A. Lecture 3: Nondeterministic computation. https://people.clarkson.edu/˜alexis/
PCMI/Notes/lectureB03.pdf, 2000. [Online; accessed
29-June-2000].

References

Bender, E. M., Gebru, T., McMillan-Major, A., and Shmitchell,
S. On the dangers of stochastic parrots: Can language models
be too big? In Proceedings of the 2021 ACM Conference on
Fairness, Accountability, and Transparency, pp. 610–623, 2021.

Cwe-327: Use of a broken or risky cryptographic algorithm, 2006.
URL https://cwe.mitre.org/data/definitions/
327.html.

Cwe-780: Use of rsa algorithm without oaep, 2009. URL https:
//cwe.mitre.org/data/definitions/780.html.

Black, S., Gao, L., Wang, P., Leahy, C., and Biderman, S.
GPT-Neo:
Large scale autoregressive language modeling
with mesh-tensorﬂow, 2021. URL http://github.com/
eleutherai/gpt-neo.

A6:2017-security misconﬁguration,
2017.
URL https:
//owasp.org/www-project-top-ten/2017/
A6 2017-Security Misconfiguration.html.

Blodgett, S. L., Barocas, S., Daum´e III, H., and Wallach, H. Language (technology) is power: A critical survey of “bias” in nlp.
arXiv preprint arXiv:2005.14050, 2020.

Abid, A., Farooqi, M., and Zou, J. Persistent anti-muslim bias in
large language models. arXiv preprint arXiv:2101.05783, 2021.

Acemoglu, D. and Restrepo, P. Robots and jobs: Evidence from us
labor markets. Journal of Political Economy, 128(6):2188–2244,
2020a.

Acemoglu, D. and Restrepo, P. The wrong kind of ai? artiﬁcial intelligence and the future of labour demand. Cambridge Journal
of Regions, Economy and Society, 13(1):25–35, 2020b.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.,
Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell,
A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T.,
Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse,
C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark,
J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and
Amodei, D. Language models are few-shot learners. ArXiv,
abs/2005.14165, 2020.

Agrawal, H., Horgan, J. R., London, S., and Wong, W. E. Fault
localization using execution slices and dataﬂow tests. Proceedings of Sixth International Symposium on Software Reliability
Engineering. ISSRE’95, pp. 143–151, 1995.

Bureau of Labor Statistics, U. D. o. L. Computer programmers.
Occupational Outlook Handbook, 2021a.
URL https:
//www.bls.gov/ooh/computer-and-informationtechnology/computer-programmers.htm.

Bureau of Labor Statistics, U. D. o. L. Bls - software developers.
Occupational Outlook Handbook, 2021b.
URL https:
//www.bls.gov/ooh/computer-and-informationtechnology/software-developers.htm.

Allamanis, M., Tarlow, D., Gordon, A., and Wei, Y. Bimodal modelling of source code and natural language. In Bach, F. and Blei,
D. (eds.), Proceedings of the 32nd International Conference
on Machine Learning, volume 37 of Proceedings of Machine
Learning Research, pp. 2123–2132, Lille, France, 07–09 Jul
2015. PMLR. URL http://proceedings.mlr.press/
v37/allamanis15.html.

Alley, E. C., Khimulya, G., Biswas, S., AlQuraishi, M., and
Church, G. M.
Uniﬁed rational protein engineering with
sequence-based deep representation learning. Nature methods,
16(12):1315–1322, 2019.

Carlini, N., Tram`er, F., Wallace, E., Jagielski, M., Herbert-Voss,
A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson,
U., Oprea, A., and Raffel, C. Extracting training data from
large language models.
In 30th USENIX Security Symposium (USENIX Security 21). USENIX Association, August
2021. URL https://www.usenix.org/conference/

Evaluating Large Language Models Trained on Code

usenixsecurity21/presentation/carliniextracting.

Eghbal, N. Working in public: the making and maintenance of
open source software. Stripe Press, 2020.

Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D.,
and Sutskever, I. Generative pretraining from pixels. In International Conference on Machine Learning, pp. 1691–1703.
PMLR, 2020.

Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou,
L., Qin, B., Liu, T., Jiang, D., et al. Codebert: A pre-trained
model for programming and natural languages. In Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pp. 1536–1547, 2020.

Frey, C. B. The technology trap. Princeton University Press, 2019.

Child, R., Gray, S., Radford, A., and Sutskever, I. Generating long
sequences with sparse transformers. ArXiv, abs/1904.10509,
2019.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster,
C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S.,
and Leahy, C. The pile: An 800gb dataset of diverse text for
language modeling. 2020.

Christiano, P. Clarifying ”ai alignment”. AI Alignment Forum,
2018.
URL https://www.alignmentforum.org/
posts/ZeE7EKHTFMBs8eMxn/clarifying-aialignment.

Goldblum, M., Tsipras, D., Xie, C., Chen, X., Schwarzschild, A.,
Song, D., Madry, A., Li, B., and Goldstein, T. Dataset security
for machine learning: Data poisoning, backdoor attacks, and
defenses, 2021.

Clarkson, M. R., Finkbeiner, B., Koleini, M., Micinski, K. K.,
Rabe, M. N., and S´anchez, C. Temporal logics for hyperproperties. In International Conference on Principles of Security and
Trust, pp. 265–284. Springer, 2014.

Goues, C. L., Dewey-Vogt, M., Forrest, S., and Weimer, W. A
systematic study of automated program repair: Fixing 55 out of
105 bugs for $8 each. 2012 34th International Conference on
Software Engineering (ICSE), pp. 3–13, 2012.

Clement, C., Drain, D., Timcheck, J., Svyatkovskiy, A., and Sundaresan, N. Pymt5: Multi-mode translation of natural language
and python code with transformers. In Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 9052–9065, 2020.

Graves, A. Generating sequences with recurrent neural networks,
2014.

Graves, A., Wayne, G., and Danihelka, I. Neural turing machines.
arXiv preprint arXiv:1410.5401, 2014.

Crawford, K.
The trouble with bias.
NIPS 2017 Keynote,
2017.
URL https://www.youtube.com/watch?v=
fMym BKWQzk.

Crawford, K. Atlas of AI: Power, Politics, and the Planetary Costs
of Artiﬁcial Intelligence. Yale University Press, 2021.

Graves, A., Wayne, G., Reynolds, M., Harley, T., Danihelka, I.,
Grabska-Barwi´nska, A., Colmenarejo, S. G., Grefenstette, E.,
Ramalho, T., Agapiou, J., et al. Hybrid computing using a
neural network with dynamic external memory. Nature, 538
(7626):471–476, 2016.

Dai, A. M. and Le, Q. V. Semi-supervised sequence learning.
Advances in neural information processing systems, 28:3079–
3087, 2015.

Gulwani, S. Automating string processing in spreadsheets using input-output examples. In PoPL’11, January 26-28, 2011,
Austin, Texas, USA, January 2011.

Das, A., Kottur, S., Gupta, K., Singh, A., Yadav, D., Moura, J. M.,
Parikh, D., and Batra, D. Visual dialog. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition,
pp. 326–335, 2017.

Gulwani, S., Harris, W. R., and Singh, R. Spreadsheet data manipulation using examples. Commun. ACM, 55:97–105, 2012.

He, P., Liu, X., Gao, J., and Chen, W.
Deberta: Decodingenhanced bert with disentangled attention.
arXiv preprint
arXiv:2006.03654, 2020.

Davis, B.
Protecting applications with automated software
diversity, Sep 2018. URL https://galois.com/blog/
2018/09/protecting-applications-withautomated-software-diversity.

Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Łukasz
Kaiser. Universal transformers, 2019.

Helmuth, T. and Spector, L. General program synthesis benchmark
suite. In Proceedings of the 2015 Annual Conference on Genetic
and Evolutionary Computation, pp. 1039–1046, 2015.

Devlin, J., Uesato, J., Bhupatiraju, S., Singh, R., rahman Mohamed,
A., and Kohli, P. Robustﬁll: Neural program learning under
noisy i/o. In ICML, 2017.

Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A.,
Guo, E., Burns, C., Puranik, S., He, H., Song, D., et al. Measuring coding challenge competence with apps. arXiv preprint
arXiv:2105.09938, 2021.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pretraining of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Hindle, A., Barr, E. T., Su, Z., Gabel, M., and Devanbu, P. On the
naturalness of software. In 2012 34th International Conference
on Software Engineering (ICSE), pp. 837–847. IEEE, 2012.

Dhariwal, P., Jun, H., Payne, C., Kim, J. W., Radford, A., and
Sutskever, I. Jukebox: A generative model for music. arXiv
preprint arXiv:2005.00341, 2020.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The
curious case of neural text degeneration, 2020.

Drain, D., Wu, C., Svyatkovskiy, A., and Sundaresan, N. Generating bug-ﬁxes using pretrained transformers. Proceedings of
the 5th ACM SIGPLAN International Symposium on Machine
Programming, 2021.

Husain,
H.,
Wu,
H.-H.,
Gazit,
T.,
Allamanis,
M.,
and
Brockschmidt, M. Codesearchnet challenge: Evaluating the
state of semantic code search. ArXiv, abs/1909.09436, 2019.

Evaluating Large Language Models Trained on Code

Jain, P., Jain, A., Zhang, T., Abbeel, P., Gonzalez, J., and
Stoica, I. Contrastive code representation learning. ArXiv,
abs/2007.04973, 2020.

Lu, J., Batra, D., Parikh, D., and Lee, S. Vilbert: Pretraining taskagnostic visiolinguistic representations for vision-and-language
tasks. arXiv preprint arXiv:1908.02265, 2019.

Jeffrey, D., Feng, M., Gupta, N., and Gupta, R. Bugﬁx: A learningbased tool to assist developers in ﬁxing bugs. 2009 IEEE 17th
International Conference on Program Comprehension, pp. 70–
79, 2009.

Lu, S., Guo, D., Ren, S., Huang, J., Svyatkovskiy, A., Blanco, A.,
Clement, C., Drain, D., Jiang, D., Tang, D., Li, G., Zhou, L.,
Shou, L., Zhou, L., Tufano, M., Gong, M., Zhou, M., Duan, N.,
Sundaresan, N., Deng, S. K., Fu, S., and Liu, S. Codexglue:
A machine learning benchmark dataset for code understanding
and generation. ArXiv, abs/2102.04664, 2021.

Jones, C. and Bonsignour, O. The economics of software quality.
Addison-Wesley Professional, 2011.

Kaiser, Ł. and Sutskever, I. Neural gpus learn algorithms. arXiv
preprint arXiv:1511.08228, 2015.

Maddison, C. J. and Tarlow, D. Structured generative models of
natural source code. In Proceedings of the 31st International
Conference on International Conference on Machine Learning
(ICML), pp. II–649, 2014.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess,
B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D.
Scaling laws for neural language models, 2020.

Manna, Z. and Waldinger, R. J.
Toward automatic program
synthesis.
14(3):151–165, March 1971.
ISSN 0001-0782.
doi: 10.1145/362566.362568.
URL https://doi.org/
10.1145/362566.362568.

Kenton, Z., Everitt, T., Weidinger, L., Gabriel, I., Mikulik, V.,
and Irving, G. Alignment of language agents. arXiv preprint
arXiv:2103.14659, 2021.

Masanet, E., Shehabi, A., Lei, N., Smith, S., and Koomey, J.
Recalibrating global data center energy-use estimates. Science,
367(6481):984–986, 2020.

Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., and Socher,
R. Ctrl: A conditional transformer language model for controllable generation, 2019.

Korel, B. and Rilling, J. Application of dynamic slicing in program
debugging. In AADEBUG, 1997.

Menezes, A., van Oorschot, P., and Vanstone, S. Handbook of
Applied Cryptography. Discrete Mathematics and Its Applications. CRC Press, 2018. ISBN 9780429881329. URL https:
//books.google.com/books?id=YyCyDwAAQBAJ.

Koza, J. R., Andre, D., Keane, M. A., and Bennett III, F. H. Genetic
programming III: Darwinian invention and problem solving,
volume 3. Morgan Kaufmann, 1999.

Menick, J. and Kalchbrenner, N. Generating high ﬁdelity images
with subscale pixel networks and multidimensional upscaling,
2018.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean,
J. Distributed representations of words and phrases and their
compositionality. In Advances in neural information processing
systems, pp. 3111–3119, 2013.

Kulal, S., Pasupat, P., Chandra, K., Lee, M., Padon, O.,
Aiken,
A.,
and Liang,
P. S.
Spoc:
Search-based
pseudocode to code.
In Wallach, H., Larochelle, H.,
Beygelzimer, A., d'Alch´e-Buc, F., Fox, E., and Garnett,
R. (eds.),
Advances in Neural Information Processing
Systems, volume 32. Curran Associates, Inc., 2019.
URL
https://proceedings.neurips.cc/paper/2019/
file/7298332f04ac004a0ca44cc69ecf6f6bPaper.pdf.

Ohm, M., Plate, H., Sykosch, A., and Meier, M. Backstabber’s
knife collection: A review of open source software supply chain
attacks, 2020.

Lacasse, N. Open-sourcing gvisor, a sandboxed container runtime,
2018.

Lachaux, M.-A., Rozi`ere, B., Chanussot, L., and Lample, G.
Unsupervised translation of programming languages. ArXiv,
abs/2006.03511, 2020.

O’Keefe, C., Lansky, D., Clark, J., and Payne, C. Comment regarding request for comments on intellectual property protection
for artiﬁcial intelligence innovation. Before the United States
Patent and Trademark Ofﬁce Department of Commerce, 2019.
URL https://perma.cc/ZS7G-2QWF.

Leveson, N. Improving the standard risk matrix: Part 1. 2019.
URL http://sunnyday.mit.edu/Risk-Matrix.pdf.

O*NET.
15-1252.00 - software developers, 2021.
URL
https://www.onetonline.org/link/summary/151252.00.

Li, P. L., Ko, A. J., and Begel, A. What distinguishes great software
engineers? Empirical Software Engineering, 25(1):322–352,
2020.

Oord, A. v. d., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O.,
Graves, A., Kalchbrenner, N., Senior, A., and Kavukcuoglu, K.
Wavenet: A generative model for raw audio. arXiv preprint
arXiv:1609.03499, 2016.

Ling, W., Blunsom, P., Grefenstette, E., Hermann, K. M., Koˇcisk`y,
T., Wang, F., and Senior, A. Latent predictor networks for code
generation. In Proceedings of the 54th Annual Meeting of the
Association for Computational Linguistics (ACL), pp. 599–609,
2016.

Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with
contrastive predictive coding. arXiv preprint arXiv:1807.03748,
2018.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D.,
Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V.
Roberta: A robustly optimized bert pretraining approach. ArXiv,
abs/1907.11692, 2019.

O’Neill, M. and Spector, L. Automatic programming: The open
issue?
Genetic Programming and Evolvable Machines, pp.
1–12, 2019.

Evaluating Large Language Models Trained on Code

Pantridge, E., Helmuth, T., McPhee, N. F., and Spector, L. On
the difﬁculty of benchmarking inductive program synthesis
methods. In Proceedings of the Genetic and Evolutionary Computation Conference Companion, pp. 1589–1596, 2017.

Rokon, M. O. F., Islam, R., Darki, A., Papalexakis, E. E., and
Faloutsos, M. Sourceﬁnder: Finding malware source-code
from publicly available repositories in github.
In 23rd International Symposium on Research in Attacks, Intrusions
and Defenses (RAID 2020), pp. 149–163, San Sebastian,
October 2020. USENIX Association. ISBN 978-1-93913318-2.
URL https://www.usenix.org/conference/
raid2020/presentation/omar.

Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L.M., Rothchild, D., So, D., Texier, M., and Dean, J. Carbon
emissions and large neural network training. arXiv preprint
arXiv:2104.10350, 2021.

Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C.,
Lee, K., and Zettlemoyer, L. Deep contextualized word representations. arXiv preprint arXiv:1802.05365, 2018.

Schuster, R., Song, C., Tromer, E., and Shmatikov, V.
You
autocomplete me: Poisoning vulnerabilities in neural code
completion.
The Advanced Computing Systems Association, 2020. URL https://www.usenix.org/system/
files/sec21summer schuster.pdf.

Schwartz, R., Dodge, J., Smith, N. A., and Etzioni, O. Green ai,
2019.

Pierrot, T., Ligner, G., Reed, S., Sigaud, O., Perrin, N., Laterre, A.,
Kas, D., Beguir, K., and de Freitas, N. Learning compositional
neural programs with recursive tree search and planning, 2021.

Shin, E. C., Polosukhin, I., and Song, D. Improving neural program
synthesis with inferred execution traces. Advances in Neural
Information Processing Systems, 31:8917–8926, 2018.

Planning, S. The economic impacts of inadequate infrastructure for
software testing. National Institute of Standards and Technology,
2002.

Simon, H. A.
Experiments with a heuristic compiler.
J.
ACM, 10(4):493–506, October 1963.
ISSN 0004-5411.
doi: 10.1145/321186.321192.
URL https://doi.org/
10.1145/321186.321192.

Python Software Foundation and JetBrains.
Python developers survey 2020 results,
2020.
URL
https:
//www.jetbrains.com/lp/python-developerssurvey-2020/.

Stack Overﬂow.
2020 developer survey,
2020.
URL
https://insights.stackoverflow.com/survey/
2020#overview.

Qi, Z., Long, F., Achour, S., and Rinard, M. An analysis of patch
plausibility and correctness for generate-and-validate patch generation systems. Proceedings of the 2015 International Symposium on Software Testing and Analysis, 2015.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss,
C., Radford, A., Amodei, D., and Christiano, P. Learning to
summarize from human feedback, 2020.

Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I.
Improving language understanding by generative pre-training.
2018.

Sukhbaatar, S., Szlam, A., Weston, J., and Fergus, R. End-to-end
memory networks, 2015.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and
Sutskever, I.
Language models are unsupervised multitask
learners. 2019.

Sutskever, I., Vinyals, O., and Le, Q. V. Sequence to sequence
learning with neural networks. In Advances in neural information processing systems, pp. 3104–3112, 2014.

Trinkenreich, B., Wiese, I., Sarma, A., Gerosa, M., and Steinmacher, I. Women’s participation in open source software: A
survey of the literature. arXiv preprint arXiv:2105.08777, 2021.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.
Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.

Tufano, M., Watson, C., Bavota, G., Penta, M. D., White, M.,
and Poshyvanyk, D. An empirical study on learning bug-ﬁxing
patches in the wild via neural machine translation. ACM Transactions on Software Engineering and Methodology (TOSEM),
28:1 – 29, 2019.

Raffel, C., Shazeer, N. M., Roberts, A., Lee, K., Narang, S.,
Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the
limits of transfer learning with a uniﬁed text-to-text transformer.
ArXiv, abs/1910.10683, 2020.

Tufano, M., Drain, D., Svyatkovskiy, A., Deng, S. K., and Sundaresan, N. Unit test case generation with transformers and
focal context. 2020.

Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A.,
Chen, M., and Sutskever, I. Zero-shot text-to-image generation.
ArXiv, abs/2102.12092, 2021.

Reed, S. and de Freitas, N. Neural programmer-interpreters, 2016.

Van Oord, A., Kalchbrenner, N., and Kavukcuoglu, K. Pixel recurrent neural networks. In International Conference on Machine
Learning, pp. 1747–1756. PMLR, 2016.

Ren, S., Guo, D., Lu, S., Zhou, L., Liu, S., Tang, D., Sundaresan,
N., Zhou, M., Blanco, A., and Ma, S. Codebleu: a method
for automatic evaluation of code synthesis.
arXiv preprint
arXiv:2009.10297, 2020.

Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo,
D., Ott, M., Zitnick, C. L., Ma, J., et al. Biological structure
and function emerge from scaling unsupervised learning to
250 million protein sequences. Proceedings of the National
Academy of Sciences, 118(15), 2021.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention
is all you need. In Guyon, I., Luxburg, U. V., Bengio, S.,
Wallach, H., Fergus, R., Vishwanathan, S., and Garnett,
R. (eds.),
Advances in Neural Information Processing
Systems, volume 30. Curran Associates, Inc., 2017.
URL
https://proceedings.neurips.cc/paper/2017/
file/3f5ee243547dee91fbd053c1c4a845aaPaper.pdf.

Evaluating Large Language Models Trained on Code

Wang, B. and Komatsuzaki, A. GPT-J-6B: A 6 Billion Parameter
Autoregressive Language Model. https://github.com/
kingoflolz/mesh-transformer-jax, May 2021.

Weston, J., Chopra, S., and Bordes, A. Memory networks, 2015.

Woolf, M. Fun and dystopia with ai-based code generation using gpt-j-6b, June 2021. URL https://minimaxir.com/
2021/06/gpt-j-6b/.

Xu, F. F., Vasilescu, B., and Neubig, G. In-ide code generation
from natural language: Promise and challenges. arXiv preprint
arXiv:2101.11149, 2021.

Yin, P. and Neubig, G. A syntactic neural model for generalpurpose code generation. In Proceedings of the 55th Annual
Meeting of the Association for Computational Linguistics (ACL),
pp. 440–450, 2017.

Zaremba, W. and Sutskever, I. Learning to execute. arXiv preprint
arXiv:1410.4615, 2014.

Zellers, R., Lu, X., Hessel, J., Yu, Y., Park, J. S., Cao, J., Farhadi,
A., and Choi, Y. Merlot: Multimodal neural script knowledge
models. arXiv preprint arXiv:2106.02636, 2021.

Zhao, T. Z., Wallace, E., Feng, S., Klein, D., and Singh, S. Calibrate before use: Improving few-shot performance of language
models. arXiv preprint arXiv:2102.09690, 2021.

Figure 13. Comparing the amount of bias and variance of two
estimators of pass@k. While the top expression may look correct,
it underestimates the true value by a considerable margin. The
unbiased estimator may have a slightly higher variance initially but
allows for a fair comparison across different numbers of samples.

Ziegler, A. A ﬁrst look at rote learning in github copilot suggestions., Jun 2021. URL https://docs.github.com/en/
github/copilot/research-recitation.

"

#

#

A. Estimating pass@k

1 −

Ec

= 1 −Ec

 n−c
k

 n
k


" n−c
k

 n
k


!

n−k
X

pi(1 −p)n−i

= 1 −

n
i

 n−i
k

 n
k


i=0

!

n−k
X

= 1 −

pi(1 −p)n−i

n −k
i

i=0

!

= 1 −(1 −p)k
n−k
X

pi(1 −p)n−k−i

n −k
i

i=0

= 1 −(1 −p)k.

While all estimators mentioned previously are consistent,
only the empirical estimate used by Kulal et al. (2019),
and (1) are unbiased. Evaluating pass@k in an unbiased
way with any number of samples n is important for fair
comparison. For example, estimating pass@k = 1 −(1 −
pass@1)k with 1 −(1 −ˆp)k using the empirical pass@1,
results in a consistent underestimate as shown in Figure 13.
The gap doesn’t fully close even when n > 5k, and results
can seem better with more samples. The interpretation of
this estimator is that we draw k samples with replacement
from a pool of n candidates, but the k samples are not
independent.

B. Random Problems and Solutions from
Codex-12B

We show 8 random problems from HumanEval along with
8 random samples per problem generated from Codex-12B
at temperature 0.8.

def words_string(s):
"""

(1) is unbiased, because it estimates the fail probability
(1−pass@1)k as the probability of drawing k failed samples
without replacement. To show this, note that c, the number
of correct samples that pass the unit tests, is distributed
Binom(n, p), where p is pass@1, and that (1) evaluates to 1
when n −c < k. Then,

Evaluating Large Language Models Trained on Code

You will be given a string of words separated by
commas or spaces. Your task is
to split the string into words and return an array
of the words.

### COMPLETION 1 (WRONG): ###
for i in range(2, n):
if n % i == 0:
return False
return True

For example:
words_string("Hi, my name is John") == ["Hi", "my",
"name", "is", "John"]
words_string("One, two, three, four, five, six") ==
["One", "two", "three", "four", "five", "six"]
"""

### COMPLETION 2 (WRONG): ###
for i in range(2, n):
if n % i == 0:
return False
return True

### COMPLETION 3 (WRONG): ###
# TODO: implement this function
pass

### COMPLETION 1 (CORRECT): ###
words = []
splitted_word = ""
for char in s:
if char == " ":
if splitted_word != "":
words.append(splitted_word)
splitted_word = ""
elif char != ",":
splitted_word += char
if splitted_word != "":
words.append(splitted_word)
return words

### COMPLETION 4 (CORRECT): ###
if n < 2: return False
if n == 2: return True
if n%2 == 0: return False
return not any(n%k == 0 for k in range(3,int(n
**0.5)+1,2))

### COMPLETION 2 (WRONG): ###
return s.split()

### COMPLETION 3 (WRONG): ###
words = s.split(" ")
words = [w.strip(",") for w in words]
words = [w.strip() for w in words]
return words

### COMPLETION 5 (WRONG): ###
def helper(n):
# check if an odd number is multiple of any
integer up to the square root of the number
# if this is true, n is not prime
for i in range(2, int(n**0.5 + 1)):
if n % i == 0:
return False
return True
# if helper returns true, then n is prime, else n
is not prime
if helper(n):
return True
else:
return False

### COMPLETION 4 (WRONG): ###
# Split the string into words
words = s.split(’,’)
# Remove any whitespace around each element
words = [w.strip() for w in words]
# Return an array of the words in the string
return words

### COMPLETION 5 (WRONG): ###
return s.split()

### COMPLETION 6 (CORRECT): ###
prime = True
if n == 1:
return False
for i in range(2, n):
if n % i == 0:
prime = False
return prime

### COMPLETION 6 (WRONG): ###
return s.split()

### COMPLETION 7 (WRONG): ###
return s.split(’,’)

### COMPLETION 7 (WRONG): ###
for i in range(2, n):
if n % i == 0:
return False
return True

### COMPLETION 8 (WRONG): ###
return s.split(sep=None)

### COMPLETION 8 (CORRECT): ###
if n == 2 or n == 3:
return True
if n < 2 or n % 2 == 0:
return False
if n < 9:
return True
if n % 3 == 0:
return False
r = int(n**0.5)
f = 5
while f <= r:
if n % f == 0:
return False
if n % (f+2) == 0:
return False
f += 6
return True

def is_prime(n):
"""Return true if a given number is prime, and
false otherwise.
>>> is_prime(6)
False
>>> is_prime(101)
True
>>> is_prime(11)
True
>>> is_prime(13441)
True
>>> is_prime(61)
True
>>> is_prime(4)
False
>>> is_prime(1)
False
"""

---

*Source: arXiv:2107.03374*
