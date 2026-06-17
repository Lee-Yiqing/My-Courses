---
id: A5
title: "Large Language Models for Software Engineering: A Systematic Literature Review"
domain: A
year: 2024
arxiv_id: "2308.10620"
confidence: verified
source: "arXiv:2308.10620"
node_type: paper
---

# Large Language Models for Software Engineering: A Systematic Literature Review

**Domain**: [[domain_A|Vibe Coding / Prompt-Driven Development]] | **Year**: 2024 | **Confidence**: [x] verified


## Authors
[[author_Xinyi Hou|Xinyi Hou]], [[author_Yanjie Li|Yanjie Li]], [[author_Hao Chen|Hao Chen]], et al.


## Keywords
- [[kw_LLM4SE|LLM4SE]]
- [[kw_systematic literature review|systematic literature review]]
- [[kw_software engineering|software engineering]]
- [[kw_LLM survey|LLM survey]]

## Abstract

(Abstract not available - see PDF content below)

## Paper Content

1

Large Language Models for Software Engineering: A
Systematic Literature Review

XINYI HOU∗, Huazhong University of Science and Technology, China
YANJIE ZHAO∗, Huazhong University of Science and Technology, China
YUE LIU, Monash University, Australia
ZHOU YANG, Singapore Management University, Singapore
KAILONG WANG, Huazhong University of Science and Technology, China
LI LI, Beihang University, China
XIAPU LUO, The Hong Kong Polytechnic University, China
DAVID LO, Singapore Management University, Singapore
JOHN GRUNDY, Monash University, Australia
HAOYU WANG†, Huazhong University of Science and Technology, China

Large Language Models (LLMs) have significantly impacted numerous domains, including Software Engineering (SE). Many recent publications have explored LLMs applied to various SE tasks. Nevertheless, a
comprehensive understanding of the application, effects, and possible limitations of LLMs on SE is still in its
early stages. To bridge this gap, we conducted a systematic literature review (SLR) on LLM4SE, with a particular focus on understanding how LLMs can be exploited to optimize processes and outcomes. We select and
analyze 395 research papers from January 2017 to January 2024 to answer four key research questions (RQs).
In RQ1, we categorize different LLMs that have been employed in SE tasks, characterizing their distinctive
features and uses. In RQ2, we analyze the methods used in data collection, preprocessing, and application,
highlighting the role of well-curated datasets for successful LLM for SE implementation. RQ3 investigates
the strategies employed to optimize and evaluate the performance of LLMs in SE. Finally, RQ4 examines the
specific SE tasks where LLMs have shown success to date, illustrating their practical contributions to the
field. From the answers to these RQs, we discuss the current state-of-the-art and trends, identifying gaps
in existing research, and flagging promising areas for future study. Our artifacts are publicly available at
https://github.com/xinyi-hou/LLM4SE_SLR.

CCS Concepts: • General and reference →Surveys and overviews; • Software and its engineering →
Software development techniques; • Computing methodologies →Artificial intelligence.

Additional Key Words and Phrases: Software Engineering, Large Language Model, Survey

∗Co-first authors who contributed equally to this work.
†Haoyu Wang is the corresponding author (haoyuwang@hust.edu.cn).

arXiv:2308.10620v6  [cs.SE]  10 Apr 2024

Authors’ addresses: Xinyi Hou, xinyihou@hust.edu.cn, Huazhong University of Science and Technology, Wuhan, China;
Yanjie Zhao, yanjie_zhao@hust.edu.cn, Huazhong University of Science and Technology, Wuhan, China; Yue Liu, yue.liu1@
monash.edu, Monash University, Melbourne, Australia; Zhou Yang, zyang@smu.edu.sg, Singapore Management University,
Singapore; Kailong Wang, wangkl@hust.edu.cn, Huazhong University of Science and Technology, Wuhan, China; Li Li,
lilicoding@ieee.org, Beihang University, Beijing, China; Xiapu Luo, csxluo@comp.polyu.edu.hk, The Hong Kong Polytechnic
University, Hong Kong, China; David Lo, davidlo@smu.edu.sg, Singapore Management University, Singapore; John Grundy,
John.Grundy@monash.edu, Monash University, Melbourne, Australia; Haoyu Wang, haoyuwang@hust.edu.cn, Huazhong
University of Science and Technology, Wuhan, China.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and
the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2024 Association for Computing Machinery.
1049-331X/2024/12-ART1 $15.00
https://doi.org/XXXXXXX.XXXXXXX

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

1:2
X Hou, Y Zhao, Y Liu, Z Yang, K Wang, L Li, X Luo, D Lo, J Grundy, and H Wang

ACM Reference Format:
Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang, Kailong Wang, Li Li, Xiapu Luo, David Lo, John Grundy, and Haoyu
Wang. 2024. Large Language Models for Software Engineering: A Systematic Literature Review. ACM Trans.
Softw. Eng. Methodol. X, Y, Article 1 (December 2024), 79 pages. https://doi.org/XXXXXXX.XXXXXXX

1
INTRODUCTION
In the field of language processing, traditional Language Models (LMs) have been foundational
elements, establishing a basis for text generation and understanding [293]. Increased computational
power, advanced machine learning techniques, and access to very large-scale data have led to a
significant transition into the emergence of Large Language Models (LLMs) [526, 558]. Equipped
with expansive and diverse training data, these models have demonstrated an impressive ability to
simulate human linguistic capabilities, leading to a sea of changes across multiple domains. With
their capacity to learn from massive corpora and generate plausible text, LLMs are blurring the line
between human and machine-produced language. They have provided researchers and engineers
alike with a powerful tool to explore the complexity and richness of human communication,
consequently sparking a transformational period in the field of language processing and beyond.
Software Engineering (SE) – a discipline focused on the development, implementation, and
maintenance of software systems – is one of those areas reaping the benefits of the LLM revolution [275]. The utilization of LLMs in SE primarily emerges from an innovative perspective where numerous SE challenges can be effectively reframed into data, code, or text analysis tasks [452]. Using LLMs to address these SE tasks has shown a wealth of potential breakthroughs [33, 37, 210, 399, 427, 488, 489, 536]. The applicability of LLMs is particularly pronounced
in tasks such as code summarization [443], which involves yielding an abstract natural language
depiction of a code’s functionality, as well as the generation of well-structured code [515] and
code artifacts like annotations [245]. Codex, an LLM with 12 billion parameters, has demonstrated
the ability to solve 72.31% of complex Python programming challenges posed by humans [43].
GPT-4 [320], an LLM from OpenAI, has been used with a strong performance in several SE tasks,
encompassing code writing, understanding, execution, and reasoning. It not only handles real-world
applications and diverse coding challenges but also shows the ability to explain results in natural
language and generate code from pseudocode [30].
Simultaneously, researchers have embarked on a series of research activities regarding LLMrelated works, where a number of literature reviews or survey papers have been produced [36, 87,
506]. Table 1 summarises some of these. However, these related studies have limitations. They either
focus narrowly on a single SE scope, such as the application of LLMs in software testing [448] and
natural-language-to-code (NL2Code) tasks [526], or they are primarily centered on Machine Learning (ML) or Deep Learning (DL) models [452, 466, 509], overlooking more advanced and recently
emerged LLM applications, such as ChatGPT [317], which are increasingly finding applications
within the SE field [269, 400, 427, 475]. Alternatively, they merely offer a preliminary exploration of
the performance of LLMs in various SE tasks through empirical experiments [74, 275, 400, 493, 521],
or analyze existing partially relevant studies to reveal the challenges in this field [85] without
conducting a systematic literature survey. Furthermore, some works have investigated the application of Code LLMs in SE [543, 564], yet have not fully considered general LLMs like ChatGPT and
LLaMA [431], which are also widely applied to various SE tasks [144, 325, 382, 497]. The integration
of LLMs within SE is undoubtedly a complex endeavor, requiring key considerations including
the choice of the right model, comprehension of the unique features of different LLMs, devising
pre-training and fine-tuning strategies, handling of data, evaluation of outcomes, and surmounting
implementation challenges [526]. Despite the burgeoning interest and ongoing explorations in the
field, a detailed and systematic review of LLMs’ application in SE has been notably absent

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

Large Language Models for Software Engineering: A Systematic Literature Review
1:3

Table 1. State-of-the-art surveys related to LLMs for SE.

Reference
Year
Scope of models1
Scope of SE tasks
SLR2 Time frame
# Collected Papers
Zhang et al. [543]
2023
Code LLM
Automated program repair
✓
2017-2023
185
Zheng et al. [564]
2023
Code LLM
General SE scope
✓
2021-2023
149
Fan et al. [85]
2023
LLM
General SE scope
×
Not specified
Zan et al. [526]
2023
LLM (12M+)
NL2Code
×
2020-2023
Not specified
Wang et al. [448]
2023
LLM (117M+)
Software testing
✓
2019-2023
52
Wang et al. [452]
2022
ML, DL3
General SE scope
✓
2009-2020
1,209 (ML) + 358 (DL)
Yang et al. [509]
2022
DL
General SE scope
✓
2015-2020
250
Watson et al. [466]
2022
DL
General SE scope
✓
2009-2019
128
Our work
2024
LLM
General SE scope
✓
2017-2024
395

1 “M” means million and “B” means billion. The numbers in parentheses indicate the parameter sizes of LLMs.
2 SLR stands for Systematic Literature Review. This column denotes whether the paper follows an SLR process.
3 ML and DL refer to Machine Learning and Deep Learning, respectively.

in the current literature. This gap signifies a need for understanding the relationship between
LLMs and SE. In response, our research aims to bridge this gap, providing valuable insights to the
community.
In this paper, we conduct an SLR on the utilization of LLMs in SE (LLM4SE). By mapping
the current state-of-the-art, pinpointing the key strengths, weaknesses, and gaps in the existing
LLM4SE literature, and proposing potential avenues for future research, our review aims to provide
researchers and practitioners with a thorough guide to the convergence of LLMs and SE. We
anticipate that our findings will be instrumental in guiding future inquiries and advancements in
this rapidly evolving field. This work makes the following key contributions:

• We are the first to present a comprehensive SLR on 395 papers published between January 2017
and January 2024 that focus on the use of LLM-based solutions to address SE challenges. We
conducted a detailed analysis of the selected papers based on publication trends, distribution
of publication venues, etc.
• We have classified the LLMs utilized for the reported SE tasks and have provided a summary
of the usage and trends of different LLM categories within the SE domain.
• We describe the reported data processing stages, encompassing data collection, categorization,
preprocessing, and representation.
• We discuss optimizers used for LLM4SE tasks, including tuning techniques, prevalent prompt
engineering techniques, and commonly employed evaluation metrics.
• We describe the key applications of LLM4SE encompassing a diverse range of 85 specific
SE tasks, grouped into six core SE activities – requirements engineering, software design,
software development, software quality assurance, software maintenance, and software
management.
• We have summarised key challenges that using LLMs encounters within the SE field and
have suggested several potential research directions for LLM4SE.
Section 2 presents our research questions (RQs) and elaborates on our SLR methodology. The
succeeding Sections 3 to 6 are devoted to answering each of these RQs individually. Section 7
discloses the potential threats to the validity of our study. Section 8 discusses the challenges yet to
be overcome when employing LLMs to solve SE tasks and highlights promising opportunities and
directions for future research. Section 9 concludes the whole paper.

2
APPROACH
This SLR follows the methodology proposed by Kitchenham et al. [197, 198], used in most other
SE-related SLRs [229, 261, 352, 452]. Following the guidelines provided by Kitchenham et al., our

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

1:4
X Hou, Y Zhao, Y Liu, Z Yang, K Wang, L Li, X Luo, D Lo, J Grundy, and H Wang

methodology included three main steps: planning the review (i.e., Section 2.1, 2.2), conducting the
review (i.e., Section 2.3, 2.4), and analyzing the basic review results (i.e, Section 2.5).

2.1
Research Questions
To provide a comprehensive overview of the LLM4SE field, it is important to fully comprehend
how these models are currently being applied in SE, the challenges they face, and their potential
future research directions in SE. Thus, we aim to provide an SLR of the application of LLMs to
software engineering. This study thus aims to answer the following research questions:
RQ1: What LLMs have been employed to date to solve SE tasks? RQ1 is designed to map out
the landscape of LLMs applied in the field of SE. It seeks to identify and categorize the various LLM
architectures—such as decoder-only, encoder-decoder, and encoder-only models—that have been
leveraged to address diverse SE challenges. This RQ aims to provide a comprehensive overview of
how these models are being utilized and the implications of their usage in this field.
RQ2: How are SE-related datasets collected, preprocessed, and used in LLMs? RQ2 delves
into the methodologies behind the assembly, refinement, and application of datasets in the realm of
LLMs for SE tasks. It aims to uncover the strategies for dataset collection, the criteria for dataset
selection, and the preprocessing steps essential for making the data conducive for LLM training and
application. Additionally, this question seeks to explore the types of data that are most prevalent in
SE-related LLM research and how these data types influence the modeling and outcomes.
RQ3: What techniques are used to optimize and evaluate LLM4SE? RQ3 aims to explore
the use of different optimization and evaluation techniques specific to LLMs in the context of SE.
This includes an investigation into Parameter Efficient Fine-Tuning (PEFT) methods and various
prompting techniques that are tailored to enhance LLM performance on SE tasks. Furthermore,
this RQ aims to assess the range of evaluation metrics and methodologies employed to gauge the
effectiveness and impact of LLMs in SE, providing insights into how these models are fine-tuned
and assessed for their utility and efficiency.
RQ4: What SE tasks have been effectively addressed to date using LLM4SE? This RQ aims
to identify the SE tasks that have been successfully tackled using LLMs, offering a detailed view of
the application spectrum of LLMs in SE. It seeks to identify the specific tasks within SE, such as
code generation and program repair, where LLMs have shown significant utility, and to explore the
nature and scope of these applications.

2.2
Search Strategy

As shown in Fig.1, we employed the “Quasi-Gold Standard” (QGS) [531] approach for paper search.
We conducted a manual search to identify a set of relevant studies and extracted a search string
from them. This search string was then used to perform an automated search, and subsequently, a
snowballing search was employed to further supplement the search results. This approach ensures
both search efficiency and maximum coverage, minimizing the risk of omission. Subsequently, we
employed a series of relatively strict filtering steps to obtain the most relevant studies. Specifically,
we followed five steps to determine the relevance of the studies:

(1) Select publication venues for manual search and select digital databases for automated search
to ensure coverage of all the selected venues.
(2) Establish QGS: Screen all papers for manual search and filter by inclusion/exclusion criteria
(defined in Table 3).
(3) Subjectively define the search string based on domain knowledge.
(4) Conduct an automated search using the search string defined in Step (3).

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

Large Language Models for Software Engineering: A Systematic Literature Review
1:5

Study Identiﬁcation

Automated Search

Science

IEEE Xplore
ACM Digital

DBLP

Direct

Web of 
Science
Springer
arXiv

Library

Large Language

Derive
search strings

4,035 papers

1,192 papers
10,445 papers
65,290 papers
42,166 papers
85,671 papers
9,966 papers

Model (LLM)

Snowballing Search

Export

Evaluate

Reﬁne 
search 
strings

Complement

Research 
Question 1-4

13,565 papers

forward
backward

Manual Search

Identify
relevant venues

Export

3,964 papers
9,601 papers

Export

Study selection

6 selected 
SE venues

218,765 papers

Add 13 papers

Total 395 papers

Software 
Engineering

51 papers

Study Selection

Filter out studies

Check the title,

Scan full-text to

Remove duplicate

Conduct quality

with less than 8

abstract, and

Identify venue

select primary

studies

assessment

pages

keywords

studies

80,611 papers

1,172 papers

810 papers

594 papers

382 papers

4,341 papers
4,341 papers
5,078 papers

Fig. 1. Study identification and selection process.

(5) Conduct snowballing search after performing study selection on the results of manual search
and automated search.

2.2.1
Search Items. During the manual search, we selected six of the top SE conferences and
journals (i.e., ICSE, ESEC/FSE, ASE, ISSTA, TOSEM, and TSE, as shown in Table 2) and searched for
papers that applied LLM4SE. We systematically crawled a list comprising 4,618 published papers
from the top venues. Following automated scanning via scripts, we manually verified and identified
51 papers that were relevant to our research objectives. These 51 relevant papers formed the basis
for constructing the Quasi-Gold Standard (QGS). Our search string should combine two sets of
keywords: one pertaining to SE tasks, and the other related to LLMs. Only if the paper contains
both types of keywords, there is a higher probability that it is the paper we need. The complete set
of search keywords is as follows:

• Keywords related to SE tasks: Software Engineering, Software Development, Program*1, Software
Testing, Software Mainten*, SE, Software Lifecycle, Software Design*, Code representation,
Code generation, Code comment generation, Code search, Code localization, Code completion,
Code summarization, Method name generation, Bug detection, Bug localization, Vulnerability
detection, Testing techniques, Test case generation, Program analysis, Bug classification, Defect
prediction, Program repair, Code clone detection, Bug report, Software quality evaluation, SATD
detection, Code smell detection, Compiled-related, Code review, Software classification, Code
classification, Code change, Incident detection, Requirement extraction, Requirement traceability,
Requirement validation, Effort cost prediction, Mining GitHub/Github mining, Mining SO (Stack
Overflow)/SO mining, Mining app/App mining, Mining tag/Tag mining, Developer-based mining

1The * symbol serves as a wildcard, representing any characters or character sequence. For example, “Program*” can match
“Program”, “Programming”, “Programmer”, and so on.

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

1:6
X Hou, Y Zhao, Y Liu, Z Yang, K Wang, L Li, X Luo, D Lo, J Grundy, and H Wang

Table 2. Publication venues for manual search.

Acronym
Venues
ASE
International Conference on Automated Software Engineering
ESEC/FSE
Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering
ICSE
International Conference on Software Engineering
ISSTA
International Symposium on Software Testing and Analysis
TOSEM
Transactions on Software Engineering and Methodology
TSE
Transactions on Software Engineering

Table 3. Inclusion criteria and Exclusion criteria.

Inclusion criteria
1)
The paper claims that an LLM is used.
2)
The paper claims that the study involves an SE task.
3)
The paper with accessible full text.
Exclusion criteria
1)
Short papers whose number of pages is less than 8.
2)
Duplicate papers or similar studies with different versions from the same authors.
3)
Studies belonging to books, thesis, monographs, keynotes, panels, or venues not executing a full
peer-review process.
4)
Tool demos and editorials.
5)
The paper is published in a workshop or a doctoral symposium.
6)
The paper is a grey publication, e.g., a technical report or thesis.
7)
Non-English written literature.
8)
The paper mentions the use of LLMs without describing the employed techniques.
9)
The paper leverages SE methods to enhance LLMs, rather than focusing on using LLMs for SE tasks.

• Keywords related to LLMs: LLM, Large Language Model*, Language Model*, LM, PLM, Pretrained, Pre-training, Natural Language Processing, NLP, Machine Learning, ML, Deep Learning,
DL, Artificial Intelligence, AI, Transformer, BERT, Codex, GPT, T5, Sequence Model*, Attention
Model*, Transfer Learning, Neural Network*, ChatGPT, GPT-*
It is important to note that the list of keywords related to LLMs that we set up includes Machine
Learning, Deep Learning, and other such terms that do not seem to be necessarily related to LLMs.
The reason for this is that we want to avoid omitting papers related to our research as much as
possible, so the process of performing automated searches expands our search scope.

2.2.2
Search Datasets. After determining the search string, we conducted an automated search
across seven widely used databases, which are capable of covering all published or latest papers.
Given that the first paper about the Transformer architecture [436], which forms the basis for LLMs,
was published in 2017, we focused our search on papers published from that year onward2. Two
authors independently performed the automated search, and the search results from each database
were merged and deduplicated. Specifically, we obtained 1,192 papers from IEEE Xplore, 10,445
papers from the ACM Digital Library, 62,290 papers from ScienceDirect, 42,166 papers from Web of
Science, 85,671 papers from Springer, 9,966 papers from arXiv, and 4,035 papers from DBLP.

2.3
Study Selection
2.3.1
Study Inclusion and Exclusion Criteria. Based on our search strategy, we initially obtained
218,765 papers that potentially relate to our research. Next, we needed to further evaluate the
relevance of these papers based on inclusion and exclusion criteria (To ensure that our inclusion
and exclusion criteria were sufficiently objective and rational, we designed these criteria following
several state-of-the-art SLR papers [302, 452, 466, 509].), as shown in Table 3, so that the selected
papers can directly address our research questions. The paper selection process, as illustrated in

2The cut-off date for the paper collection process of this version is January 31, 2024.

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

Large Language Models for Software Engineering: A Systematic Literature Review
1:7

Table 4. Checklist of Quality Assessment Criteria (QAC) for LLM studies in SE.

ID
Quality Assessment Criteria
QAC1
Is the study relevant to SE tasks?
QAC2
Does the study utilize LLMs?
QAC3
Is the research not a secondary study, such as an SLR, review, or survey?
QAC4
Was the research published in a high-repute venue?
QAC5
Is there a clear motivation for the research?
QAC6
Does the study provide a clear description of the techniques used?
QAC7
Are the experimental setups, including experimental environments and
dataset information, described in detail?
QAC8
Does the study clearly confirm the experimental findings?
QAC9
Are the key contributions and limitations of the study discussed?
QAC10
Does the study make a contribution to the academic or industrial community?

Fig. 1, consists of six phases. In the first phase, we conducted automated filtering to exclude papers
with less than 8 pages [23, 452] (Exclusion criteria 1), reducing the number of papers to 80,611. In
the second phase, we examined the titles, abstracts, and keywords of the papers to identify those
that include relevant LLM-related keywords. We then expanded the search scope to avoid missing
relevant papers, including ML, DL, and other related keywords that may not directly correspond to
LLM. The purpose of this phase is to narrow down the scope and filter out papers directly related
to LLM (Inclusion criteria 1). Papers that are filtered out in this phase are then manually reviewed
in the fifth phase. Additionally, we excluded 448 non-English written literature (Exclusion criteria
7). After the second phase, the number of papers was reduced to 5,078.
The third phase involves identifying the venues of the papers (Exclusion criteria 3). We extracted
publication information such as “journal”, “URL”, “DOI”, and “series” to determine the publication
sources. For papers from arXiv in 2023 and 2024, we chose to retain them, considering that this
field is emerging and many works are in the process of submission. Although these papers did not
undergo peer review, we have a quality assessment process to eliminate papers with low quality.
This step resulted in 1,172 papers.
In the fourth phase, we merged and deduplicated the remaining papers from the seven databases
and the manually searched paper list (Exclusion criteria 2), resulting in 810 papers. We then reviewed
the full texts of the papers and excluded 190 papers that were grey publications or were published
in workshops or doctoral symposiums (Exclusion criteria 4, 5, 6). By further assessing the quality
of the papers, we identified 382 papers directly relevant to our research. This phase primarily
involved excluding papers that mentioned LLMs but did not directly apply them, such as papers
that only discussed LLMs in future work or focused on evaluating the performance of LLM-enabled
tools [448] (Exclusion criteria 8). For systematic views, survey, and review papers, we have retained
them and will assess their content during the quality assessment phase to determine their relevance
to our research.

2.3.2
Study Quality Assessment. A well-crafted quality assessment can help to prevent biases
introduced by low-quality studies and can indicate to readers where caution about conclusions
should be drawn [508]. We formulated ten Quality Assessment Criteria (QAC), as shown in Table 4.
These aim to assess the relevance, clarity, validity, and significance of included papers. We used a
scoring system of -1, 0, 1 (irrelevant/unmet, partially relevant/met, relevant/fully met). The first
three questions were designed for the remaining 382 papers in the fifth stage. If QAC1, QAC2, or
QAC3 received a score of -1, there is no need to proceed with QAC4-QAC10, and the paper can
be excluded directly. QAC4-QAC10 involved assessing the content of the papers using a scoring
system of 0, 1, 2, 3 (poor, fair, good, excellent). Finally, we calculated the total score of QAC4-QAC10
for each paper. For published papers, the maximum score for QAC4-QAC10 should be 21 (3 × 7).

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

1:8
X Hou, Y Zhao, Y Liu, Z Yang, K Wang, L Li, X Luo, D Lo, J Grundy, and H Wang

TSE, 14

300

273

ESEC/FSE, 12

ICSE, 41

TOSEM, 11

250

ASE, 10

200

SANER, 10

ICSME, 9

arXiv, 241

150

EMNLP, 7

100

ISSTA, 7

Number of papers

56

46

ICML, 6

50

7
13

ICPC, 5

0

NeurIPS, 5 
Others, 17

2020
2021
2022
2023
2024

(a) Distribution of papers across venues.

(b) Distribution of papers over years.

Fig. 2. Overview of the selected 395 papers’ distribution.

Fig. 3. Topics discussed in the collected papers.

We retained papers with a score of 16.8 (21 × 0.8) or above. For unpublished papers on arXiv, the
score for QAC4 is always 0, and the maximum score for QAC5-QAC10 should be 18 (3 × 6). We
retained papers with a score of 14.4 (18 × 0.8) or above. After this quality assessment, we obtained
a final set of 382 papers.

2.4
Snowballing Search
To identify any additional possibly relevant primary studies, we conducted a snowballing search.
Snowballing refers to using the reference list of a paper or the citations to the paper to identify
additional papers. Snowballing could benefit from not only looking at the reference lists and
citations but also complementing them with a systematic way of looking at where papers are
actually referenced and where papers are cited. Using the references and the citations respectively
is referred to as backward and forward snowballing.
Before conducting snowballing, a set of initial papers needs to be prepared. In this study, the
initial paper list consists of the remaining 382 papers after the quality assessment. We performed
forward and backward snowballing, which resulted in the collection of 3,964 and 9,610 papers,
respectively. After initial deduplication, we were left with 5,152 papers. We then conducted the full
study selection process on these 5,152 papers, including deduplicating them with the 382 papers
from performing snowballing on the initial list. As a result, we obtained an additional 13 papers.

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

Large Language Models for Software Engineering: A Systematic Literature Review
1:9

Table 5. Extracted data items and related research questions (RQs).

RQ
Data Item
1,2,3,4
The category of SE task
1,2,3,4
The category of LLM
1,4
Characteristics and applicability of LLMs
2
The adopted data handling techniques
3
The adopted weight training algorithms and optimizer
3
The selected evaluation metrics
4
The SE activity to which the SE task belongs
4
The developed strategies and solutions

2.5
Data Extraction and Analysis
We finally obtained 395 relevant research papers after searching and snowballing. Fig. 2 presents
an overview of the distribution of the included papers. As shown in Fig. 2 (a), 154 papers are
published in peer-reviewed venues. ICSE is the most common of these venues, with a contribution
of 41 papers. Other venues with noteworthy contributions include TSE, ESEC/FSE, and TOSEM,
contributing 14, 12, and 11 papers respectively. Meanwhile, the remaining 241 papers are published
on arXiv, an open-access platform that serves as a repository for scholarly articles. This finding is
not surprising since much new LLM4SE research is rapidly emerging and thus many works are
just completed and are likely in the peer review process. Despite the non-peer-reviewed nature of
these papers, we have performed a rigorous quality assessment process on all collected papers, to
ensure the quality of validity of our findings. This approach allows us to include all high-quality
and relevant publications while maintaining high research standards.
Fig. 2 (b) shows the temporal distribution of the included papers. The number of publications
has seen a rapidly growing trend since 2020. In 2020 and 2021, there are only 7 and 13 relevant
papers, respectively. However, by 2022, the number of papers has increased dramatically to 56.
What’s surprising is that, in 2023 alone, the number of published papers has already reached 273.
And within just one month in 2024, 46 relevant papers are published. This rapid growth trend
demonstrates that there is a growing research interest in the domain of LLM4SE.
In order to visualize the main content of our collection of papers, we generated a word cloud
based on the abstracts of 395 papers as shown in Fig. 3. The most frequently occurring words
include “code”, “LLM”, “language”, “model”, “large”, “task”, “software”,“generation”, “performance”,
and “program”, clearly indicating the main themes explored in these papers. The terms “code” and
“software” emphasize the core elements of software engineering, while “LLM”, “large”, “language”
and “model” denote the use of large language models in a variety of tasks. The terms “generation”,
“task”, and “program” emphasize the use of the LLM for automatic code generation and other SE
tasks. In addition, “performance” reflects the evaluation and assessment of the effectiveness of LLM
in SE applications. The word cloud provides further visual evidence that the literature we have
collected is closely related to our research topic.
We then conducted data extraction during the full-text review. This extraction phase collected
all relevant data that would facilitate a comprehensive and insightful response to the RQs outlined
in Section 2.1. As depicted in Table 5, we extracted data including the classification of SE tasks,
their corresponding activities, as well as the category, characteristics, and applicability of the LLMs.
With this collected data, we systematically analyzed the relevant aspects of LLM4SE.

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

1:10
X Hou, Y Zhao, Y Liu, Z Yang, K Wang, L Li, X Luo, D Lo, J Grundy, and H Wang

ALBERT (6)

BERTOverﬂow (3)

seBERT (2)

Encoder-only

BERT (50)

CuBERT (3)

GraphCodeBERT (25)

CodeRetriever (1)

Features

Encoder

Input Text

PRCBERT (1)

Trace BERT (3)

RoBERTA (24)

CostSens BERT (1)

CodeBERT (51)

Sentence-BERT (2)

Encoder-decoder

CoTexT (4)
AlphaCode (6)

CoditT5 (1)

Output Text

BART
PLBART (15)

Codetrans (2)

NatGen (2)

CodeT5+ (7)

Decoder

Features

Encoder

T5 (20)
CodeT5 (46)

UniXcoder (16)

CodeReviewer (2)
SPT-Code (2)

Input Text

CodeGeeX (8)
InstructGPT (5)
CodeParrot (6)
PolyCoder (8)

CodeGeeX2 (3)

StableLM (1)

CodeLlama (19)

Decoder-only

GPT-2 (17)

GPT-Neo (13)
ChatGPT (72)

CodeGPT (26)
CodeGen (44)

LaMDA (2)

WizardCoder (12)

CodeLlama2 (1)

GPT-1 (4)

Output Text

GPT-3.5 (54)

GPT-3 (12)
GPT-4 (53)

GPT-J (13)

CodeGen2 (7)

LLaMA (14)

Llama2 (10)
Llama2-Chat (2)

Decoder

GPT-NeoX (5)

PaLM2 (1)

Bard (2)

Vicuna (11)

SantaCoder (5)
CodeFuse (1)

Input Text

XLNet (4)

Codex (62)

BLOOM (5)
InCoder (29)

PaLM (4)
PaLM-Coder (3)
Claude (3)

DeepSeek Coder (1)

DeepSeek (3)

Copilot (7)

PyCodeGPT (5)
StarCoder (25)
Claude2 (2)
OPT (5)
PanGu-Coder (1)

2024

2018
2019
2020
2021
2022
2023

Fig. 4. Distribution of the LLMs (as well as LLM-based applications) discussed in the collected papers. The
numbers in parentheses indicate the count of papers in which each LLM has been utilized.

3
RQ1: WHAT LLMS HAVE BEEN EMPLOYED TO DATE TO SOLVE SE TASKS?

3.1
Large Language Models (LLMs)
Pre-trained language models (PLMs) have demonstrated impressive capabilities in solving various NLP tasks [202, 381, 468, 558]. Researchers have observed that scaling up the model sizes
significantly enhances their capacity, leading to remarkable performance improvements when the
parameter scale surpasses a certain threshold [137, 381, 422]. The term “Large Language Model”
(LLM) was introduced to distinguish language models based on their parameter size, specifically
referring to large-sized PLMs [558]. However, we note that the literature lacks a formal consensus on the minimum parameter scale for LLMs, as the model’s capacity is intertwined with both
data size and total compute [448]. In this paper, we adopt the LLM scope division and taxonomy
introduced by Pan et al.[326] and categorize the mainstream LLMs investigated in this study
into three groups according to their architectures: encoder-only, encoder-decoder, and decoderonly LLMs. This taxonomy and relevant models are shown in Fig. 4. We have included the LLMs
used by each work and their parameter sizes (if declared in the paper) in our public repository:
https://github.com/xinyi-hou/LLM4SE_SLR. Additionally, Table 6 summarizes the LLMs with different architectures suitable for different types of SE tasks.
Encoder-only LLMs. Encoder-only LLMs are a type of neural network architecture that utilizes
only the encoder component of the model [64]. The encoder’s function is to process and encode
the input sentence into a hidden representation, capturing the relationships between words and
the overall context of the sentence. Notable instances of encoder-only LLMs include BERT [64]
and its variants [92, 118, 211, 260]. As an example, BERT’s structure, based on the Transformer’s

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

Large Language Models for Software Engineering: A Systematic Literature Review
1:11

Table 6. Summary of LLMs with different architectures used in SE tasks.

Model
Type
Example of SE tasks
Encoder-only
Understanding
Code Understanding
Bug localization
Vulnerability detection
Encoder-Decoder
Understanding and Generation
Code summarization
Code translation
Program repair
Decoder-only
Generation
Code generation
Code completion
Test case generation

encoder architecture, has been referenced in 50 our selected primary studies. Its distinctive bidirectional attention mechanism simultaneously considers the left and right context of each word
during training. In the SE domain, other prominent models like CodeBERT [92], GraphCodeBERT [118], RoBERTa [260], and ALBERT [211] have been widely employed. Specialized models
such as BERTOverflow [415] and CodeRetriever [234] have been specifically developed for SE
applications. These models differ from BERT by leveraging program structure, introducing new
pre-training tasks, or engaging new modalities, thereby improving the architecture’s application to
code-related tasks. For example, CodeBERT integrates a token prediction scheme to comprehend
code by predicting subsequent tokens, enhancing its understanding of programming languages for
tasks like code completion and bug detection [92]. GraphCodeBERT introduces edge-type prediction, recognizing relationships between code elements as a graph. This enables GraphCoderBERT to
leverage code structure, improving its effectiveness in tasks like code summarization and program
analysis [118]. Encoder-only LLMs have shown efficacy in tasks requiring a nuanced understanding
of the entire sentence or code snippet. Examples include code review, bug report understanding,
and named entity recognition pertaining to code entities [19, 231, 297, 344, 380, 502].
Encoder-decoder LLMs. Encoder-decoder LLMs incorporate both encoder and decoder modules [436]. The encoder ingests the input sentence and encodes it into a hidden space, effectively
capturing the underlying structure and semantics. This hidden representation serves as an intermediary language, bridging the gap between diverse input and output formats. Conversely, the decoder
utilizes this hidden space to generate the target output text, translating the abstract representation
into concrete and contextually relevant expressions. Models such as PLBART [5], T5 [350], and
CodeT5 [464] embodies this architecture. Further advancements are evident in CodeT5+ [461],
while AlphaCode [237] and CoTexT [338] showcase the architecture’s adaptability to various SE
tasks. The encoder-decoder design offers flexible training strategies and is proficient in handling
multifaceted tasks such as summarization, translation, and question-answering. Within the field of
SE, this ability has been successfully applied to tasks like code summarization [9, 115, 287]. The
encoder module’s capacity to understand and represent both the structure and semantics of code
is pivotal, allowing the decoder to translate this comprehension into concise, human-readable
summaries.
Decoder-only LLMs. Decoder-only LLMs exclusively utilize the decoder module to generate the
target output text, following a distinct training paradigm that emphasizes sequential prediction [348].
Unlike the encoder-decoder architecture, where the encoder processes input text, the decoderonly architecture begins with an initial state and predicts subsequent tokens, gradually building
the output text. This approach relies heavily on the model’s ability to understand and anticipate

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

1:12
X Hou, Y Zhao, Y Liu, Z Yang, K Wang, L Li, X Luo, D Lo, J Grundy, and H Wang

language structure, syntax, and context. GPT-series models, such as GPT-1 [348], GPT-2 [349], GPT3 [29], GPT-3.5 [318], GPT-4 [320], as well as their notable derivative, ChatGPT [317]3, represent
their major implementations. More specialized versions like CodeGPT [268], InstructGPT [321],
Codex [43], Copilot [109]4, and others have been fine-tuned for specific tasks in SE. Open-source
models like GPT-J [444], GPT-Neo [28], GPT-NeoX [27], LLaMA [431], and Vicuna [51] also follow
this architecture. Decoder-only LLMs are usually more suitable for various generation tasks, such as
code generation and code completion. These models can generally perform downstream tasks from
a few examples or simple instructions without adding prediction heads or fine-tuning, making them
valuable tools in SE research. 2022 marked a surge in the development of decoder-only LLMs,
a trend that gained further momentum in 2023, notably with the launch of commercial
products by leading Internet companies. For example, Google launched Gemini [112], Meta
introduced LLaMA [431] and Llama 2 [432], and Anthropic unveiled Claude [18], etc. Contrary
to LLMs such as GPT-4 and its derivative application, ChatGPT, released by OpenAI, which were
promptly integrated into SE tasks, these new additions have not yet found widespread application
within the SE field. Their potential remains largely unexplored, with opportunities for further
assessment and utilization in specific tasks and challenges. The continued advancement of these
models emphasizes the active exploration and innovation within decoder-only architectures.

19

94

52

Encoder-only

8

9

24

2024
2023
2022
2021
2020

85

17

Encoder-decoder

2

0

77

432

73

Decoder-only

9

2

0
40
80
120
160
200
240
280
320
360
400
440
480

Number of instances utilizing an LLM in the collected papers

Fig. 5. Trends in the application of LLMs with different architectures in SE tasks over time.

3.2
Trend Analysis

As shown in Fig. 5, in the span from 2020 to 2024, the architecture of LLMs has witnessed notable
shifts in preference and application within SE tasks. The specific choices between decoder-only,
encoder-decoder, and encoder-only structures have shaped the direction of research and solutions
in the SE domain [478]. This analysis explores trends in the adoption of these architectures over
the years, reflecting the evolving dynamics of LLM for SE tasks.
Evolution of LLM architectures in 2021. The year 2020 saw research papers predominantly
concentrating on encoder-only LLMs for SE tasks, evidenced by a total of eight papers. Decoderonly LLMs or encoder-decoder LLMs were scarcely featured in that year’s research. A marked
change occurred in 2021. Out of 19 papers in 2021, nine were dedicated to decoder-only LLMs,

3ChatGPT is a conversational agent built upon the GPT architecture, with GPT-3.5 and GPT-4 being specific versions of the
architecture, each representing successive advancements.
4Copilot is an application built upon LLMs tailored for coding tasks. For convenience, all subsequent references in this
paper to LLMs and their applications, such as ChatGPT and Copilot, will collectively be referred to as LLMs.

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

Large Language Models for Software Engineering: A Systematic Literature Review
1:13

constituting 47.37% of the research. Additionally, two papers, or 10.53%, focused on encoder-decoder
LLMs. Encoder-only LLMs witnessed a slight decline, representing 42.1% of the field with eight
papers. This rapid transition can be linked to the generative capability of decoder-only LLMs.
Researchers [212, 369, 400] found that these models, e.g., GPT series, requiring minimal fine-tuning,
could produce not only syntactically correct but also functionally relevant code snippets. Their
proficiency in grasping the context of code quickly made them a preferred choice.
Diversity of LLM architectures in 2022. 2022 experienced a significant increase in diversity,
with more varied LLM architectures finding representation. Out of a total of 142 papers, 73 were
centered around decoder-only LLMs, comprising 51.41% of the studies. Encoder-decoder LLMs
made their presence known in 17 papers, accounting for 11.97%. Meanwhile, encoder-only LLMs
led the field slightly with 52 papers, capturing 36.62% of the research interest. This diverse distribution suggests an exploration phase where researchers were actively assessing and leveraging
different architectures to suit varied needs and challenges. The near-equal interest across different
architectures underscores the field’s richness, indicating that no single approach had become the
definitive choice.
Dominance of the decoder-only architecture in 2023. 2023 signaled a strong shift towards
decoder-only LLMs. An impressive 432 instances of utilizing decoder-only LLMs were recorded
across 195 unique papers, reflecting that a single paper might employ multiple such models. These
papers focusing on decoder-only LLMs constituted a significant 70.7% of the total research this year.
In comparison, encoder-decoder LLMs were the subject of 85 papers, contributing 13.91%, while
encoder-only LLMs appeared to stabilize, with 94 papers, representing 15.39% of the 2023 research
landscape. This trend signifies a shift in focus and resources toward exploring and harnessing the
decoder-only architecture as the primary approach in many current and future LLM4SE research
and applications.
Exploration of the LLM architecture in 2024. The initial trends in January 2024 showcase the
ongoing evolution of LLM architectures. Among the 120 papers examined, decoder-only LLMs continued to maintain a prominent position, with 77 papers dedicated to this architecture, constituting
64.17% of the research. Encoder-decoder LLMs appeared in 24 papers, representing 20% of the total,
while encoder-only LLMs were featured in 19 papers, making up 15.83%. Although there is a slight
decrease in the dominance of decoder-only architectures compared to the previous year, they still
hold a central role. The persistent exploration of encoder-decoder and encoder-only architectures
suggests an enduring interest in diverse configurations within the SE research community.
Criteria for LLM selection in SE tasks. The selection of an LLM for SE tasks should involve
careful consideration rather than arbitrary choice. Key factors guiding this selection encompass the
model’s proficiency in understanding the context of code, its ability to generate relevant content,
responsiveness to fine-tuning, and demonstrated performance on SE-specific benchmarks [224,
238, 491]. Given the stringent syntactical rules and functional requirements inherent to SE tasks,
models capable of seamlessly integrating these complex aspects were typically favored.
Task-specific fine-tuning. A notable trend is the customization of LLMs for precise SE tasks [160,
231, 535]. By fine-tuning models with datasets tailored to specific functions such as bug detection
or code review, researchers were able to achieve marked performance improvements [55, 204].
In conclusion, the evolution of LLMs for SE, transitioning from encoder-only to decoder-only
architectures, highlights the field’s vibrancy and adaptability. This shift has fundamentally altered
the approach to SE tasks, reflecting the ongoing innovation within the discipline.

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

1:14
X Hou, Y Zhao, Y Liu, Z Yang, K Wang, L Li, X Luo, D Lo, J Grundy, and H Wang

RQ1 - Summary

(1) There are more than 70 different LLMs used for SE tasks in our selected primary studies. Based
on the underlying architecture or principles of different LLMs, we classified the summarized
LLMs into three categories, i.e., decoder-only, encoder-decoder, and encoder-only LLMs.
(2) We observed that each LLM architecture serves a specific purpose in SE tasks, with encoderonly LLMs focusing on comprehensive understanding, encoder-decoder LLMs used for tasks
requiring understanding of input information followed by content generation, and decoder-only
LLMs being more suitable for generation tasks.
(3) We analyzed the trend of LLM usage for SE tasks. The most widely used LLMs are with
decoder-only architectures. There are over 45 LLMs in the decoder-only category and 195 papers
have researched the application of decoder-only LLMs to SE tasks.

4
RQ2: HOW ARE SE-RELATED DATASETS COLLECTED, PREPROCESSED, AND USED
IN LLMS?

Data plays a crucial role in the model training phase [413]. First, data is collected to obtain diversity
and richness to ensure that the model can cope with different scenarios and situations. Second, data
is classified to clarify the training objectives of the model and avoid confusion and misinformation.
The preprocessing of data is indispensable to clean and transform the data to improve its quality.
Finally, data is formatted into a structure suitable for model processing, allowing the LLM to learn
the data’s features and patterns effectively. We analyze the reported processes of data collection,
data classification, data preprocessing, and data representation in our selected primary studies on
LLM4SE.

4.1
How are the datasets for training LLMs sourced?

Data is an indispensable and critical factor in training LLMs, which determines the generalization
ability, effectiveness, and performance of the models [413]. Adequate, high-quality, and diverse
data is critical to allow models to fully learn features and patterns, optimize parameters, and ensure
reliability in validation and testing. We first investigate the methods used to obtain the dataset.
By analyzing the methods of data collection, we divided the data sources into four categories:
open-source datasets, collected datasets, constructed datasets, and industrial datasets. Open-source
datasets [38, 189, 449, 528] refer to publicly accessible collections of data that are often disseminated
through open-source platforms or repositories. For example, datasets like HumanEval [43], which
consists of 164 manually crafted Python problems, each accompanied by its respective unit tests.
The open-source nature of these datasets ensures their credibility and allows for community-driven
updates, making them a reliable resource for academic research. Collected datasets [149, 285, 380, 427]
are those that researchers compile directly from a multitude of sources, including but not limited
to, major websites, forums, blogs, and social media platforms. For instance, researchers [35, 373,
473, 502] often scrape data from Stack Overflow [323] threads or GitHub [108] issues comments to
create a dataset tailored to their specific research questions. Constructed datasets [83, 185, 201, 532]
are specialized datasets that researchers create by modifying or augmenting collected datasets to
better align with their specific research objectives. These modifications can be carried out through
manual or semi-automatic methods and may include the generation of domain-specific test sets,
annotated datasets, or synthetic data. For example, researchers often take a collected dataset of
code snippets and manually annotate them with bug types to create a constructed dataset for
studying automated program repair techniques [88, 173, 483]. Industrial datasets [11, 290, 462]
are those obtained from commercial or industrial entities and often contain proprietary business

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

Large Language Models for Software Engineering: A Systematic Literature Review
1:15

data, user behavior logs, and other sensitive information. These datasets are particularly valuable
for research that aims to address real-world business scenarios. However, the acquisition of such
datasets is often complicated by issues related to business confidentiality and data privacy. For
example, in a collaborative effort with China Merchants Bank (CMB), Wang et al. [462] were able to
access 21 projects from CMB’s repositories. Access to such data would likely require non-disclosure
agreements and other legal safeguards to protect business interests. Each of these dataset types
offers unique advantages and challenges, and the choice between them should be guided by the
specific requirements and constraints of the research project at hand.

300

235

240

180

120

84

49

60

Number of papers

6
0

Open-source

Collected

Constructed

Industrial

datasets

datasets

datasets

datasets

Fig. 6. The collection strategies of LLM-related datasets.

Fig. 6 shows the collection strategies of LLM-related datasets. As can be seen from the data in
the figure, 235 studies used open-source datasets for training LLMs. One of the main reasons
for using open-source datasets in LLM training is their authenticity and credibility. Open-source
datasets usually contain real-world data collected from various sources (such as relevant studies that
have been conducted), which makes them highly reliable and representative of real-world scenarios.
This helps LLMs learn from real examples to better understand real-world applications and improve
their performance. Second, since LLMs are a topic that has just recently emerged, a lack of suitable
training sets does exist. Therefore, researchers often collect data from sites such as Stack Overflow
and GitHub and build datasets to make the data more composite for SE tasks. Among the 395
papers we studied, we discovered that merely six studies utilized industrial datasets. This
suggests a potential misalignment between the properties of datasets used in academic research
and those encountered in real-world industrial contexts. This divergence underscores the need for
future research to investigate industrial datasets, thereby ensuring that LLMs are applicable and
robust across both academic and industrial scenarios.
Note that some papers use multiple datasets that span different categories, e.g., Xu et al. [493]
evaluated the performance of Codex, GPT-J, GPT-Neo, and other LLMs on SE tasks, and Mastropaolo
et al. [287] investigated the use of T5 in several code-related tasks such as fixing bugs and generating
code comments. For different LLMs or different SE tasks, researchers may use different training
datasets. On the other hand, some papers focus on exploring how existing LLMs (e.g., ChatGPT)
are used in SE tasks [475] and do not specify the dataset used for model training, as these LLMs
like ChatGPT often do not require users to prepare training data themselves for general usage
scenarios.

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

1:16
X Hou, Y Zhao, Y Liu, Z Yang, K Wang, L Li, X Luo, D Lo, J Grundy, and H Wang

Table 7. Data types of datasets involved in prior studies.

Category
Data type
Total
Text-based
Programming tasks/problems (42)
Prompts (33)
151
datasets
SO (i.e. Stack Overflow) posts (12)
Bug reports (11)
Requirements documentation (9)
APIs/API documentation (8)
Q&A pairs (6)
Vulnerability descriptions (4)
Reviews (4)
Logs (3)
Methods (3)
Project issues (3)
Code comments (2)
Theorems (2)
Buggy text (1)
Dockerfiles (1)
Outage descriptions (1)
Semantic merge conflicts (1)
Site text (1)
Software development tasks (1)
User intents (1)
Software specifications (1)
User reviews (1)
Code-based
Source code (60)
Bugs/Buggy code (16)
103
datasets
Vulnerable source code (8)
Patches (4)
Code changes (3)
Test suites/cases (3)
Bug-fix pairs (2)
Error code (2)
Error-fix pairs (1)
Flaky test cases (1)
Identifiers (1)
Labeled clone pairs (1)
Packages (1)
Graph-based
GUI Images (1)
1
datasets
Software
Code repository (9)
Android apps (3)
20
repository
Issues and commits (3)
Pull-requests (2)
-based datasets
Industrial projects (1)
Open-source projects (1)
Web applications (1)
Combined
Programming tasks and test suites/cases (17)
Source code and comments (12)
55
datasets
Programming tasks and solutions (8)
Source code and description (3)
Code-text pairs (2)
Souce code and API usage sequences (2)
Source code and test suites/cases (2)
Bug report and test suites/cases (1)
Buggy code and comments (1)
Buggy code and solutions (1)
Code files and summaries (1)
Binary code and related annotations (1)
Failing test code and error messages (1)
Source code and Q&A pairs (1)
Source code, methods, and logs (1)
Vulnerable code and description (1)

*See Appendix A for the full table including references.

4.2
What types of SE datasets have been used in existing LLM4SE studies?
Data types play a pivotal role in shaping the architecture and selection of LLMs, as they directly
influence the extraction of implicit features and subsequent model decisions[35, 106, 390, 504]. The
choice of data types can significantly impact the overall performance and generalization ability
of the LLMs. We examine and classify the types of SE datasets employed in LLM4SE studies. By
investigating the relationship between data types, model architectures, and performance, we seek
to shed light on the critical role of data types in the success of LLM4SE applications.
Data type categorization. We classified the data types of all datasets into five categories: codebased, text-based, graph-based, software repository-based, and combined data types. Table 7 describes the specific data included in the data types corresponding to the datasets we summarized
from the 395 studies. We can find that most of the studies used text-based datasets, accounting
for a total of 151. The dominance of text-based datasets in training LLMs for SE tasks highlights the
models’ exceptional natural language processing capabilities. These LLMs excel in understanding
and processing textual data, making them an ideal choice for tasks that involve code comprehension,
bug fixing, code generation, and other text-oriented SE challenges. Their ability to process and

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

Large Language Models for Software Engineering: A Systematic Literature Review
1:17

learn from vast amounts of text data enables them to provide powerful insights and solutions for
various SE applications.
The most prevalent type of data utilized in training LLMs for SE tasks is programming
tasks/problems with 42 instances observed among the surveyed papers. This dominance
can be attributed to the diverse and challenging nature of programming problems, which provide
LLMs with opportunities to generalize knowledge and skills across various SE challenges, fostering
a robust understanding of software concepts and enhancing performance across a wide range of
tasks, including code generation, code completion, and code summarization, etc. Prompts follow
closely behind programming tasks, with 33 instances observed in the surveyed papers, providing
task-specific guidance to LLMs, serving as cues or instructions for the models, and helping them
understand the context and requirements of SE tasks. This combination helps the models develop a
robust understanding of software concepts and perform well in a wide range of tasks. There are
also SO (i.e., Stack Overflow) posts (12), bug reports (11), etc., which are among the more numerous
data types in text-based datasets.
The predominance of source code (60) as the most abundant data type in code-based datasets can
be attributed to its fundamental role in SE. Source code serves as the foundation of any software
project, containing the logic and instructions that define the program’s behavior. Therefore, having
a large volume of source code data is crucial for training LLMs to understand the intricacies of
software development, enabling them to effectively generate, analyze, and comprehend code in
various SE tasks. There are also common data types, such as bugs/buggy code (16) and patches (4),
for program repair tasks. Additionally, vulnerable source code (8) is used for vulnerability detection
tasks. Graph-based datasets are used in some research studies for SE tasks, e.g., Kolthoff et al. [203]
used a dataset composed of screenshots from Google Play Android applications to construct a
graphical user interface (GUI) repository in their study on LLM for the rapid prototyping task.
These datasets represent code using graph structures, capturing relationships and dependencies
between code components.
Software repository-based datasets are compilations of data extracted from version control
systems, such as Git repositories, containing code, documentation, and related artifacts. This data
includes Code repository (3), issues and commits (3), and so on. The data in software repositories
can provide a wealth of information covering all aspects of the software development process,
including code evolution history, records of issue fixes and feature improvements, code quality
assessments, and so on. These data are valuable for studying behaviors and trends in the software
development process, improving software quality and development efficiency, and evaluating the
performance of software engineering techniques. Therefore, many studies have used software
repository-based datasets for empirical analysis and model training.
Some studies employed combined datasets containing multiple datatypes. Among them, the
most common type is “programming tasks and test suites/cases”. Other combinations of data
types include “source code and comments”, “programming tasks and solutions”, “source code and
description ”, “code-text pairs”, etc.

4.3
How do data types influence the selection of data-preprocessing techniques?

For the training and application of LLMs, the raw dataset needs to be subjected to data processing
to obtain a clean and suitable dataset for model training. The data processing steps [216, 279]
involve operations such as data cleaning, noise removal, normalization, etc. To ensure consistency
and quality of the data, different data types may require different processing methods to improve
the performance and effectiveness of LLMs in SE tasks. In this section, we aim to detail the data
preprocessing procedures for the two most used types of datasets, i.e., text-based datasets and
code-based datasets.

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

1:18
X Hou, Y Zhao, Y Liu, Z Yang, K Wang, L Li, X Luo, D Lo, J Grundy, and H Wang

Duplicated

Data
extraction

Initial data
segmentation

Unqualified
data deletion

Text 
preprocessing

Data 
tokenization

Data 
segmentation

instance 
deletion

Fig. 7. The data preprocessing procedure for text-based datasets.

Unqualified

Duplicated

Uncompilable

Data

Data

Code

Data

extraction

compilation

representation

segmentation

data 
deletion

instance 
deletion

data 
deletion

Fig. 8. The data preprocessing procedure for code-based datasets.

The data preprocessing procedure for text-based datasets. As displayed in Fig. 7, the steps
of text-based dataset preprocessing consist of seven steps in total, yet there are some differences
from the code-based dataset preprocessing steps. The process begins with data extraction [54,
55, 83, 504], where relevant text is carefully extracted from SE documentation from a variety of
sources, including bug reports [55], requirements documents [203], code comments [343], and
API documentation [190]. This step ensures that the dataset captures diverse, task-specific textual
information. After data extraction, the text is initially segmented and categorized according to the
specific requirements of the research task. For example, the text can be segmented into sentences
or further broken down into individual words as needed for analysis [129, 204]. To ensure the
quality and relevance of the dataset, substandard data deletion is performed to eliminate any invalid
or irrelevant text. For example, the dataset used by Lee et al. [216] was constructed from bug
reports, and in the “unqualified data deletion” process the researchers filtered out bug reports
with fewer than 15 words because the text was too short to contain contextual information.
Next, preprocessing operations are performed on the text to standardize and clean it. Common
preprocessing steps include removing certain symbols, stop words, and special characters [351, 462].
This standardized form of text facilitates the efficient processing of LLMs. To avoid introducing
bias and redundancy in the dataset, researchers eliminated duplicate instances by removing any
duplicate text samples [129, 204, 493]. This step enhances the diversity of the dataset and helps
the model to generalize better to new inputs. “Data tokenization” is a key step in preparing the
text for LLMs [271]. Text is labeled into smaller units, such as words or subwords, so that LLMs
are easier to manage and process efficiently. Finally, the preprocessed dataset is partitioned into
different subsets, usually including a training set, a validation set, and a test set.
The data preprocessing procedure for code-based datasets. We now summarize the process of
preprocessing a code-based dataset, which consists of seven steps. Fig. 8 describes the individual
data processing steps in detail and gives examples. The first step is data extraction, which involves
retrieving relevant code segments from different sources such as software repositories or version
control systems [183, 504]. Depending on the requirements of the research task [287, 522], code
segments can be extracted at different levels of granularity, ranging from individual methods and
functions to entire source code files or even complete software projects. The next step is to remove
any code segments that do not meet predefined criteria or quality standards [223, 343, 390]. This
filtering process ensures that the extracted code is relevant to the specific SE task under study,
thus eliminating incomplete or irrelevant code snippets. To avoid introducing bias and redundancy
during model training, the third step involves removing duplicate instances [56, 493, 560]. Any
duplicate code instances are identified and removed from the dataset, thus increasing the diversity
and uniqueness of the data. After the data extraction and filtering steps, the fourth step, data
compilation, comes into play. The extracted and filtered code segments are merged and compiled

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

Large Language Models for Software Engineering: A Systematic Literature Review
1:19

into a unified code dataset. This compilation process simplifies data storage and access and facilitates
subsequent analysis and model training [35, 283]. In the fifth step, the problem of invalid or
non-executable code is solved by removing data that cannot be compiled. Any code segments
that cannot be compiled or executed are removed from the dataset to ensure that the remaining
code instances are valid and usable during model training and evaluation. The sixth step is code
representation, which consists of converting the code segments into a suitable representation that
can be processed by the LLMs. This conversion can take different forms: token-based representation
involves tokenizing the source or binary code into distinct tokens; tree-based representation parses
the code into Abstract Syntax Trees (AST); and graph-based representation generates a Program
Dependence Graph (PDG), encompassing Control Flow Graphs (CFG) and Call Graphs (CG). Finally,
in the “data segmentation” step, the preprocessed dataset is partitioned into different subsets for
training, validation, and testing [56, 473]. The training set is used to train the LLM, the validation
set helps to tune the hyperparameters and optimize the model performance, and the testing set
evaluates the model’s ability on unseen data. By strictly adhering to these seven preprocessing
steps, researchers can create structured and standardized code-based datasets, thus facilitating the
effective application of LLMs for a variety of SE tasks such as code completion, error detection, and
code summarization.
It is worth emphasizing that the order of these steps is not fixed and can be adjusted based on the
specific research task and its associated requirements. Researchers need to carefully consider the
objectives, characteristics of the dataset, and the desired outcomes when determining the optimal
sequence for these preprocessing techniques.

4.4
What input formats are the datasets for LLM training converted to?
Once suitable datasets have been carefully chosen and clean data has been achieved through the
preprocessing steps, the next critical aspect is the transformation of the data into appropriate
formats that can effectively serve as inputs for LLMs. Table 8 shows four distinct data input types
that emerged during the research: Token-based input, Tree/Graph-based input, Pixel-based input,
and Hybrid-based input. We now detail each as follows:

Table 8. The various input forms of LLMs proposed in prior studies. See Appendix B for the full table including
references.

Category
Input forms
Total
Token-based input
Text in tokens (150)
Code in tokens (118)
347
Code and text in tokens (78)
Tree/Graph-based input
Code in tree structure (2)
Code in graph structure (3)
5
Pixel-based input
Pixel (1)
1
Hybrid-based input
Hybrid input forms (2)
2

Token-based input. Token-based input [7, 9, 19] involves representing code and text as sequences
of tokens, which are smaller units like words or subwords. Text in tokens refers to the tokenization
of textual data, such as documentation, bug reports, or requirements, enabling the LLMs to process
and analyze natural language descriptions effectively. Code and text in tokens combine both code
and its associated textual context, allowing the model to capture the relationships between code
elements and their descriptions. Code in tokens refers to the representation of code snippets broken
down into meaningful tokens, allowing the LLMs to understand programming language syntax
and semantics at a fine-grained level.
Tree/Graph-based input. Tree-based input [275, 315, 555] represents code as hierarchical tree
structures, capturing the syntactic relationships between code elements. Each node in the tree

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

1:20
X Hou, Y Zhao, Y Liu, Z Yang, K Wang, L Li, X Luo, D Lo, J Grundy, and H Wang

represents a code element, and the edges represent the hierarchical nesting of control flow statements and other code structures. This form of input allows the LLMs to understand the code’s
hierarchical structure and perform tasks like code completion and bug fixing. Graph-based input
represents code as a graph structure, where nodes represent code elements and edges represent the
relationships between them. Unlike trees, graphs allow more flexible and complex relationships
between code elements, enabling the model to capture non-linear dependencies in the code. This
form of input is used in tasks like code summarization and vulnerability detection by considering
the code’s intricate relationships.
Pixel-based input. Pixel-based input [301] visualizes code as images, where each pixel represents
a code element or token. This visual representation allows the LLMs to process and understand
code through image-based learning. In this input form, LLMs learn from the visual patterns and
structures in the code to perform tasks like code translation or generating code visualizations.
Hybrid-based input. Hybrid-based input [313] combines multiple modalities to provide LLMs
with diverse perspectives for better code comprehension. For example, a hybrid input may combine
code in tokens with visual representations of code, allowing the model to learn both from the finegrained details in the tokenized code and from the overall visual structure of the code. This approach
enhances the model’s ability to understand complex code patterns and improve performance in
tasks such as code comprehension and code generation.
During our investigation of LLM-based models for SE tasks, we observed distinct trends in the
usage of different input forms during the training process. Token-based input forms, namely
code in tokens and text in tokens were the most prevalent, collectively constituting
approximately 97.75% of the studies5. Specifically, code in tokens was widely adopted in 118
studies, accounting for approximately 33.24% of the total studies, demonstrating its popularity as a
primary choice for representing code snippets. This approach allowed LLMs to grasp programming
language syntax and semantics effectively, making it suitable for a wide range of code-related
tasks. Similarly, text in tokens was utilized in 150 studies, comprising around 42.25% of the total
studies. This input form allowed LLMs to process natural language descriptions, bug reports,
and documentation with greater efficiency and accuracy. The popularity of token-based input
forms underscores their significance in leveraging the power of LLMs for software engineering
applications.
In contrast, tree/graph-based input forms, such as code in tree-structure, were used in
only seven studies, making up approximately 1.4% of the total. Although less prevalent, this
input type emerged as a promising choice to represent the hierarchical structure and syntactic relationships within code. Its adoption indicated an ongoing exploration of tree-based representations
in specialized tasks, such as code completion and bug fixing.
Pixel-based input and hybrid-based input were relatively less common, each found in
one study, contributing approximately 0.28% of the total studies each. While their adoption
rates were lower, these input forms presented intriguing possibilities for specific applications.
Pixel-based input offered a unique visual representation of code, potentially advantageous for code
translation tasks. Meanwhile, hybrid-based input, combining multiple modalities (e.g., code in tree
structure and text in tokens in Niu et al.’s work [313]), showcased the potential for enhancing code
comprehension tasks by offering diverse perspectives for the models to learn from.
In summary, the trends in input form usage reveal a strong preference for token-based input,
demonstrating its versatility and effectiveness in various SE tasks. However, ongoing exploration
of other input forms, such as tree/graph-based, pixel-based, and hybrid-based, suggests a dynamic
and evolving landscape in the application of LLMs for SE, with potential for further innovation and

5This refers to studies that explicitly state input forms of LLMs, i.e., a total of 355 papers as shown in Table 8.

ACM Trans. Softw. Eng. Methodol., Vol. X, No. Y, Article 1. Publication date: December 2024.

---

*Source: arXiv:2308.10620*
