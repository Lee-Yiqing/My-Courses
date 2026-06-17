---
id: B4
title: "StarCoder 2 and The Stack v2: The Next Generation"
domain: B
year: 2024
arxiv_id: "2402.19173"
confidence: verified
source: "arXiv:2402.19173"
node_type: paper
---

# StarCoder 2 and The Stack v2: The Next Generation

**Domain**: [[domain_B|LLM Code Generation]] | **Year**: 2024 | **Confidence**: [x] verified


## Authors
[[author_Anton Lozhkov|Anton Lozhkov]], [[author_Raymond Li|Raymond Li]], [[author_Loubna Ben Allal|Loubna Ben Allal]], [[author_Federico Cassano|Federico Cassano]], [[author_Joel Lamy-Poirier|Joel Lamy-Poirier]], [[author_Nouamane Tazi|Nouamane Tazi]], [[author_Ao Tang|Ao Tang]], [[author_Dmytro Pykhtar|Dmytro Pykhtar]], [[author_Jiawei Liu|Jiawei Liu]], [[author_Yuxiang Wei|Yuxiang Wei]], [[author_Tianyang Liu|Tianyang Liu]], [[author_Max Tian|Max Tian]], [[author_Denis Kocetkov|Denis Kocetkov]], [[author_Arthur Zucker|Arthur Zucker]], [[author_Younes Belkada|Younes Belkada]], [[author_Zijian Wang|Zijian Wang]], [[author_Qian Liu|Qian Liu]], [[author_Dmitry Abulkhanov|Dmitry Abulkhanov]], [[author_Indraneil Paul|Indraneil Paul]], [[author_Zhuang Li|Zhuang Li]], [[author_Wen-Ding Li|Wen-Ding Li]], [[author_Megan Risdal|Megan Risdal]], [[author_Jia Li|Jia Li]], [[author_Jian Zhu|Jian Zhu]], [[author_Terry Yue Zhuo|Terry Yue Zhuo]], [[author_Evgenii Zheltonozhskii|Evgenii Zheltonozhskii]], [[author_Nii Osae Osae Dade|Nii Osae Osae Dade]], [[author_Wenhao Yu|Wenhao Yu]], [[author_Lucas Krauß|Lucas Krauß]], [[author_Naman Jain|Naman Jain]], [[author_Yixuan Su|Yixuan Su]], [[author_Xuanli He|Xuanli He]], [[author_Manan Dey|Manan Dey]], [[author_Edoardo Abati|Edoardo Abati]], [[author_Yekun Chai|Yekun Chai]], [[author_Niklas Muennighoff|Niklas Muennighoff]], [[author_Xiangru Tang|Xiangru Tang]], [[author_Muhtasham Oblokulov|Muhtasham Oblokulov]], [[author_Christopher Akiki|Christopher Akiki]], [[author_Marc Marone|Marc Marone]], [[author_Chenghao Mou|Chenghao Mou]], [[author_Mayank Mishra|Mayank Mishra]], [[author_Alex Gu|Alex Gu]], [[author_Binyuan Hui|Binyuan Hui]], [[author_Tri Dao|Tri Dao]], [[author_Armel Zebaze|Armel Zebaze]], [[author_Olivier Dehaene|Olivier Dehaene]], [[author_Nicolas Patry|Nicolas Patry]], [[author_Canwen Xu|Canwen Xu]], [[author_Julian McAuley|Julian McAuley]], [[author_Han Hu|Han Hu]], [[author_Torsten Scholak|Torsten Scholak]], [[author_Sebastien Paquet|Sebastien Paquet]], [[author_Jennifer Robinson|Jennifer Robinson]], [[author_Carolyn Jane Anderson|Carolyn Jane Anderson]], [[author_Nicolas Chapados|Nicolas Chapados]], [[author_Mostofa Patwary|Mostofa Patwary]], [[author_Nima Tajbakhsh|Nima Tajbakhsh]], [[author_Yacine Jernite|Yacine Jernite]], [[author_Carlos Muñoz Ferrandis|Carlos Muñoz Ferrandis]], [[author_Lingming Zhang|Lingming Zhang]], [[author_Sean Hughes|Sean Hughes]], [[author_Thomas Wolf|Thomas Wolf]], [[author_Arjun Guha|Arjun Guha]], [[author_Leandro von Werra|Leandro von Werra]], [[author_Harm de Vries|Harm de Vries]]


## Keywords
- [[kw_StarCoder2|StarCoder2]]
- [[kw_open LLM|open LLM]]
- [[kw_The Stack v2|The Stack v2]]
- [[kw_code generation|code generation]]
- [[kw_BigCode|BigCode]]
- [[kw_SE|SE]]
- [[kw_AI|AI]]

## Abstract

The BigCode project, an open-scientific collaboration focused on the responsible development of Large Language Models for Code (Code LLMs), introduces StarCoder2. In partnership with Software Heritage (SWH), we build The Stack v2 on top of the digital commons of their source code archive. Alongside the SWH repositories spanning 619 programming languages, we carefully select other high-quality data sources, such as GitHub pull requests, Kaggle notebooks, and code documentation. This results in a training set that is 4x larger than the first StarCoder dataset. We train StarCoder2 models with 3B, 7B, and 15B parameters on 3.3 to 4.3 trillion tokens and thoroughly evaluate them on a comprehensive set of Code LLM benchmarks. We find that our small model, StarCoder2-3B, outperforms other Code LLMs of similar size on most benchmarks, and also outperforms StarCoderBase-15B. Our large model, StarCoder2- 15B, significantly outperforms other models of comparable size. In addition, it matches or outperforms CodeLlama-34B, a model more than twice its size. Although DeepSeekCoder- 33B is the best-performing model at code completion for high-resource languages, we find that StarCoder2-15B outperforms it on math and code reasoning benchmarks, as well as several low-resource languages. We make the model weights available under an OpenRAIL license and ensure full transparency regarding the training data by releasing the SoftWare Heritage persistent IDentifiers (SWHIDs) of the source code data.

## Paper Content

Under review as submission to TMLR

StarCoder 2 and The Stack v2: The Next Generation

Anton Lozhkov1
Raymond Li2
Loubna Ben Allal1
Federico Cassano4
Joel Lamy-Poirier2

Nouamane Tazi1
Ao Tang3
Dmytro Pykhtar3
Jiawei Liu7
Yuxiang Wei7
Tianyang Liu25

Max Tian2
Denis Kocetkov2
Arthur Zucker1
Younes Belkada1
Zijian Wang5
Qian Liu12

Dmitry Abulkhanov5
Indraneil Paul32
Zhuang Li14
Wen-Ding Li26
Megan Risdal24
Jia
Li5
Jian Zhu16
Terry Yue Zhuo14,15
Evgenii Zheltonozhskii13
Nii Osae Osae Dade28

Wenhao Yu20
Lucas Krauß5
Naman Jain27
Yixuan Su30
Xuanli He23
Manan Dey31

Edoardo Abati5
Yekun Chai33
Niklas Muennighoff29
Xiangru Tang34
Muhtasham
Oblokulov18
Christopher Akiki9,10
Marc Marone8
Chenghao Mou5
Mayank Mishra19

Alex Gu17
Binyuan Hui5
Tri Dao21
Armel Zebaze1
Olivier Dehaene1
Nicolas Patry1

Canwen Xu25
Julian McAuley25
Han Hu14
Torsten Scholak2
Sebastien Paquet2
Jennifer
Robinson6
Carolyn Jane Anderson22
Nicolas Chapados2
Mostofa Patwary3
Nima
Tajbakhsh3
Yacine Jernite1
Carlos Muñoz Ferrandis1
Lingming Zhang7
Sean Hughes6

Thomas Wolf 1
Arjun Guha4,11
Leandro von Werra1,⋆
Harm de Vries2,⋆

1Hugging Face
2ServiceNow Research
3Nvidia
4Northeastern University
5Independent
6ServiceNow
7University of Illinois Urbana-Champaign
8Johns Hopkins University
9Leipzig University
10ScaDS.AI
11Roblox
12Sea AI Lab
13Technion – Israel Institute of Technology
14Monash University
15CSIRO’s
Data61
16University of British Columbia
17MIT
18Technical University of Munich
19IBM Research
20University of Notre Dame
21Princeton University
22Wellesley College
23University College London
24Kaggle
25UC San Diego
26Cornell University
27UC Berkeley
28Mazzuma
29Contextual AI
30Cohere
31Salesforce
32Technical University of Darmstadt
33Baidu
34Yale University

Corresponding authors (⋆) can be contacted at contact@bigcode-project.org

Abstract

arXiv:2402.19173v1  [cs.SE]  29 Feb 2024

The BigCode project,1 an open-scientific collaboration focused on the responsible development
of Large Language Models for Code (Code LLMs), introduces StarCoder2. In partnership
with Software Heritage (SWH),2 we build The Stack v2 on top of the digital commons of their
source code archive. Alongside the SWH repositories spanning 619 programming languages,
we carefully select other high-quality data sources, such as GitHub pull requests, Kaggle
notebooks, and code documentation. This results in a training set that is 4× larger than the
first StarCoder dataset. We train StarCoder2 models with 3B, 7B, and 15B parameters on
3.3 to 4.3 trillion tokens and thoroughly evaluate them on a comprehensive set of Code LLM
benchmarks.

We find that our small model, StarCoder2-3B, outperforms other Code LLMs of similar size
on most benchmarks, and also outperforms StarCoderBase-15B. Our large model, StarCoder215B, significantly outperforms other models of comparable size. In addition, it matches or
outperforms CodeLlama-34B, a model more than twice its size. Although DeepSeekCoder33B is the best-performing model at code completion for high-resource languages, we find
that StarCoder2-15B outperforms it on math and code reasoning benchmarks, as well as
several low-resource languages. We make the model weights available under an OpenRAIL
license and ensure full transparency regarding the training data by releasing the SoftWare
Heritage persistent IDentifiers (SWHIDs) of the source code data.

1https://www.bigcode-project.org
2https://www.softwareheritage.org/

1

Under review as submission to TMLR

1
Introduction

Large Language Models for Code (Code LLMs; Chen et al., 2021; Nijkamp et al., 2023; Rozière et al., 2023;
Guo et al., 2024) have rapidly emerged as powerful assistants for writing and editing code. As of January 30,
2024, GitHub CoPilot has garnered over 1.3 million paying subscribers, with over 50,000 organisations opting
for the enterprise version (MSFT Q2 Earning Call, 2024), estimated to increase developer productivity by up
to 56% as well as developer satisfaction (Peng et al., 2023; Ziegler et al., 2024). ServiceNow recently disclosed
that their “text-to-code” solution, built from fine-tuning StarCoderBase models (Li et al., 2023), results in
a 52% increase in developer productivity (Yahoo Finance, 2024). Despite the initial focus on generating
code snippets from natural language instructions or other code snippets, Code LLMs exhibit the potential
to enhance all phases of the software development cycle (Hou et al., 2023; Fan et al., 2023; Wang et al.,
2024; Zhuo et al., 2023b; Chai et al., 2023). This includes speeding up the implementation of new projects,
improving quality assurance for developed software, helping detect and fix bugs, simplifying maintenance
tasks, and easing migration to newer software.

The development process of LLMs can exhibit different levels of openness (Solaiman, 2023; Ding et al.,
2022; Akiki et al., 2022). Proprietary models like OpenAI’s GPT-4 (OpenAI et al., 2023) and Google’s
Gemini (Gemini Team et al., 2023) provide access to the model through a paid API but do not disclose
development details. On the other hand, open-weight models like Code LLaMa (Rozière et al., 2023),
Mistral (Jiang et al., 2023), and DeepSeekCoder (Guo et al., 2024) have released the model weights. This
enables the open-source community to run these models locally, inspect the model representations, and finetune them on their tasks. However, the model developers have not disclosed their training data. Consequently,
content creators do not know if their data was used for training, social scientists cannot scrutinize the dataset
for bias and toxicity, and LLM developers lack information as to what extent the training set is contaminated
with test benchmarks. More broadly, this practice hinders scientific progress as other research teams cannot
readily reuse each other’s training data. Other LLM development projects, like Allen AI’s OLMo (Groeneveld
et al., 2024), Eleuther AI’s Pythia (Biderman et al., 2023), and BigScience’s BLOOM (BigScience Workshop,
2022; Scao et al., 2022a), have adopted a fully open development approach by releasing training data, training
frameworks, and evaluation suites.

The BigCode project was established in September 2022 as an open scientific collaboration focused on the
open and responsible development of Code LLMs. BigCode is stewarded by ServiceNow and Hugging Face in
the spirit of open governance (BigCode collaboration et al., 2023) and has brought together more than 1,100
members from diverse academic institutes and industry labs. The community previously released The Stack
v1 (Kocetkov et al., 2023), a 6.4 TB dataset of permissively licensed source code in 384 programming languages.
The Stack v1 includes a governance tool called “Am I in The Stack,” designed for developers to verify if their
source code is included in the dataset. It also provides an opt-out process for those who prefer to exclude their
code from the dataset. In December 2022, the BigCode community released SantaCoder (Ben Allal et al.,
2023), a strong-performing 1.1B parameter model trained on Java, JavaScript, and Python code from The
Stack v1. Building upon this success, the community further scaled up its effort and released StarCoder on
May 4th, 2023 (Li et al., 2023). At its release, the 15B parameter StarCoder model was the best open-access
LLM for code.

This technical report describes the development process of The Stack v2 and StarCoder2. The Stack v2 builds
upon the foundation of Software Heritage’s vast source code archive, which spans over 600 programming
languages. In addition to code repositories, we curate other high-quality open data sources, including Github
issues, pull requests, Kaggle and Jupyter notebooks, code documentation, and other natural language datasets
related to math, coding, and reasoning. To prepare the data for training, we perform deduplication, create
filters to eliminate low-quality code, redact Personally Identifiable Information (PII), remove malicious code,
and handle opt-outs from developers who requested to have their code removed from the dataset. With this
new training set of 900B+ unique tokens, 4× larger than the first StarCoder dataset, we develop the next
generation of StarCoder models. We train Code LLMs with 3B, 7B, and 15B parameters using a two-stage
training process (Rozière et al., 2023; Guo et al., 2024). We start base model training with a 4k context
window and subsequently fine-tune the model with a 16k context window. We ensure that the training
process does not exceed more than 5 epochs over the dataset (Muennighoff et al., 2023). However, we push

2

Under review as submission to TMLR

the number of training tokens far beyond the compute-optimal number suggested by Chinchilla (Harm’s law;
de Vries, 2023) and train relatively small models within the range of 3.3 to 4.3 trillion tokens. We thoroughly
assess and compare the performance of these models on a suite of code LLM benchmarks (Cassano et al.,
2023b; Austin et al., 2021; Chen et al., 2021; Liu et al., 2023a; Lai et al., 2023; Muennighoff et al., 2024a;
Cassano et al., 2024; Liu et al., 2023b; Ding et al., 2023; Gu et al., 2024; Cobbe et al., 2021; Pearce et al.,
2022; Dhamala et al., 2021; Nozza et al., 2021; Gehman et al., 2020), finding that:

• The StarCoder2-3B model outperforms other Code LLMs of similar size (StableCode-3B and
DeepSeekCoder-1.3B) on most benchmarks. Moreover, it matches or surpasses the performance of
StarCoderBase-15B.

• The StarCoder2-15B model significantly outperforms other models of comparable size (CodeLlama13B), and matches or outperforms CodeLlama-34B. DeepSeekCoder-33B is the best model at
code completion benchmarks for high-resource languages. However, StarCoder2-15B matches or
outperforms DeepSeekCoder-33B on low-resource programming languages (e.g., D, Julia, Lua,
and Perl). Moreover, when we consider benchmarks that require models to reason about code
execution (Gu et al., 2024) or mathematics (Cobbe et al., 2021), we find that StarCoder2-15B
outperforms DeepSeekCoder-33B.

• The StarCoder2-7B model outperforms CodeLlama-7B but is behind DeepSeekCoder-6.7B. It is not
clear to this report’s authors why StarCoder2-7B does not perform as well as StarCoder2-3B and
StarCoder2-15B for their size.

2
Data Sources

In this section, we elaborate on the process of obtaining training data, encompassing not just the data
sourced from Software Heritage (§2.1) but also GitHub issues (§2.2), pull requests (§2.3), Jupyter and Kaggle
notebooks (§2.4), documentation (§2.5), intermediate representations (§2.6), small math and coding datasets
(§2.7), and other natural language datasets (§2.8).

2.1
Source Code

Software Heritage
We build the Stack v2 on top of the Software Heritage (SH) archive (Abramatic et al.,
2018), maintained by the non-profit organization of the same name. The mission of Software Heritage is to
collect and preserve all knowledge taking the form of source code. We work with the SH graph dataset (Pietri
et al., 2020), a fully deduplicated Merkle DAG (Merkle, 1987) representation of the full archive. The SH
graph dataset links together file identifiers, source code directories, and git commits, up to the entire states
of repositories, as observed during periodic crawls by Software Heritage.

Extracting repositories
We leverage the 2023-09-06 version of the SH graph dataset as the primary
source. We start by extracting the most recently crawled versions of all GitHub repositories and filtering
them to retain only the main branch. The branch is considered main if the repository metadata in GHArchive
lists it as the default branch or if its name is main or master. We only extract the latest revision (commit)
from the main branch and deduplicate the repositories based on the unique hashes of their contents (column
directory_id of the SH dataset). The repositories’ directory structure is reconstructed by recursively
joining the directory_entry table of the dataset to itself using the directory_id and target columns and
concatenating the directory and file names (column name) into full paths. We only traverse the directory tree
up to level 64. The individual file contents are downloaded from the SH content S3 bucket if the compressed
file size is less than 10MB.

License detection
We extract repository-level license information from GHArchive (Github Archive, 2024)
for all repositories with matching names in the SWH dataset. When the repo-level license is not available,
i.e., for 96.93% of repositories, we use the ScanCode Toolkit (ScanCode, 2024) to detect file-level licenses as
follows:

3

Under review as submission to TMLR

no

yes

Is the GitHub
license empty?

yes

no

Is the GitHub license permissive?

non-permissive
permissive

yes
no

Did ScanCode
detect licenses?

yes
no

no license
Are all detected licenses permissive?

permissive
non-permissive

Figure 1: File-level license assignment logic.

• Find all files that could contain a license using a regular expression in Appendix A.3. This allows us
to gather files that either explicitly contain a license (e.g., LICENSE, MIT.txt, Apache2.0) or contain
a reference to the license (e.g., README.md, GUIDELINES);

• Apply ScanCode’s license detection to the matching files and gather the SPDX3 IDs of the detected
licenses;

• Propagate the detected licenses to all files that have the same base path within the repository as the
license file.

Once the file-level license information is gathered, we decide whether the file is permissively licensed,
non-permissively licensed, or unlicensed, following the algorithm described in Figure 1.

The licenses we consider permissive are listed in Appendix A.4. This list was compiled from the licenses
approved by the Blue Oak Council (Blue Oak Council, 2024), as well as licenses categorized as “Permissive”
or “Public Domain” by ScanCode (ScanCode License Categories, 2024).

Data licenses
We consider three types of files: permissively licensed, non-permissively licensed (e.g.,
copyleft), and unlicensed files. The main difference between the Stack v2 and the Stack v1 is that we include
both permissively licensed and unlicensed files. We exclude commercial licenses since their creators do
not intend their code to be used for commercial purposes. We also exclude copyleft-licensed code due to
uncertainty regarding the community’s stance on using such data for LLM training and its relatively low
volume.

Language detection
While the Stack v1 (Kocetkov et al., 2023) detects programming languages by their
file extension, we instead rely on a language classifier. Specifically, we use go-enry based on GitHub’s library
linguist (go-enry, 2024) to detect the programming language for each file. We detect 658 unique languages
in TheStackV2-dedup, some of which get removed at the data inspection stage (see next paragraph).

3System Package Data Exchange, https://spdx.dev.

4

Under review as submission to TMLR

Table 1: A comparison of The Stack v1 and v2 on 32 popular programming languages. We show the size
and number of files for different data splits: The Stack v1 deduped, The Stack v2 deduped, and the training
data used for StarCoder2-15B.

The-stack-v1-dedup
The-stack-v2-dedup
The-stack-v2-swh-full
Language
Size (GB)
Files (M)
Size (GB)
Files (M)
Size (GB)
Files (M)

Assembly
1.58
0.25
13.02
0.77
7.74
0.70
Batchfile
0.29
0.25
2.11
1.13
1.02
0.99
C
57.43
8.53
202.05
20.78
114.92
19.18
C#
46.29
10.84
239.89
51.23
169.75
48.49
C++
50.89
6.37
353.89
43.18
211.33
42.23
CMake
0.45
0.19
2.58
1.74
2.27
1.70
CSS
22.61
2.99
161.68
23.87
8.00
1.88
Dockerfile
0.572
0.42
1.27
1.90
1.21
1.88
Fortran
0.17
1.84
4.66
0.27
3.61
0.26
Go
25.74
4.73
54.60
9.30
25.83
8.62
Haskell
2.36
0.54
5.11
1.25
4.17
1.23
HTML
146.76
9.53
2,419.87
90.23
99.09
5.23
Java
89.30
20.15
548.00
154.28
199.68
62.27
JavaScript
141.65
21.11
1,115.42
108.87
199.99
66.91
Julia
1.54
0.30
6.12
0.45
1.83
0.43
Lua
3.28
0.56
33.91
2.35
15.22
2.24
Makefile
1.49
0.66
21.30
4.22
5.19
2.78
Markdown
75.25
21.0
281.04
82.78
244.17
81.42
Perl
2.63
0.39
7.82
1.15
5.66
1.06
PHP
66.84
15.90
224.59
46.03
183.70
45.14
PowerShell
1.25
0.27
3.97
0.68
2.46
0.66
Python
64.30
12.96
233.29
56.93
191.61
56.19
R
0.30
0.04
22.39
5.15
19.05
4.29
Ruby
7.14
3.41
31.70
17.79
23.38
17.51
Rust
9.53
1.38
15.60
2.22
12.43
2.19
Scala
4.86
1.36
12.73
4.45
11.30
4.32
Shell
3.38
22.69
19.82
10.68
13.51
10.01
SQL
12.22
0.99
281.45
5.29
35.75
4.52
Swift
0
0
23.76
7.23
22.32
7.16
TeX
5.44
0.55
35.86
3.19
30.01
2.86
TypeScript
28.82
10.64
61.01
23.85
49.14
23.28
Visual Basic
1.49
0.16
16.63
1.06
7.48
0.81

Total
875.85
181.00
6,457.14
784.30
1,922.82
528.44

Visual data inspection
Similar to the first StarCoder, we involve the BigCode community in a data
inspection sprint to remove extensions with low-quality training data. We start from the annotations of the
previous iteration that eliminated 36 out of the 300 extensions (of the 86 included programming languages).
For StarCoder2, we only ran the data inspection for the not-yet-annotated programming languages (i.e.,
excluding the 86 languages of StarCoderBase). To streamline this process, we limited our inspection to
extensions that include over 1,000 files and represent over 0.5% of the files in their respective languages. The
remaining extensions were retained without further inspection, as they only make up a small volume. With
the help of 15 annotators from the BigCode community, we visually inspected around 1000 extensions and
excluded 130 (see appendix A.1 for the complete list). Our data inspection step excluded 39 programming
languages from the dataset (appendix A.2), resulting in a final count of 619 programming languages.

Basic filters
We apply a set of basic filters to the dataset to remove autogenerated files, data files, or other
low-quality training data.

5

Under review as submission to TMLR

• Long line filters: we first remove all files with more than 100k lines as those files are likely to be data
or generated code. We also remove files with an average line length of more than 100 characters or
a maximum line length of more than 1000 characters for all languages, excluding HTML, JSON,
Markdown, Roff, Roff Manpage, SMT, TeX, Text, and XML. For the mentioned languages, we
remove files where the longest line exceeds 100k characters.

• Autogenerated filter: we remove files classified as auto-generated by the is_generated function
of go-enry (go-enry, 2024). Additionally, we exclude files containing one of {“auto-generated”,
“autogenerated”, “automatically generated”, “generated automatically”, “this file is generated”} in
the first 5 lines of the file.

• Alpha filter: we remove files with less than 25% of alphabetic characters for all languages except
Motorola 68K Assembly and WebAssembly, where we only remove files with less than 25% of
alpha-numeric characters due to the syntax of those languages.

• Encoded data filter: we detect files with inline encoded data using the following regular expressions:

– Base64 strings: [a-zA-Z0-9+/\n=]{64,}
– Hexadecimal sequences: (?:\b(?:0x|\\x)?[0-9a-fA-F]{2}(?:,|\b\s*)){8,}
– Unicode strings: (?:\\u[0-9a-fA-F]{4}){8,}

We remove the file if any of the substrings matching these expressions is longer than 1024 characters
or if the fraction of matched characters is more than 50% of the file.

Language-specific filters
In addition to the basic filters, we apply the following set of language-specific
filters.

• For Text, JSON, YAML, Web Ontology Language, and Graphviz (DOT), we remove files with more
than 512 lines to minimize the impact of repeated tokens in data files.

• For HTML, we keep only the files where visible text is at least 100 characters long and makes up at
least 20% of the code, similar to the processing pipeline of StarCoder (Li et al., 2023).

• For Text, we keep only files with “requirement” in the lowercased filename, or if the filename without
the extension is one of {“readme”, “notes”, “todo”, “description”, “cmakelists”}.

2.2
Github Issues

We incorporate GitHub issues collected from GHArchive (Github Archive, 2024). We exclude pull requests
here as we process them separately in §2.3.

A Github issue consists of a series of events with actions, such as opening the issue, creating a comment, or
closing the issue. Each event includes the author’s username, a message, an action, and a creation date. We
follow the processing pipeline of StarCoder (Li et al., 2023), which we recap below:

• First, we removed auto-generated text when users replied to issues via email (for more information,
see Li et al., 2023, Appendix A). We also deleted issues with a short message (less than 200 characters)
and truncated long comments in the middle to a maximum of 100 lines while retaining the last
20 lines. This removed 17% of the volume — a similar percentage as in StarCoderBase.

• Next, we excluded comments from bots. To do so, we searched for keywords in the username of
the comment’s author (for more information, see Li et al., 2023, Appendix A). This step eliminated
3% of the issues, much less than the 17% reported in StarCoder (Li et al., 2023). This discrepancy
is primarily because our dataset does not include pull requests, which are often the source of a
significant proportion of bot-generated content.

6

Under review as submission to TMLR

• We used the number of users engaged in the conversation as an indicator of quality. Our criterion was
to include conversations that have two or more users. However, we also preserved conversations that
involved a single user if the total text within comments was less than 7,000 characters (96th percentile).
Additionally, we excluded issues authored by a single user if they contained more than ten events, as
they tended to be of poor quality or originate from overlooked bots. By implementing these filters,
we removed 38% of the remaining issues. Lastly, we anonymized the usernames in the conversations
by replacing them with a participant counter within the conversation (following the process of
StarCoder).

2.3
Pull Requests

We include code reviews by gathering pull request events from GHArchive (Github Archive, 2024) and the
corresponding source code from Software Heritage (Software Heritage, 2024b). Pull requests are requests to
merge particular code changes from one branch into another on GitHub. Typically, they involve multiple
rounds of code review discussions and additional cycles of code changes before they get merged into the
target branch.

Data collection
Specifically, for each pull request, we aggregate the PullRequestEvent, PullRequestReviewEvent, PullRequestReviewCommentEvent, IssueCommentEvent, and IssuesEvent events found on GHArchive.
More details about the differences between these events can be found in the Github documentation. Next,
we extract all base and head commit IDs from these events and retrieve the corresponding code files from
Software Heritage. As we do not have access to the commit diffs, we generate them by identifying changes
between files at the same path. We consider files present in the base but absent in the head as deletions, while
we consider files absent in the base but present in the head as additions. This process yields approximately
300M PRs, accompanied by a volume of 15 TB of base code. Among these, there are 215M closed PRs
originating from around 24M repositories.

PR filters
We remove PRs that 1) have been opened by bots, 2) consist only of comments by bots, 3) have
a non-permissive license, 4) have been opted out, 5) changes the base during the PR, 6) are not approved or
merged, or 7) lack initial diffs (either due to absent data from Software Heritage or because all data have
been filtered in other steps).

File filters
We remove files from the base commit if they satisfy one of the following conditions: 1) the
file is a deletion or addition, 2) the file length exceeds 1 million characters, 3) the fraction of alphanumeric
characters is less than 0.25, 4) the fraction of hexadecimal characters is greater than 0.25, 5) the max number
of lines surpasses 100,000, 6) the average line length exceeds 100, 7) the max line length surpasses 1,000, or
8) the presence of non-English text in Markdown

Title and description filtering
We apply the following heuristic filters to clean up the PRs further. We
exclude PRs with changes to the base, those not approved or merged, and those lacking initial diffs (either
due to absent data from Software Heritage or being filtered out in previous steps). We also exclude PRs
when the title is less than 10 characters or contains the words ’dependencies’, ’dependency’, ’depend’, or
’release’. We exclude PRs when the description is less than 20 characters or contains ’Qwiet’.

Truncating inputs
We shorten lengthy input fields in the PRs as follows. We truncate titles to 500
characters and descriptions to 80 lines, only displaying the first 60 and the last 20 lines. If the description
length still exceeds 1000 characters, we truncate it.

Processing comments
Following the processing of GitHub issues (§2.2), we remove comments from bots
and strip auto-generated text when users post via email reply. We anonymize the usernames of authors as
described in §3.2. We remove comments from PRs with less than 20 characters unless they are PR review
comments. For code review comments, we remove the full diff hunk if it exceeds 10,000 characters while
keeping the filename and comment.

7

Under review as submission to TMLR

Subsampling PRs
To increase the diversity in the PRs, we sub-sample them on a per-repository basis.
For repositories with 1 PR (after filtering), we retain it with a probability of 0.8. We linearly decrease this
retention probability to 0.1 for repositories with 1,000 PRs. For repositories with more than 1,000 PRs, we
set the retention probability such that we retain only 100 PRs. Finally, we sub-sample YAML and JSON files
with 10% retention probability when their file size exceeds 50% of the total base files size or when the file
path contains one of the keywords: ’pack’, ’lock’, ’yarn’, ’output’, ’swagger’, ’openapi’, or ’output’.

Max sequence length
We determine the maximum sequence length of PRs by first investigating the
data distribution after the processing steps mentioned above. We find 3.7M PRs with up to 1M characters,
resulting in 194 GB of data. This reduces to 3.3M PRs when we set a limit of 100K characters, resulting in a
dataset size of 67.3 GB. (appendix A.5 has more details about sequence length statistics.) For the StarCoder2
models, we opt to include PRs with up to 100K characters (translating to roughly 25k tokens). Since we
are pre-training with a limited context of 4K tokens, not all PRs fit into the context window. However, as
described in §5.2, we format the PRs so that the diffs are local and do not require long context.

2.4
Notebooks

We include notebooks from two separate sources: Jupyter notebooks extracted from the Software Heritage
archive and notebooks released by the Kaggle platform.

2.4.1
Jupyter Notebooks

We transform Jupyter Notebooks into scripts and structured notebooks following the same pipeline as
StarCoder (Li et al., 2023). One key difference is that we keep the markdown structure of the text blocks
while it is removed in StarCoder. For completeness, we recap these preprocessing steps below.

Jupyter – scripts
We utilize Jupytext4 to convert notebooks to scripts. To initiate the conversion process,
Jupytext requires the identification of the specific programming languages within each notebook. This
information is typically available in the metadata of most notebooks. In cases where it is not, we use the
Guesslang library5 to identify the programming language, using a probability threshold of 0.5 or higher. Our
initial dataset comprised 11 million notebooks, of which 3 million were excluded due to parsing errors. After
near-deduplication, the dataset was reduced to 4 million notebooks converted to scripts.

Jupyter – structured
To create this dataset, we first filtered out notebooks that did not contain any
Python code or Markdown text using the metadata information of each notebook. Only notebooks explicitly
marked as ‘Python’ in the metadata were kept. Then, for each notebook, consecutive Markdown blocks
or code blocks were merged into a single Markdown or code block, respectively. Eventually, we ended up
with consecutive code-text pairs in temporal order grouped by each notebook. Each Jupyter code-text pair
contained the Markdown text immediately preceding the code block and the Python code, forming a natural
instruction pair. We also included the formatted output of a code block if the output cell was non-empty;
otherwise, it was marked by a special <empty_output> token. If consecutive code blocks have multiple output
cells before merging, we only retain the output of the last code block. After these preprocessing steps and
near-deduplication, we ended up with 4.6M structured Jupyter notebooks.

2.4.2
Kaggle Notebooks

We include Python notebooks released by the Kaggle platform6 under an Apache 2.0 license, starting with an
initial dataset of 3.6M notebooks. Note that this Kaggle dataset does not include the output cells, only the
markdown and code cells.

Cleaning
We start the data cleaning process by dropping notebooks with less than 100 characters and
those with syntax errors. We also remove the templated text at the beginning of notebooks (see appendix A.7

4https://jupytext.readthedocs.io/
5https://guesslang.readthedocs.io/
6https://www.kaggle.com/datasets/kaggle/meta-kaggle-code

8

Under review as submission to TMLR

for the templates). These steps remove 18% of the notebooks. Next, we convert the notebooks to the
structured and script format, following the processing of the Jupyter notebooks in §2.4.1. Finally, we remove
near-duplicates using the pipeline described in §3.1, eliminating 78% of the notebooks and leaving us with
580k notebooks.

Dataset description
To provide the model with more context regarding the content and objectives of the
notebook, we include metadata about the Kaggle dataset whenever this information is available. We find
that 42% of the notebooks are associated with a Kaggle dataset and include its title and description at the
beginning of each notebook.

Dataset schema
In addition to these high-level dataset descriptions, we scanned the code inside the
notebooks for instances of read_csv. We found that 25% of the samples were loading CSV datasets. We
extracted and incorporated detailed information about these datasets as follows. First, we used the Kaggle
API to download the datasets and successfully retrieved 8.6% of the notebooks.
The remaining cases
were attributed to either the dataset being unavailable or encountering challenges downloading it within a
reasonable time frame. For the downloaded datasets, we prefix the output of df.info() to the notebook,
which displays the column names and their dtypes, the non-null values count, and the memory usage. We
also include four sample rows from the dataset.

2.5
Documentation

Documentation from package managers
We crawl documentation from several package manager
platforms, including npm, PyPI, Go Packages, Packagist, Rubygems, Cargo, CocoaPods, Bower, CPAN,
Clojars, Conda, Hex and Julia. We first retrieve the names of the most popular libraries across various
platforms from libraries.io. These library names are then used to search through individual package managers,
enabling us to obtain the respective homepages for each library. We systematically crawled the documentation
files from the obtained homepage links or, alternatively, extracted information from the provided README
or documentation files on the platform. For documents obtained through homepage links, we adhere to the
same processing strategy outlined below in the paragraph titled “Documentation from websites”. When
extracting documents from the REwang2023softwareADME or documentation files on the platform, we
employ distinct heuristics to extract the text using markdown formats whenever feasible, aiming to maintain
a simple and effective format. It is worth noting that many libraries available on PyPI and Conda have their
associated documentation hosted on Read the Docs, which typically offers more comprehensive documentation.
Consequently, we prioritize utilizing Read the Docs as the primary source of documentation for these libraries.
For these documents hosted on Read the Docs, we follow the same processing procedure outlined in the
paragraph titled “Documentation from websites”.

PDFs from package managers
For documents related to the R language, we extracted text from all
PDF files hosted on CRAN using the pdftotext library.7 This library is particularly effective in preserving
the formatting, including spaces within code snippets. For LaTeX-related documentation, we extracted the
documentation, tutorial, and usage guide PDFs of LaTeX packages from CTAN, filtered out image-heavy
PDFs, and converted the rest into markdown using the Nougat neural OCR tool.

Documentation from websites
We collect code documentation from a carefully curated list of websites
as detailed in Table 2. We start by systematically exploring the website from its initial URL listed in Table 2,
using a queue to store URLs within the same domain. This queue expands dynamically as we discover new
links during the crawl. Given that most documents comprise HTML pages, we focus our processing pipeline
on (1) content extraction and (2) content concatenation. To extract the content, we utilize the trafilatura
library8 to convert each HTML page into XML format, simultaneously eliminating redundant navigation and
index bars, elements that often recur in documentation. Next, we converted the XML format to markdown
using our XML-to-Markdown conversion script. In the second stage, to compile these documents into a
single text, we first do a near-deduplication of the content extracted from different HTML pages. This

7https://github.com/jalan/pdftotext
8https://github.com/adbar/trafilatura

9

Under review as submission to TMLR

Programming Language Usage

R

Go

Rust
JavaScript

Erlang
Unknown

TeX
Ruby
Python

YAML
Markdown

SQL
Objective-C

Programming Languages

Perl
PHP
Julia
JSON

HTML

CSS
Haskell

102
103
104

Number of Occurrences

Figure 2: The distribution of the top 20 programming languages in our crawled documentation collection.

step was essential since we have observed that certain document pages only comprise website layouts (e.g.,
navigation bars) instead of fruitful information for documents, resulting in a substantial amount of duplicated
content. To accomplish this, we treat each HTML page from a single website as a cluster and apply the
minhash locality-sensitive hashing technique to identify and eliminate similar pages, using a threshold of 0.7.
Finally, we assemble the gathered content from different pages of the same website in the order of web page
crawling, ensuring a cohesive narrative. This parallels the “breadth-first search” approach, where all nodes at
the current depth are explored before proceeding to the next depth level. Also, we collected code-relevant
data from existing web crawls such as RefinedWeb (Penedo et al., 2023), OSCAR (Ortiz Suárez et al.,
2019), and esCorpius (Gutiérrez-Fandiño et al., 2022). We use regular expressions to identify programming
language-specific constructs within the documents and to detect the “docs.” substring in the page URLs.
The resulting dataset primarily comprises content sourced from programming blogs, coding tutorials, and
platforms like Read the Docs, with the exclusion of the documents gathered above.

Free textbooks
We scraped free programming books compiled in the Free Programming Books project,
which aims at promoting the distribution of free programming e-books. First, we extract all links and identify
those with a PDF extension. Subsequently, we downloaded all available PDF files and utilized the pdf2text
library to extract text from these PDF files. Finally, we parsed 3,541 books whose languages span across
different regions, including English, Chinese, Japanese, Spanish, and others.

Language identification
Finally, we have employed a dual approach to identify the main programming
language used by each document. We leverage predefined rules when the source of the document unequivocally
corresponds to a specific programming language and resort to the guesslang9 library in cases where such
correspondence is not explicit. The resultant programming language distribution is graphically represented in
Figure 2.

2.6
Intermediate Representations

We augment source code by pairing its intermediate representations (IR) to enhance the model’s understanding
of low-resource programming languages. The key rationale behind this approach is that a shared intermediate

9https://github.com/yoeo/guesslang

10

Under review as submission to TMLR

Table 2: The websites scraped for the code documentation dataset.

Website Name
URL

DevDocs API Documentation
https://devdocs.io
MDN Web Docs
https://developer.mozilla.org
TensorFlow Docs
https://www.tensorflow.org
Linux Docs
https://www.kernel.org/doc/Documentation
Swift Programming Language
https://docs.swift.org/swift-book/documentation/the-swift-programming-language
Flutter API Reference
https://api.flutter.dev
TypeScript
https://www.typescriptlang.org/docs/handbook
Json.NET Documentation
https://www.newtonsoft.com/json/help/html
NVIDIA Documentation Hub
https://docs.nvidia.com
Oracle Java Tutorial
https://docs.oracle.com/javase/tutorial/java
Qiskit Documentation
https://qiskit.org/documentation
Q# Quantum Programming
https://learn.microsoft.com/en-us/azure/quantum/user-guide
Pony Tutorial
https://tutorial.ponylang.io
Zephir Documentation
https://docs.zephir-lang.com/0.12/en/introduction
Qemu Documentation
https://www.qemu.org/documentation
C# Documentation
https://learn.microsoft.com/en-us/dotnet/csharp
Hugging Face Documentation
https://huggingface.co/docs
LLVM Doc
https://llvm.org/docs
GCC Online Documentation
https://gcc.gnu.org/onlinedocs
Matlab Documentation
https://www.mathworks.com/help/matlab
Boost C++ Libraries
https://www.boost.org/doc
Maxima Manual
https://maxima.sourceforge.io/docs/manual/maxima_singlepage.html
Qt Documentation
https://doc.qt.io

representation might help to anchor low-resource constructs to similar ones in high-resource languages (Zhuo
et al., 2023b).

LLVM
We select LLVM (Lattner & Adve, 2004) as the intermediate representation due to its widespread
availability on GitHub, increasing the probability that there is sufficient training data to learn the semantics
of the language. In addition, LLVM is widely adopted as an IR and is the target representation of many
compiler frontends across several programming languages.10

Data collection
Existing attempts to extract IR from free-form source code either suffer from low
compilation success rates (Szafraniec et al., 2023) or use bespoke language-specific mechanisms to track
dependency code to compile successfully (Grossman et al., 2023). We sidestep this by sourcing self-contained
compilation units from accepted solutions to programming word problems (Rosetta Code, 2023; Mirzayanov,
2020; Puri et al., 2021; Caballero et al., 2016). We compile ≈4M sources in total across C++, C, Objective-C,
Python, Rust, Go, Haskell, D, Fortran, Swift, and Nim in size optimized (-OZ equivalent) and performance
optimized (-O3 equivalent) mode. We opt to use the size-optimized IR in most of the pairs due to context
length considerations. However, for 20% of the pairs, we use the performance-optimized IR. This is done to
maximize transfer from the pre-training stage, where the model sees LLVM code in the wild, which is more
likely to be in this form. We use clang11 for compiling C++, C and Objective-C, codon12 for compiling
Python, rustc13 for compiling Rust, gollvm14 for compiling Go, ghc15 for compiling Haskell, ldc16 for
compiling D, flang17 for compiling Fortran, and nlvm18 for compiling Nim. We clean headers along with
superfluous platform, vendor, and memory layout-specific information from the IR before pairing it with its
source.

10https://llvm.org/ProjectsWithLLVM/
11https://clang.llvm.org/
12https://docs.exaloop.io/codon
13https://www.rust-lang.org/
14https://go.googlesource.com/gollvm/
15https://www.haskell.org/ghc/
16https://wiki.dlang.org/LDC
17https://flang.llvm.org/docs/
18https://github.com/arnetheduck/nlvm

11

Under review as submission to TMLR

2.7
LHQ19

We include several small high-quality datasets for math and coding:

• APPS (train) (Hendrycks et al., 2021) is a popular text2code benchmark in Python with a train
set of 5,000 examples. We include one solution per programming problem.

• Code Contest (Li et al., 2022) is similar to APPS but includes solutions in several programming
languages, namely Python 2/3, C++, and Java. We include one solution per problem and language
and arrive at a dataset of 13k+ examples.

• GSM8K (train) (Cobbe et al., 2021) is the train split of GSM8K, a popular evaluation benchmark
for testing the math reasoning capabilities of LLMs. The dataset consists of 7k+ examples.

• GSM8K (SciRel) (Yuan et al., 2023) is an augmented version of GSM8K that includes alternative
reasoning paths for the questions in GSM8K. The extended version contains 110k examples.

• Deepmind Mathematics (Saxton et al., 2019) is a synthetic dataset of math questions and
answers across various domains (algebra, arithmetic, calculus, comparison, measurement, numbers,
polynomials, probability) and varying difficulty (easy-medium-hard). The dataset consists of 110M+
(short) examples.

• Rosetta Code (Rosetta Code, 2023; Nanz & Furia, 2015) is a dataset with over 1100 everyday
programming tasks with solutions in as many different programming languages as possible.

• MultiPL-T (Cassano et al., 2023a) is high-quality data in Lua, Racket, and OCaml based on
automatically translating extracted Python functions and validating them with unit tests. The total
dataset comprises over 200k examples.

• Proofsteps is part of the AlgebraicStack (Azerbayev et al., 2024), a dataset used to train the Lemma
family of models. We also include proofsteps-lean, which was extracted from mathlib 4 (mathlib
Community, 2020), and proofsteps-isabelle, which was built on top of the PISA dataset (Jiang
et al., 2021). Proofsteps-lean contains over 3k examples, while proofsteps-isabelle contains over 250k
examples.

2.8
Other Natural Language Datasets

StackOverflow
We include 11 million questions and their corresponding multiple responses from the Stack
Overflow dump dated 2023-09-14 (StackExchange Archive, 2024). We filtered out questions with fewer than
three answers. Upon inspecting the dataset, we found many mismatches between questions and answers
due to inherent format errors in the Stack Overflow dump. We leveraged Llama-2-70b-chat-hf (Touvron
et al., 2023) to increase the quality of the dataset as follows. We selected 20,000 examples and asked
Llama-2-70b-chat-hf to rate the question-answer pairs. See Appendix A.6 for the exact prompt. Next,
we pick the 10,000 highest-scoring pairs as positive examples and use the remaining 10,000 answers to
create negative examples by randomly pairing them with other questions. We use this dataset to train a
binary classifier by embedding the question and answer with a well-performing sentence embedding model
(sentence-transformers/all-MiniLM-L12-v220 (Reimers & Gurevych, 2019; Muennighoff et al., 2022a))
and minimizing the cosine distance between them. Next, we plot the embedding scores for a subset of the
question-answer pairs and manually determine the threshold to 0.1. As a question can have multiple answers,
we average the scores of question-answer pairs and remove all questions with an average score below 0.1. We
end up with 11.4 million questions and over 10B tokens.

ArXiv
We include the ArXiv subset of the RedPajama dataset (Together Computer, 2023). This dataset is
downloaded from the publicly available Amazon S3 bucket (Arxiv, 2024). We further processed the dataset
only to retain latex source files and remove preambles, comments, macros, and bibliographies from these files.
The final dataset is roughly 30B tokens.

19Leandro’s High-Quality dataset
20https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2

12

Under review as submission to TMLR

Wikipedia
We include the English subset of Wikipedia. Specifically, we use the version collected by
RedPajama (RedPajama Wiki, 2024), which is derived from the 2023-03-20 dump. We follow RedPajama’s
processing steps and eliminate hyperlinks and templates from the Wikipedia pages. The full dataset comprises
around 6 billion tokens.

OpenWebMath
We include OpenWebMath (Paster et al., 2023), an open dataset of high-quality mathematical text extracted from CommonCrawl. The full dataset comprises almost 15B tokens.

3
Preprocessing Pipeline

We apply several preprocessing steps, such as deduplication (§3.1), PII redaction (§3.2), benchmark decontamination (§3.3), malware removal (§3.4), and opt-out deletion requests (§3.5), to the data sources described
in the previous section. Since not all steps are applied to each data source, we summarize the preprocessing
pipeline per data source in Table 3.

3.1
Removing Near-Duplicates

We deduplicate the source code, pull requests, notebooks, issues, and documentation. We do not deduplicate
the already preprocessed natural language datasets, such as Arxiv, StackExchange, OpenWebMath, Wikipedia,
and the small high-quality math and reasoning datasets.

We followed the deduplication pipeline of SantaCoder (Ben Allal et al., 2023). This process first calculates
the MinHashes (Broder, 2000) of all code files and then utilizes Locally Sensitive Hashing (LSH) to group
files based on their MinHash fingerprints. During the LSH stage, “similar” files are assigned to the same
buckets, identifying them as duplicates. Only one file from each duplicate group is chosen. In addition to the
SantaCoder approach, to preserve repository context, we prioritize files from repositories with higher star
and fork counts or from the latest commit date as a tiebreaker. We used 5-grams and a Jaccard similarity of
0.7. We refer to this blogpost for more background information regarding the deduplication pipeline.

3.2
PII Redaction

To reduce the likelihood of re-distributing Personally Identifiable Information (PII) present in the training data,
we make diligent efforts to redact PII from the training set. We largely follow the steps from StarCoder (Li
et al., 2023) and leverage the StarPII model to redact various PII entities. Below, we provide more details on
how we apply it to each data source.

Redacting PII entities
We use StarPII to redact names, emails, keys, passwords, IP addresses, and
usernames from source code, pull requests, issues, and StackOverflow. We do not make any modifications
to the model or redaction logic described in the StarCoder paper (Li et al., 2023). For OpenWebMath and
documentation, we only redact names, keys, and emails, while we only redact emails for arXiv using the regex
described in Ben Allal et al. (2023).

Redacting usernames
The conversations in issues, pull requests, and StackOverflow often contain
usernames in the message thread.
We anonymize the author usernames by substituting them with a
participant counter specific to the conversation, like username_1 to represent the second participant. These
pseudonyms are added at the start of each comment to maintain the speaker’s identity. Moreover, any
references to these usernames in the messages are removed. Only the usernames of actively participating
individuals in the conversation are masked, and mentions of non-participating users remain unaffected.

3.3
Decontamination

To ensure the performance of StarCoder is not artificially inflated on our test benchmarks, we decontaminate
the training set from our test sets. Specifically, we remove files that contain docstrings or solutions from
HumanEval and MBPP, docstrings from APPS, questions from GSM8K, or prompts from DS1000. In contrast

13

Under review as submission to TMLR

Table 3: Overview of the data processing steps applied to each data source.

Dataset
Dedup
Malicious Code
Decontaminate
Opt-out
PII
Source Code
Yes
Yes
Yes
Yes
StarPII
Pull Requests
Yes
Yes
Yes
Yes
StarPII + Usernames
Jupyter/Kaggle Notebooks
Yes
Yes
Yes
Yes/No
StarPII
Issues
Yes
Yes
Yes
Yes
StarPII + Usernames
Docs
Yes
No
No
No
StarPII: Names, Keys, Emails
LHQ
No
No
No
No
No
Arxiv
No
No
No
No
Email
OpenWebMath
No
No
Yes
No
StarPII: Names, Keys, Emails
Wikipedia
No
No
No
No
No
StackExchange
No
No
Yes
No
StarPII + Usernames

to the first iteration of StarCoder (Li et al., 2023), we further enhance the recall of the decontamination
process by removing whitespace during string matching. Note that we exclude docs, LHQ, arXiv, and
Wikipedia from this decontamination step.

3.4
Malware Removal

We scan our training set to identify possible instances of malware in the source code, pull requests, notebooks,
and issues. To this end, we use ClamAV 1.2 (ClamAV, 2024) with additional unofficial malware signatures
published by SaneSecurity (Sane Security, 2024) as of 2023-11-16. Signatures with a high risk of False
Positives (as determined by SaneSecurity) were not used. See Table 26 for the most frequently detected
malware signatures in the unfiltered code dataset. In summary, this step eliminates 59,442 files from the
dataset, constituting only 0.009% of the 654M files.

3.5
Removing Opt-outs

We announced the upcoming training run of StarCoder2 on X21 and updated the "Am I in the stack"
governance tool with the new repositories from The Stack v2. Developers were granted until November 20,
2023, to submit their opt-out requests. After the cut-off date, we eliminated 1,561 repositories associated
with 91 users and organizations. A total of 22,066 files were removed from the source code dataset (excluding
issues and PRs).

4
Data Composition

Model capacity
With a much larger training set available, we decided to tailor our data composition to
each model size. We reason that smaller models, having limited capacity, should be exposed to a less diverse
dataset. This intuition is supported by research in multi-lingual NLP showing that languages compete for
model capacity (Arivazhagan et al., 2019; Conneau et al., 2020; Scao et al., 2022b). Hence, we first create a
smaller version of the SWH code dataset, selecting a subset of 17 widely-used programming languages. We
use this variant to train the 3B and 7B models, whereas we use the full version with all 619 programming
languages for the 15B model. To further limit the diversity in the training set for the 3B model, we also
exclude some natural language datasets (see “Data composition per model size”).

Downsampling languages
Similar to StarCoderBase, we adhere to the natural distribution of the data as
much as possible. Before constructing the source code datasets, we examined the data distribution among
the programming languages. Compared to StarCoderBase, we found slightly larger variations among the
high-resource languages. The observed data volume (in GB) is as follows: Java (479.68), JavaScript (277.25),
C++ (204.49), Python (190.99), PHP (171.57), C# (166.22), and C (114.49). We decided to downsample both
Java and Javascript to 200GB to put these high-resource languages on a more equal footing. Furthermore, we

21https://x.com/BigCodeProject/status/1721583097580249254?s=20

14

Under review as submission to TMLR

Table 4: Overview of the data composition of StarCoder2 models. We refer to the training set of the 3B
model as the-stack-v2-train-3B.

Dataset
Tokens (B)
3B
7B
15B

the-stack-v2-train-smol
525.5
✓
✓
✗
the-stack-v2-train-full
775.48
✗
✗
✓

Pull requests
19.54
✓
✓
✓

the-stack-v2-train-extras

Issues
11.06
✓
✓
✓
Jupyter structured
14.74
✓
✓
✓
Jupyter scripts
16.29
✓
✓
✓
Kaggle scripts
1.68
✓
✓
✓
Documentation
1.6
✓
✓
✓
OpenWebMath
14.42
✗
✓
✓
Wikipedia
6.12
✗
✓
✓
StackOverflow
10.26
✓
✓
✓
Arxiv
30.26
✗
✓
✓
LHQ
5.78
✓
✓
✓
Intermediate Repr.
6
✓
✓
✓

Unique tokens (B)
622.09
658.58
913.23

preserved 254GB of markdown data while reducing the size of HTML to 100 GB. This decision was driven by
the anticipation that markdown would likely contain more code documentation, whereas HTML is commonly
associated with webpages. Lastly, we subsampled data files like JSON, XML, and YAML to 8GB and a few
other data formats to 1 GB. See Table 28 in Appendix C.2 for the full list of subsampled languages.

Repository-context
After subsampling some programming languages, we compile the source code from
Software Heritage into repository-context-aware datasets. Each example in the dataset is a full repository
with files arranged in a random order. As previously noted, we create two versions of the SWH dataset,
the-stack-v2-train-smol and the-stack-v2-train-full, as further detailed in the subsequent paragraphs.

The-stack-v2-train-smol
For the small variant, we select 17 widely used programming languages and
include a curated set of documentation and configuration languages.

– Rust
– SQL
– Shell
– Swift
– TypeScript

• Specifically, we include the following programming languages:
– C
– C#
– C++
– Go
– Java
– JavaScript

– Kotlin
– Lua
– PHP
– Python
– R
– Ruby

– RDoc
– RMarkdown

– Text
– reStructuredText

• And incorporate the following languages associated with code documentation:
– AsciiDoc
– HTML
– Markdown

• We also include several configuration languages and files, which we list in Appendix C.1.

• Despite limiting the languages to this subset, we obtain a dataset of 525B+ unique tokens.

The-stack-v2-train-full
For the full variant, we include all 619 programming languages. Although this
subset significantly enhances language diversity (adding 600+ programming languages), it contributes only
around 250B tokens to the dataset, culminating in 775B+ tokens.

15

Under review as submission to TMLR

Data composition per model size
In Table 4, we summarize the data composition for the 3B, 7B,
and 15B models. We use the-stack-v2-train-extras to denote all supplementary sources gathered for
StarCoder2, excluding the source code obtained from SWH. For the 3B, we use the-stack-v2-train-smol
and exclude OpenWebMath, Wikipedia, and Arxiv from the extra data sources in §2. This leads to a dataset
of 622B+ unique tokens. For the 7B, we include OpenWebMath, Wikipedia, and Arxiv, leading to a slightly
larger dataset of 658B+ unique tokens. For the 15B, we include the-stack-v2-train-full dataset and all
extra data sources listed in §2, resulting in a dataset with 913B+ unique tokens. The size of this dataset is
4× the size of the training dataset for StarCoderBase.

5
Data Formatting

We present the formatting guidelines for each of the data sources below. We provide the templates below
in which ⟨token⟩refers to a sentinel token, and metadata and data refer to placeholders for data fields,
respectively.

5.1
Source Code

We prepend the repository name and file paths to the context of the code file. We only add this metadata
with a 50% probability to enable the model to operate without this information. We use the following format
when adding the repository name and file paths:

<repo_name>reponame<file_sep>filepath1\ncode1<file_sep>filepath2\ncode2 ... <|endoftext|>.

We use the following format when we do not include this meta-data:

<file_sep>code1<file_sep>code2 ... <|endoftext|>.

Repository-context
Starcoder1 was trained with file-context, i.e., the setting where random files are
joined into the context window. In this work, we explore training with repository-context, wherein files from
the same repository are grouped together. While we considered various methods for grouping files within the
repository, we ultimately arranged them in a random order within the same repository.

FIM
To enable the model to perform code infilling tasks, we apply the fill-in-the-middle transformation (FIM;
Bavarian et al., 2022) to the source code. While we explored several FIM variants in preliminary experiments,
we opted for repo-context file-level FIM in the StarCoder2 models. In this FIM variant, repositories are
selected with a 50% chance of being candidates for FIM. The selected repository examples are split by
<|endoftext|> and <file_sep> tokens. Next, we apply the FIM transformation to each chunk with a 50%
probability. We do not apply FIM to the repository metadata (<repo_name>reponame). Below, we provide
an example of the FIM format when it’s only applied to the second source file:

<repo_name>reponame<file_sep>filepath0\ncode0<file_sep><fim_prefix>filepath1\n
code1_pre<fim_suffix>code1_suf<fim_middle>code1_mid<file_sep> ...<|endoftext|>

5.2
Pull Requests

Formatting pull requests is challenging as we aim to create a compact representation of a potentially long
sequence of code changes and comments. We refer to §2.3 for details on how we removed and truncated long
input fields of the pull request. Here, we focus on how to render the PR into a structured format that can be
consumed by the LLM.

For files part of the base commit, we include the entire file with 0.2 probability; otherwise, we display a range
of changes in the base files across all commit heads of the PR.22 We randomly add up to 32 lines before and
after the changes.

22We take the union of file line changes in all commits

16

Under review as submission to TMLR

We use diff hunks to display modifications between the before and after state of the file, ensuring that changes
are reasonably localized. Additionally, within the diff hunks, we incorporate 3-10 randomly selected context
lines both before and after the specific change.

We structure the PR format as follows. The first block presents the title, description, and complete base files
or modifications made to them. Subsequently, we outline the first set of head diff hunks:

<pr>Title: title\nusername_0: description
<pr_status>opened
<repo_name>reponame

<pr_base>
<pr_file>filepath_1
<pr_base_code>file_content/changes_1
...
<pr_file>filepath_N
<pr_base_code>file_content/changes_N

<pr_diff>
<pr_file>filepath_1
<pr_diff_hunk>diff_hunk_1
...
<pr_diff_hunk>diff_hunk_K
...
<pr_file>filepath_M
<pr_diff_hunk>diff_hunk_1
...
<pr_diff_hunk>diff_hunk_J

The second block is repeated for each new head commit in the PR, covering general comments, review
comments, and code review comments. The block concludes with the diff hunks between the pull request
base and the new head, reflecting the outcome of discussions and comments. Note that it’s also possible
for users to close and reopen the pull request. As in Github issues, we refer to authors by their participant
counter within the conversation, e.g., username_1, to refer to the second participant in the issue.

<pr_comment>username_id: comment
<pr_event_id>comment_id
...
...
...
<pr_review>username_id: review_comment\n
<pr_event_id>review_id
<pr_review_state>[approved, rejected, commented, changes_required]
...
...
...
<pr_review_comment>
<pr_event_id>comment_id
<pr_in_reply_to_review_id>review_id (opt)
<pr_in_reply_to_comment_id>comment_id (opt)
<pr_file>filepath
<pr_diff_hunk_comment_line>line_number
<pr_diff_hunk>diff_hunk_content
<pr_comment>username_id: comment

17

Under review as submission to TMLR

...
...
...
<pr>username_id
<pr_status>closed
<pr_is_merged>False
...
<pr>Title: title\nusername_id: description
<pr_status>[opened, reopened, edited]
...
...
...
<pr_file>filepath_1
<pr_diff_hunk>diff_hunk_1
...
<pr_diff_hunk>diff_hunk_K
...
<pr_file>filepath_M
<pr_diff_hunk>diff_hunk_1
...
<pr_diff_hunk>diff_hunk_J

We only add the following final block when the PR is closed.

<pr>username_id
<pr_status>closed
<pr_is_merged>True
<|endoftext|>

5.3
GitHub Issues

We use sentinel tokens to mark the opening of an issue and subsequently include its title. We separate the
sequence of comments by a <issue_comment> token and include an anonymized speaker identifier before
the comment. Specifically, we refer to authors by their participant counter within the conversation, e.g.,
username_1, to refer to the second participant in the issue. To distinguish between the different turns,
we use comment_1, id1 to refer to the second comment and its anonymized speaker id, respectively. The
<issue_closed> token is added if the issue is closed.

<issue_start>Title: title\nusername_id0: comment_0<issue_comment>username_id1: comment_1
... <issue_closed (optional)><issue_comment>username_idn: comment_n<|endoftext|>

5.4
Notebooks

Jupyter – scripts
We format Jupyter scripts as a single code block, starting with a <jupyter_script>
token.

<jupyter_script>code<|endoftext|>

Jupyter – structured
Parsed Jupyter notebooks are chains of text, code, and outputs. We separate the
cells with sentinel tokens. Note that we use text2, code2, output2 to refer to the 3rd triplet in the notebook.

<jupyter_start><jupyter_text>text0<jupyter_code>code0
<jupyter_output>output0<jupyter_text> ... <|endoftext|>

18

Under review as submission to TMLR

Kaggle – scripts
When available, we prepend the associated dataset title and description to Kaggle
notebooks (42% of the samples). For 8.6% of the notebooks, we add granular information on the dataset’s
schema. Below is the format we use:

<jupyter_start><jupyter_text>title\ndescription\nKaggle dataset identifier: data_identifier
<jupyter_code>import pandas as pd\n\ndf = pd.read_csv(data_path1)\ndf.info()
<jupyter_output>df_info_output1
<jupyter_text>Examples:\nexample1_1\n..example1_4
...
<jupyter_script>code<|endoftext|>

Some notebooks might load more than one csv file, so we repeat the blocks of data information content for
all files.

Note that we introduce a new special token <jupyter_script> to append the final script of the converted
Kaggle notebook. This token helps differentiate the script, which is usually long, from code that follows
<jupyter_code> token, typically shorter.

Kaggle – structured
Structured Kaggle notebooks are similar to structured Jupyter notebooks, except
that they don’t have an output cell, so we only include text and code blocks and keep the tokens used in
Jupyter Notebooks:

<jupyter_start><jupyter_text>text0<jupyter_code>code0<jupyter_text> ... <|endoftext|>

5.5
StackExchange

We concatenate questions and answers in the StackOverflow dataset using a format similar to the GitHub
issues. We start with the question and then add answers in random order. We include the upvote score
alongside the answer and, if applicable, denote it as the selected answer. Note that we do not have the title
of the conversations for the StackExchange dataset.

<issue_start>username_id0: question
<issue_comment>username_id1: answer_1\nUpvotes: score [selected answer](Optional)
...
<issue_comment>username_idn: answer_n\nUpvotes: score [selected answer](Optional)<|endoftext|>

5.6
Intermediate Representations

We split 50/50 between translating from source code to intermediate representation (code->intermediate)
and vice-versa (intermediate->code). Regarding the intermediate representation, we use the size-optimized
version 80% of the time and the performance-optimized version 20% of the time. We use separate sentinel
tokens to indicate the direction of the translation.

code<code_to_intermediate>intermediate_representation
intermediate_representation<intermediate_to_code>code

6
Model architecture and training details

In this section, we provide all details regarding the model architecture (§6.1), tokenizer (§6.2), training details
(§6.3), and CO2 emissions during training (§6.4).

23Estimated with 6ND, where N is the number of parameters and D is the number of training tokens. Includes base and
long-context training.

19

Under review as submission to TMLR

Table 5: Overview of the sentinel tokens.

Token
Description

<|endoftext|>
end of text/sequence
<fim_prefix>
FIM prefix
<fim_middle>
FIM middle
<fim_suffix>
FIM suffix
<fim_pad>
FIM pad
<repo_name>
repository name
<file_sep>
file separator
<issue_start>
start of GitHub issue
<issue_comment>
start of GitHub issue comment
<issue_closed>
GitHub issue closed event
<jupyter_start>
start of Jupyter notebook
<jupyter_text>
start of Jupyter text cell
<jupyter_code>
start of Jupyter code cell
<jupyter_output>
start of Jupyter output cell
<jupyter_script>
start of Jupyter script (converted kaggle notebook)
<empty_output>
output cell without content
<code_to_intermediate>
translate source code to intermediate representation
<intermediate_to_code>
translate intermediate representation to source code
<pr>
start of pull request
<pr_status>
status of pull request
<pr_is_merged>
whether pr is merged
<pr_base>
start of list of base files
<pr_file>
path of pull request file
<pr_base_code>
code that is part of the base commit in the PR
<pr_diff>
start of a diff
<pr_diff_hunk>
diff hunk
<pr_comment>
general comment
<pr_event_id>
GitHub id of review comment or code review comment
<pr_review>
start of review
<pr_review_state>
review state (e.g. approved, rejected)
<pr_review_comment>
code review comment
<pr_in_reply_to_review_id>
GitHub event id of review
<pr_in_reply_to_comment_id>
GitHub event id of comment
<pr_diff_hunk_comment_line>
line number of code review comment

6.1
Model Architecture

We introduce a few architectural changes compared to StarCoderBase. First, we replace learned positional
embeddings with Rotary Positional Encodings (RoPE; Su et al., 2021), as we confirmed significant performance
gains in a preliminary ablation study. Following DeepseekCoder (Guo et al., 2024) and Code LLaMA (Rozière
et al., 2023), we use a base period θ = 1e5. The second architectural modification we make is replacing
Multi-Query Attention (MQA; Shazeer, 2019) with Grouped Query Attention (Ainslie et al., 2023, GQA;
). However, we keep the number of key-value heads relatively low—2 for the 3B, 4 for the 7B and 15B—to
prevent significantly slowing down inference.

We summarize all other hyperparameters, such as the number of layers and hidden dimension, in Table 6.

20

---

*Source: arXiv:2402.19173*
