---
id: B2
title: "Competition-Level Code Generation with AlphaCode"
domain: B
year: 2022
arxiv_id: "2203.07814"
confidence: verified
source: "arXiv:2203.07814 / Science"
node_type: paper
---

# Competition-Level Code Generation with AlphaCode

**Domain**: [[domain_B|LLM Code Generation]] | **Year**: 2022 | **Confidence**: [x] verified


## Authors
[[author_Yujia Li|Yujia Li]], [[author_David Choi|David Choi]], [[author_Junyoung Chung|Junyoung Chung]], [[author_Nate Kushman|Nate Kushman]], [[author_Julian Schrittwieser|Julian Schrittwieser]], et al.


## Keywords
- [[kw_AlphaCode|AlphaCode]]
- [[kw_competitive programming|competitive programming]]
- [[kw_code generation|code generation]]
- [[kw_massive sampling|massive sampling]]
- [[kw_filtering|filtering]]

## Abstract

(Abstract not available - see PDF content below)

## Paper Content

2022-3-16

Competition-Level Code Generation with
AlphaCode

Yujia Li*, David Choi*, Junyoung Chung*, Nate Kushman*, Julian Schrittwieser*, Rémi Leblond*, Tom
Eccles*, James Keeling*, Felix Gimeno*, Agustin Dal Lago*, Thomas Hubert*, Peter Choy*, Cyprien de
Masson d’Autume*, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey
Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de
Freitas, Koray Kavukcuoglu and Oriol Vinyals
*Joint ﬁrst authors

Programming is a powerful and ubiquitous problem-solving tool. Developing systems that can assist programmers or even generate programs independently could make programming more productive and
accessible, yet so far incorporating innovations in AI has proven challenging. Recent large-scale language models have demonstrated an impressive ability to generate code, and are now able to complete
simple programming tasks. However, these models still perform poorly when evaluated on more complex, unseen problems that require problem-solving skills beyond simply translating instructions into
code. For example, competitive programming problems which require an understanding of algorithms
and complex natural language remain extremely challenging. To address this gap, we introduce AlphaCode, a system for code generation that can create novel solutions to these problems that require deeper
reasoning. In simulated evaluations on recent programming competitions on the Codeforces platform,
AlphaCode achieved on average a ranking of top 54.3% in competitions with more than 5,000 participants. We found that three key components were critical to achieve good and reliable performance:
(1) an extensive and clean competitive programming dataset for training and evaluation, (2) large and
eﬃcient-to-sample transformer-based architectures, and (3) large-scale model sampling to explore the
search space, followed by ﬁltering based on program behavior to a small set of submissions.

1
Introduction
2

5.4
Results on APPS . . . . . . . . . .
19

2
Problem setup
5
2.1
Competitive programming . . . .
5
2.2
Evaluation . . . . . . . . . . . . .
6

6
AlphaCode’s capabilities & limitations 20
6.1
Copying from training data
. . .
21
6.2
Model solution characteristics . .
22
6.3
Sensitivity to problem descriptions
24
6.4
Sensitivity to provided metadata
24
6.5
Loss is a poor proxy for solve rate
26

3
Datasets
6
3.1
Pre-training dataset . . . . . . . .
7
3.2
CodeContests ﬁne-tuning dataset
7

arXiv:2203.07814v1  [cs.PL]  8 Feb 2022

7
Related work
27
7.1
Program synthesis
. . . . . . . .
27
7.2
Transformers for program synthesis 28
7.3
Scaling sampling . . . . . . . . .
28
7.4
Evaluation metrics
. . . . . . . .
29
7.5
Competitive programming . . . .
29

4
Approach
9
4.1
Model architecture . . . . . . . .
10
4.2
Pre-training . . . . . . . . . . . .
11
4.3
Fine-tuning . . . . . . . . . . . .
11
4.4
Large scale sampling . . . . . . .
12
4.5
Filtering . . . . . . . . . . . . . .
13
4.6
Clustering . . . . . . . . . . . . .
13

8
Broader impact
29
8.1
Applications . . . . . . . . . . . .
29
8.2
Potential risks and beneﬁts . . . .
30

9
Conclusion
31

5
Results
13
5.1
Codeforces competitions evaluation 14
5.2
CodeContests evaluation . . . . .
15
5.3
CodeContests ablations & results
15

10 Appendix
38

Corresponding author(s): yujiali@deepmind.com, davidhchoi@deepmind.com, vinyals@deepmind.com
© 2022 DeepMind. All rights reserved

Competition-Level Code Generation with AlphaCode

1. Introduction

Computer programming has emerged as a general-purpose problem-solving tool throughout science,
industry, and daily life. As part of this growth, there has been continuously increasing demand for
tools that can make programmers more productive (Matsakis and Klock, 2014), or make programming
and programming education more accessible (Resnick et al., 2009). Developing AI systems that can
eﬀectively model and understand code can transform these tools and the way we interact with them.
Systems that can generate code are not only useful, but also stepping stones that can lead to greater
understanding of AI and how it relates to programming.

Generating code that solves a speciﬁed task requires searching in the huge structured space of possible
programs, with a very sparse reward signal. Single character edits can completely change program
behaviour even if they don’t cause crashes, solutions can look dramatically diﬀerent even for the same
problem, and judging if a partial or incorrect program is useful is a diﬃcult challenge. Therefore, most
prior work has been limited to either restricted domain-speciﬁc programming languages (Gulwani,
2011) or short code snippets (Bruch et al., 2009; Raychev et al., 2014).

Recent large-scale transformer-based (Vaswani et al., 2017) language models, used to achieve impressive performance generating text (Brown et al., 2020), have successfully generated code that solves
simple programming problems in Python (Austin et al., 2021; Chen et al., 2021). A stripped-down
version of our model, without the modiﬁcations described in Section 4, performs similarly to Codex
(Table A3). However, problems used in the Codex paper and similar work consist of mostly simple
task descriptions with short solutions – far from the full complexity of real-world programming.
Generating an entire program in a general-purpose programming language such as C++ or Python,
starting from a long natural language task description, has remained an open problem. The diﬀerence
in diﬃculty between generating short code snippets and entire programs can be analogous to that of
imperative versus declarative problem solving. Generating short code snippets typically amounts to
translating the task speciﬁcation directly into code, and sometimes reduces to invoking the correct
API calls. In contrast, generating entire programs often relies on understanding the task and ﬁguring
out how to accomplish it, which requires deeper algorithmic reasoning.

Competitive programming problems represent a signiﬁcant step forward in all these aspects. Solving
such problems requires understanding complex natural language descriptions, reasoning about
previously unseen problems, mastering a wide range of algorithms and data structures, and precisely
implementing solutions that can span hundreds of lines. Solutions are evaluated by executing them
on an exhaustive suite of unknown tests, checking for correct behaviour on edge cases as well
as execution speed. The fact that the test cases used for evaluation are hidden is an important
part of the challenge. These complex problems are newly created for each competition, with the
understanding that competitors can draw on solutions to previous contests (either implicitly, by
remembering old problems, or explicitly, by searching for them). Moreover, competitive programming
is very popular; events like the International Collegiate Programming Competition (ICPC, 2021) and
the International Olympiad in Informatics (IOI, 2021) are widely recognized as some of the most
prestigious competitions in computer science, drawing hundreds of thousands of participants from
around the world. Using problems that humans ﬁnd challenging from such battle-tested competitions
ensures robustness against shortcuts and provides a meaningful benchmark for many aspects of
intelligence.

Early work using program synthesis for competitive programming has shown that large transformer
models can achieve low single-digit solve rates (Chen et al., 2021; Hendrycks et al., 2021), but could
not yet reliably generate solutions for the vast majority of problems. Furthermore, as we show in
Section 3.2.1, the lack of suﬃcient test cases in existing competitive programming datasets makes

2

Competition-Level Code Generation with AlphaCode

2500

1591

1608

AlphaCode

2000

1613

1615

1500

1617

1618

1000

Codeforces rating

1619

1620

500

1622

1623

0

Competition ranking

Competition ranking

0%
20%
40%
60%
80%
100%
%competitors 
 rating

(a) AlphaCode’s ranking in 10 contests
(b) AlphaCode’s estimated rating

Figure 1 | AlphaCode’s ranking on 10 simulated Codeforces contests and estimated rating (right
is better). AlphaCode ranked in the top 54.3% among contest participants averaged over 10 contests,
and achieved an estimated average rating of 1238. (a) shows the rating of participants (y-axis) and
their rankings in each contest (x-axis), as well as AlphaCode’s ranking for each of the 10 contests. (b)
shows the estimated rating of AlphaCode among users who have participated in at least 1 contest in
the last 6 months. AlphaCode’s estimated rating of 1238 is greater than 72% of these users.

the metrics deﬁned on them prone to high false positive rates (with 30% or more programs which
pass all tests but are not actually correct), and therefore unreliable for measuring research progress.

In this paper we present AlphaCode, a code generation system applied to solving competitive programming problems. We use large transformer language models to generate code, pre-training them
on selected GitHub code and ﬁne-tuning on our curated set of competitive programming problems.
For each unseen problem we generate a large set of program samples, ﬁlter them based on execution
results on example tests from the problem description, then cluster the remaining samples to obtain a
small set of candidates to be submitted for evaluation. We describe AlphaCode in detail in Section 4.

A core part of developing our system was ensuring that submissions are rigorously evaluated and
that evaluation problems are truly unseen during training, so diﬃcult problems cannot be solved
by copying from the training set. Towards this goal, we release a new training and evaluation
competitive programming dataset, CodeContests1 (Section 3). This dataset combines data from
various sources, splits temporally so all training data predates all evaluation problems, adds additional
generated tests to ensure correctness, and evaluates submissions in a setting that mirrors that of
competitive programming. In our evaluation (Section 3.2.1), CodeContests reduces the false positive
rate from 30-60% in existing datasets to just 4%. Our best model solves 34.2% of held-out competitive
programming problems in this dataset, using at most 10 submissions per problem (comparable to
humans), as opposed to previously reported solve rates of around 1-5% on existing datasets (see
Section 5.4).

To further validate our results, we evaluated AlphaCode on simulated programming competitions
hosted on the popular Codeforces platform2 (Section 5.1). In the evaluation of 10 recent contests
with over 5,000 participants each, AlphaCode achieved an average ranking within the top 54.3%.
Based on these results, we estimate that our system has achieved a Codeforces rating3 of 1238 which
is within the top 28%4 of users who have participated in a contest in the last 6 months (Figure 1)

1The dataset is located at https://github.com/deepmind/code_contests.
2https://codeforces.com/
3The rating system is similar to the classic Elo score and is primarily explained in three blog posts: 1, 2, and 3
4AlphaCode’s overall rating percentile is better than its per-contest percentile. We hypothesise that higher rated
competitors compete more regularly than lower rated competitors, and therefore the group ranking above AlphaCode in
contests is relatively more stable than the group ranking below.

3

Competition-Level Code Generation with AlphaCode

Example Input

Backspace
You are given two strings 𝑠and 𝑡, both consisting of lowercase English letters.
You are going to type the string 𝑠character by character, from the ﬁrst character
to the last one.

4
ababa
ba
ababa
bb
aaa
aaaa
aababa
ababa

Example Output

When typing a character, instead of pressing the button corresponding
to it, you can press the “Backspace” button. It deletes the last character you
have typed among those that aren’t deleted yet (or does nothing if there are no
characters in the current string). For example, if 𝑠is “abcbd” and you press
Backspace instead of typing the ﬁrst and the fourth characters, you will get the
string “bd” (the ﬁrst press of Backspace deletes no character, and the second
press deletes the character ’c’). Another example, if 𝑠is “abcaa” and you press
Backspace instead of the last two letters, then the resulting text is “a”.

YES
NO
NO
YES

Your task is to determine whether you can obtain the string 𝑡, if you
type the string 𝑠and press “Backspace” instead of typing several (maybe zero)
characters of 𝑠.

Explanation
In order to obtain “ba” from “ababa”,
you may press Backspace instead
of typing the ﬁrst and the fourth
characters.

There’s
no
way
to
obtain
“bb”
while typing “ababa”.

Input
The ﬁrst line contains a single integer 𝑞(1 ≤𝑞≤105) the number of test cases.
The ﬁrst line of each test case contains the string 𝑠(1 ≤|𝑠| ≤105). Each
character of 𝑠is a lowercase English letter.
The second line of each test case contains the string 𝑡(1 ≤|𝑡| ≤105). Each
character of 𝑡is a lowercase English letter.
It is guaranteed that the total number of characters in the strings over all test
cases does not exceed 2 · 105.

There’s no way to obtain “aaaa”
while typing “aaa”.

Output
For each test case, print “YES” if you can obtain the string 𝑡by typing the string
𝑠and replacing some characters with presses of “Backspace” button, or “NO” if
you cannot.
You may print each letter in any case (YES, yes, Yes will all be recognized as
positive answer, NO, no and nO will all be recognized as negative answer).

In order to obtain “ababa” while
typing “aababa”, you have to press
Backspace instead of typing the
ﬁrst character, then type all the
remaining characters.

Figure 2 | Competitive programming problem statement. Problem statement of Backspace, a
Codeforces problem (Mirzayanov, 2020). This is a problem of medium diﬃculty, with a rating of
1500. The right side shows the public example test case included in the problem description. Hidden
tests used to evaluate submissions are shown in Figure A1. A solution produced by AlphaCode is
shown in Figure 3. The entire statement is given to AlphaCode, and examples of the exact formatting
of problem descriptions seen by the model are provided in Appendix F.

(Ebtekar, 2021). These evaluations only include users who have tried such competitions, which is a
self-selected subset of all programmers. This is the ﬁrst time that a computer system has achieved
such a competitive level in programming competitions.

We also performed a detailed analysis of our system (Section 6), showing that AlphaCode does not
duplicate sections of code from the training dataset to solve problems, but instead relies heavily on
the natural language problem descriptions to create original solutions. We further examine the types
of problems the model can and cannot solve, and discuss how the validation loss is a poor proxy for
the solve rate.

4

Competition-Level Code Generation with AlphaCode

Figure 3 | Solution to Figure 2 generated by AlphaCode. The model successfully extracted the
information necessary to solve the problem from
the natural language description:

1. The problem is to ﬁgure out if it is possible
to convert one phrase to another by pressing
backspace instead of typing some letters. So
ﬁrst we read the two phrases (lines 3-4).
2. If the letters at the end of both phrases don’t
match, the last letter must be deleted. If they
do match we can move onto the second last
letter and repeat (11-18).
3. Backspace deletes two letters. The letter you
press backspace instead of, and the letter before it (19-20).
4. If we matched every letter, it is possible to
obtain string 𝑡from 𝑠(23-26).

1
 2
 3
 4
 5
 6
 7
 8
 9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26

t=int(input())
for i in range(t):
    s=input()
    t=input()
    a=[]
    b=[]
    for j in s:
        a.append(j)
    for j in t:
        b.append(j)
    a.reverse()
    b.reverse()
    c=[]
    while len(b)!=0 and len(a)!=0:
        if a[0]==b[0]:
            c.append(b.pop(0))
            a.pop(0)
        elif a[0]!=b[0] and len(a)!=1:
            a.pop(0)
            a.pop(0)
        elif a[0]!=b[0] and len(a)==1:
            a.pop(0)
    if len(b)==0:
        print("YES")
    else:
        print("NO")

2. Problem setup

2.1. Competitive programming

Programming competitions ﬁrst began in the 1970s and have since grown in popularity to include
hundreds of thousands of participants worldwide. The annual International Collegiate Programming
Contest attracts almost 60,000 students from over 3,000 universities (ICPC Factsheet, 2020), and
companies including Google (Google Code Jam, 2021) and Facebook (Facebook Hacker Cup, 2021)
hold regular competitions. The popular Codeforces platform, used throughout this paper, has more
than 500,000 active users and holds weekly competitions with tens of thousands of participants (Mirzayanov, 2020).

The exact format of a programming competition varies between contests, but in general individuals or
teams of competitors are given between 5 and 10 problem descriptions (Figure 2), and approximately
3 hours to write programs (Figure 3) to correctly solve as many problems as possible. The program
submissions are sent to a server which automatically evaluates them on an exhaustive set of hidden
tests (Figure A1). Competitors are told whether or not their submission passed all tests, though
not necessarily the exact cause of a failure. There are penalties based on the number of incorrect
submissions per problem and the amount of time it took to solve each problem (ICPC Rules, 2021).
Submissions can be written in a variety of programming languages, among which C++ and Python are
currently the most popular. Problems are often given ratings to indicate diﬃculty, and more diﬃcult
problems are worth more points.

There are three steps involved in solving a problem. First, participants must read and understand
a natural language description spanning multiple paragraphs that contains: narrative background
typically unrelated to the problem, a description of the desired solution that the competitors need
to understand and parse carefully, a speciﬁcation of the input and output format, and one or more
example input/output pairs (that we call “example tests”).

The next step is to create an eﬃcient algorithm that solves the problem. Going from “what the problem
is” to “how to solve the problem” is a great leap that requires understanding and reasoning about

5

Competition-Level Code Generation with AlphaCode

the problem, as well as a deep comprehension of a wide range of algorithms and data structures.
This leap is a signiﬁcant diﬀerence from previous works, which tend to explicitly specify what to
implement. The algorithm must also be eﬃcient enough to execute in time for the input sizes and
time limits speciﬁed by the problem,5 which often eliminates easier, naive attempts.

Finally, the algorithm must be implemented. Implementation eﬃciency matters given execution time
constraints (harder problems can sometimes only be solved in faster languages such as C++), subtle
edge cases can be diﬃcult to account for, and the solution itself can be over a hundred lines of precise
code. Participants are given small example test cases to run against, and often debug, ﬁx, and rerun
their candidate submission many times before attempting an oﬃcial submission against the hidden
tests cases. An example correct solution generated by AlphaCode for the problem in Figure 2 is given
in Figure 3, and extensive results and analysis can be found in Section 5 and 6.

2.2. Evaluation

Though running a system against a live programming competition is an unbiased evaluation, it adds a
large degree of complexity and is not a stable benchmark. To alleviate this issue, we developed a proxy
measure suitable for research iteration similar to the development sets present in most supervised
learning datasets. Our measure mirrors the fundamental structure of competitions while simplifying
incidental details. The metric we use is “percentage of problems solved using 𝑛submissions from 𝑘
samples per problem”, denoted as 𝑛@𝑘.

This metric indicates the percentage of problems a model can solve if for each problem it is allowed
ﬁrst to create 𝑘samples, and then to evaluate 𝑛≤𝑘of these samples against the hidden tests. The
problem is considered solved if any of these 𝑛evaluations passes all tests. The ﬁltering method is
up to the system itself, but should only be based on information available to competitors (e.g. the
example tests given as part of the problem description, but not the hidden tests). To decrease variance
between runs, assuming both 𝑛and 𝑘are ﬁnite, the metrics we report are expectations computed using
bootstrapping on a set of samples typically much larger than 𝑘(Appendix A.3). Decreasing variance
through expectations makes comparisons of improvements more meaningful, as our validation and
test sets are relatively small, and there is signiﬁcant variance when sampling from a single model.

Limiting the amount of submissions to 𝑛emulates the penalty for incorrect submissions and prevents
systems from exploiting the evaluation metric by evaluating against the hidden tests an unreasonable
number of times. Fixing 𝑘is important for comparing diﬀerent evaluations, as we found that
performance increases with the number of samples (Section 5). Our use of bootstrapping ensures
that we can still beneﬁt from the variance reduction obtained from generating a much larger set of
𝐾≫𝑘samples to estimate the 𝑛@𝑘metric.

The setting we use to model programming competitions is 10@𝑘– 10 submissions per problem from
𝑘samples. We also use 𝑝𝑎𝑠𝑠@𝑘(solve rate with 𝑘samples), to be consistent with Chen et al. (2021),
which assumes all samples can be submitted for evaluation. 𝑝𝑎𝑠𝑠@𝑘= 𝑘@𝑘, and is an upper bound
metric for using 𝑘samples. We show solve rate with respect to diﬀerent 𝑘values as good results at
low sample budgets do not necessarily correlate with good performance at high sample budgets.

3. Datasets

All our models were ﬁrst pre-trained on a collection of open-source code from GitHub, and subsequently
ﬁne-tuned on a dataset we created (CodeContests, released here) of programming competition data.

5The time limit for the problem in Figure 2 is 2 seconds, using at most 256 MB of memory.

6

Competition-Level Code Generation with AlphaCode

The pre-training stage helps the model learn good representations of code and generate code ﬂuently,
while the ﬁne-tuning stage helps the model adapt to the target competitive programming domain.

3.1. Pre-training dataset

Our pre-training dataset is based on a snapshot of selected public GitHub repositories taken on
2021/07/14. We included all code ﬁles from several popular languages: C++, C#, Go, Java, JavaScript,
Lua, PHP, Python, Ruby, Rust, Scala, and TypeScript. Following previous work (Chen et al., 2021), we
ﬁltered out all ﬁles larger than 1MB or with lines longer than 1000 characters, to exclude automatically
generated code. We also removed duplicates of the same ﬁle, ignoring whitespace in comparisons.
After ﬁltering, our ﬁnal pre-training dataset contains a total of 715.1 GB of code. The dataset
composition across languages can be found in the appendix (Table A1).

3.2. CodeContests ﬁne-tuning dataset

Models pre-trained on GitHub can generate good code and solve simple programming problems, but
as shown in Appendix B.3 they can solve very few competitive programming problems. Fine-tuning
the model on a dedicated competitive programming dataset is critical for performance.

To facilitate ﬁne-tuning and evaluation, we curated a new dataset of competitive programming
problems, named CodeContests.6 The dataset includes problems, solutions and test cases we scraped
from the Codeforces platform, along with existing public competitive programming datasets mixed
into our training set. More concretely, the training dataset combines newly scraped data from
Codeforces (Mirzayanov, 2020) with existing data from Description2Code (Caballero et al., 2016),
and CodeNet (Puri et al., 2021). The validation and test splits of the dataset consist entirely of newly
scraped Codeforces problems. To guard against data leakage, we adopted a strict temporal split: all
pre-training and ﬁne-tuning training data appeared online before any validation problems, and all
validation problems before test ones. Following our GitHub pre-training dataset snapshot date, all
training data in CodeContests was publicly released on or before 2021/07/14. Validation problems
appeared between 2021/07/15 and 2021/09/20, and the test set contains problems published after
2021/09/21. This temporal split means that only information humans could have seen is available
for training the model (see Appendix B.3 for more details and analysis). Some basic statistics of this
dataset are shown in Table 1.

Our scraped data from Codeforces includes full problem descriptions like that shown in Figure 2,
along with metadata for each problem. The metadata includes diﬃculty ratings and tags that indicate
which approaches might be required to solve the problem (e.g. “greedy” or “dp”). Neither the diﬃculty
rating nor the tags are visible at competition time (and so should not be used at test time). Our dataset
also contains both correct and incorrect human submissions written in the most popular submission
languages: C++, Python, and Java. Each problem includes all the test cases that are accessible from
the platform: example tests in the problem statements and hidden test cases that are made available
at the evaluation result pages once a contest is ﬁnished. To improve data quality and consistency, and
to avoid duplication issues involved in merging datasets, we cleaned this data using the procedure
outlined in Appendix B.2.

The correctness of a program is checked by executing it on the test cases and comparing the program
output with the expected correct output. More details about this correctness checking process are
documented in Appendix A.2.

6The dataset can be found on GitHub.

7

Competition-Level Code Generation with AlphaCode

Tests per problem
Solutions per problem (% correct)
Split
Problems
Example
Hidden
Generated
C++
Python
Java

Train
13328
2.0
14.8
79.1
493.4 (27%)
281.1 (47%)
147.9 (46%)
Valid
117
1.5
12.9
190.0
231.6 (47%)
137.2 (55%)
131.1 (54%)
Test
165
1.7
9.4
192.7
196.0 (45%)
97.3 (54%)
105.2 (51%)

Table 1 | Statistics of our CodeContests dataset. The number of problems in each split, and the
per-problem averages for the number of test cases, number of solutions, and percentage of solutions
which are correct.

Dataset
Tests / problem
False Positive (FP) Rate
FP or Slow Rate

APPS
20.99
60%
70%
HumanEval
7.77
30%
N/A
CodeContests raw
12.4
62%
88%
CodeContests
203.7
4%
46%

Table 2 | Dataset false positive rates. The bottom row is the dataset we used, while “CodeContests
raw” does not use generated tests and does not ﬁlter out problems with insuﬃcient tests. Validation
splits were used for CodeContests and APPS. We randomly selected 50 problems our 1B parameter
model solved (from 10,000 samples per problem for APPS, 200 for HumanEval, and 1,000,000 for
CodeContests), and manually examined one solution for each problem to check whether they are
false positives or slow solutions. HumanEval does not have timing constraints for most problems, so
there is no slow rate.

3.2.1. False positives and additional generated tests

We want the test cases to be as exhaustive as possible, so that submissions cannot be marked as
correct by exploiting a lack of test coverage. Unfortunately, high-quality test cases are not readily
available. For example, the Codeforces platform does not display full test cases when they are longer
than approximately 400 characters. Lack of test coverage leads to “false positives” where incorrect
submissions are marked as correct, and “slow positives” where correct but algorithmically ineﬃcient
solutions that do not fulﬁll time and memory constraints are marked correct (e.g. a solution that
is of the wrong complexity class). These false positives do not eﬀect the evaluation on Codeforces
described Section 5.1.

Notably, both issues are common in prior datasets and the program synthesis literature, as input/output
examples are an under-speciﬁcation of program behavior (Gulwani et al., 2017). Table 2 shows
the estimated false positive rate of our dataset compared to APPS (Hendrycks et al., 2021) and
HumanEval (Chen et al., 2021), which both have many false positives. A high average number of
tests per problem does not necessarily indicate exhaustive tests, because some problems may have far
fewer tests per problem than average, and some tests may examine similar cases.

We reduced the false positive rates of our dataset by generating additional test cases, created by
mutating existing test inputs. Possible mutations are applying bit ﬂips to binary inputs, randomly
incrementing or decrementing integers, and swapping and changing characters in strings. Mutated
inputs are veriﬁed by running 30 correct solutions on them, and checking that all solutions produce
the same output. This process was run on each problem for a maximum of 10 CPU hours or 200
generated tests. Because of complex input formats, we failed to generate the full set of 200 tests for
6.3% of problems.

8

Competition-Level Code Generation with AlphaCode

Figure 4 | Overview of AlphaCode.

Lastly, we ﬁltered out problems in the validation and test splits with insuﬃcient test coverage, keeping
only problems with at least 5 hidden or generated test cases that result in at least 2 diﬀerent outputs.
This ensures a model cannot trivially solve problems by always outputting a constant, such as YES or
NO. As seen in Table 2, generated tests and ﬁltering reduced our false positive rates from 62% to
4%. CodeContests has signiﬁcantly better false positive rates than prior work even though we drew
fewer samples for both APPS and HumanEval, and the problems in those datasets are relatively less
complex (both of which tend to lower the false positive rates). However, there is still a signiﬁcant
number of problems where slow but semantically correct solutions are accepted by the tests.

4. Approach

Generating code that solves a speciﬁc task requires searching in a huge structured space of programs
with a very sparse reward signal. To make matters worse, for many domains including competitive
programming, there is a limited number of examples of such tasks and solutions to learn from. Finally,
as we restrict the amount of submissions per problem our model can do, each submission must be
used wisely.

Our system, AlphaCode, is meant to address all these challenges. A high-level view of our approach
can be seen in Figure 4. The main process is to:

1. Pre-train a transformer-based language model on GitHub code with standard language modelling
objectives. This model can reasonably represent the space of human coding, which greatly
reduces the problem search space.
2. Fine-tune the model on our dataset of competitive programming data, using GOLD (Pang and
He, 2020) with tempering (Dabre and Fujita, 2020) as the training objective. This further
reduces the search space, and compensates for the small amount of competitive programming
data by leveraging pre-training.
3. Generate a very large number of samples from our models for each problem.
4. Filter the samples to obtain a small set of candidate submissions (at most 10), to be evaluated
on the hidden test cases, by using the example tests and clustering to pick samples based on
program behaviour.

Among these, the large-scale sampling followed by ﬁltering is unique to our setup, and we found that
this process greatly improves problem solve rate. Therefore many of our design decisions were made
to facilitate eﬃcient and eﬀective sampling.

9

Competition-Level Code Generation with AlphaCode

Heads
Blocks
Training
Name
𝑛𝑝𝑎𝑟𝑎𝑚𝑠
𝑑𝑚𝑜𝑑𝑒𝑙
Query
KV
Enc
Dec
Batch
Steps
Tokens

AlphaCode 300M
284M
768
6
1
4
24
256
600k
354B
AlphaCode 1B
1.1B
1408
11
1
5
30
256
1000k
590B
AlphaCode 3B
2.8B
2048
16
1
6
36
512
700k
826B
AlphaCode 9B
8.7B
3072
24
4
8
48
1024
530k
1250B
AlphaCode 41B
41.1B
6144
48
16
8
56
2048
205k
967B

Table 3 | Architecture conﬁguration of our models at diﬀerent parameter scales. This table lists
the total number of parameters in the model 𝑛𝑝𝑎𝑟𝑎𝑚𝑠, the hidden dimension of the transformer blocks
𝑑𝑚𝑜𝑑𝑒𝑙, the number of query and key-value heads, the number of transformer blocks in the encoder
and decoder, the training batch size, the number of gradient update steps, and the number of total
training tokens. The head size is always 128, with a feed-forward fan-out ratio of 6.

4.1. Model architecture

The competitive programming code generation problem can be viewed as a sequence-to-sequence
(Sutskever et al., 2014) translation task: given a problem description 𝑋in natural language (e.g. Figure

2), produce a corresponding solution 𝑌in a programming language (e.g. Figure 3). This naturally
motivates the choice of an encoder-decoder transformer architecture (Vaswani et al., 2017) for
AlphaCode, which models 𝑝(𝑌|𝑋). The architecture takes as input to the encoder the problem
description 𝑋as a ﬂat sequence of characters (including metadata, tokenized), and samples 𝑌
autoregressively from the decoder one token at a time until an end of code token is produced,
at which point the code can be compiled and run (see Appendix F for example 𝑋, 𝑌pairs, and
https://alphacode.deepmind.com/ for an interactive model visualisation).

Compared to decoder-only architectures commonly used for language modeling and generation,
an encoder-decoder architecture allows a bidirectional description representation (tokens at the
beginning of the description can attend to tokens at the end) and the extra ﬂexibility to untie the
encoder structure from the decoder. Because problem descriptions are on average twice as long as
their corresponding human solutions, we use an asymmetric architecture with 1536 tokens for the
encoder but only 768 tokens for the decoder. We further found that using a shallow encoder and a
deep decoder signiﬁcantly improves the eﬃciency of training without hurting problem solve rate. The
exact architectures for our models are listed in Table 3. The 9B and 41B models were trained using
model parallelism, with 1 key and value head per shard. We built our model using JAX (Bradbury
et al., 2018) and Haiku (Hennigan et al., 2020), and trained them on TPUv4 accelerators using
bﬂoat16 precision.

To reduce the cost of sampling from our models, we take advantage of multi-query attention (Shazeer,
2019). Using a full set of query heads but sharing key and value heads per attention block signiﬁcantly
reduces memory usage and cache update costs, which are the main bottleneck during sampling. This
memory reduction also allows larger batch sizes for sampling, further increasing eﬃciency.

For tokenization we used a SentencePiece tokenizer (Kudo and Richardson, 2018) with a vocabulary
size of 8,000 tokens trained on a mix of GitHub and CodeContests data. The training mix ensures
it can eﬀectively tokenize programs from a range of languages, as well as the natural language
descriptions of problems. The encoder and decoder in our models use the same tokenizer.

10

Competition-Level Code Generation with AlphaCode

4.2. Pre-training

We pre-trained our models on the GitHub dataset described in Section 3, with a standard cross-entropy
next-token prediction loss for the decoder and a masked language modeling loss (Devlin et al., 2018)
for the encoder. The masked language modeling loss was essential for improving the representation
learning of the encoder. We split GitHub ﬁles by uniformly sampling pivot locations, using content
before the pivot as input to the encoder, and content after for the decoder.

Our base 1B parameter model was trained for 106 steps with a batch size of 256. Following Kaplan
et al. (2020), we adjusted the amount of training for other model sizes such that larger models are
trained more and smaller models are trained less to optimize the use of compute. However, due to
resource limitations and to make optimal use of compute, the training of our largest 41B model was
stopped early, and therefore this model was relatively undertrained compared to models at other
scales (Table 3).

We trained all models using the AdamW variant (Loshchilov and Hutter, 2017) of the Adam optimiser (Kingma and Ba, 2014) with 𝛽1 = 0.9, 𝛽2 = 0.999 for {300M, 1B, 3B} models, and 𝛽2 = 0.95
for {9B, 41B} models. We used a weight decay of 0.1 to enhance regularization. We trained the
models with an initial learning rate of 10−4, which was then cosine decayed to 10−5 at the end of
pre-training. We linearly warmed-up the learning rate from 10−9 to 10−4 over the ﬁrst 1, 000 training
steps, and clipped the global gradient norm to stay below 1.0.

4.3. Fine-tuning

We ﬁne-tuned our model on our CodeContests dataset. During ﬁne-tuning, we used the natural
language problem description for the encoder and the program solution for the decoder. Similar to
pre-training, we used both the standard next-token prediction and masked language modeling losses.
We also adopted additional conditioning and modiﬁcations that we found improved the overall solve
rate: tempering, value conditioning and prediction, and GOLD described below, as well as metadata
conditioning described in Appendix C.2. We set the initial learning rate as 10−5, and cosine decayed
it to 10−6 at the end of ﬁne-tuning. We used the same linear warm-up stage for the learning rate over
the ﬁrst 1, 000 training steps.

Tempering.
Tempering, introduced by Dabre and Fujita (2020), is a regularization technique that
makes the token probability distribution artiﬁcially smoother or sharper at training time by dividing
the output logits of a model by a scalar temperature 𝑇before the softmax layer. We observed that
when using 𝑇= 0.2 < 1, tempering helps avoid overﬁtting to our ﬁne-tuning dataset by making
the training distribution sharper, and consequently the inference distribution smoother. Notably,
this is the opposite of the suggestion of Dabre and Fujita (2020) to use 𝑇> 1 to make a sharper
inference distribution. At sampling time, we divided the logits by another temperature 𝑇′ tuned on
the validation set (𝑇′ = 0.12 for models trained with tempering only; 𝑇′ = 0.25 for models trained
with tempering and GOLD).

Value conditioning & prediction.
CodeContests contains both correct and incorrect problem
submissions. We used value conditioning and prediction to discriminate between these two types of
submissions, providing an additional training signal and allowing use of data which could otherwise
mislead the model. Similar approaches were used in, e.g., Vinyals et al. (2019). In value conditioning,
we inserted whether or not a submission was correct into the problem description so that the model
can condition on this information, as shown in Figure 5. At sampling time, the model was always
conditioned on the sample being correct. In value prediction, we added an auxiliary value prediction
task during training such that the last layer token representations before projecting to logits are also
used in a small Transformer to classify whether the submission is correct. Value prediction was not

11

Competition-Level Code Generation with AlphaCode

RATING: 1200
TAGS: dp , implementation
LANGUAGE
IS
python3
CORRECT
SOLUTION
Polycarp
must
pay
exactly n burles at the
checkout
... (rest of the
description )

Figure 5 | Example format of the additional metadata information. This is added to the top of
problem descriptions. Metadata and problem descriptions are handled identically. See Appendix F for
a full example of what is used in the decoder. The problem in this example can be found here.

used during sampling.

GOLD (Pang and He, 2020).
Solving competitive programming problems from descriptions is
inherently a one-of-many task (Nandwani et al., 2021): each unique problem allows many distinct
solutions that depend on algorithm choice, implementation, etc. CodeContests contains several orders
of magnitude more solutions than descriptions (Table 1). Standard maximum likelihood objectives
minimise loss by putting some weight on each solution in the training set (like recall), whereas
our metric measures whether a model can ﬁnd a single correct solution in the submission attempt
budget (like precision). To resolve this discrepancy, we adopted a variation of GOLD (Pang and He,
2020), an oﬄine RL algorithm which allows the model to both learn from tokens it already assigns
high likelihood to, and to ignore tokens that are not in its distribution (allowing it to concentrate on
precision). To combine GOLD and tempering, we introduce a short training phase between pretraining
and ﬁnetuning. Full details of GOLD and this combination are in Appendix C.3.

4.4. Large scale sampling

Sampling from transformer models can be easily parallelized, which allowed us to scale to millions
of samples per problem – a critical driving force for performance improvement. To ensure suﬃcient
diversity in such a large number of samples, we take a single trained model and: (i) generate half of
the samples in Python and half in C++, (ii) randomize the problem tags and ratings in the natural
language prompt (see Figure 5 for an example and Appendix C.2 for more details), and (iii) use a
relatively high sampling temperature. The single model, via the additional metadata we condition
upon, can generate solutions with diﬀerent languages, tags, and ratings. To make the most eﬀective
use of our samples we then apply ﬁltering (Section 4.5) and clustering (Section 4.6) to obtain a small
number of candidate submissions.

For problem tags and ratings conditioning, we picked random tags from the most popular 50 for the
model to condition on, and sampled ratings uniformly in the range of 800 to 3500 as these metadata
are not visible for new unseen problems in a competition. We found that conditioning on random
tags and ratings can improve performance, potentially by increasing diversity of the samples.

The optimal sampling temperature depends on the total number of samples (in general the more
samples, the higher the optimal temperature). However diﬀerent temperatures in a wide range do
not signiﬁcantly change the solve rates (Figure A5). We therefore use a ﬁxed sampling temperature
𝑇′ = 0.25 in all experiments that use tempering and GOLD, 𝑇′ = 0.12 when using tempering only,
and tune the sampling temperature separately otherwise.

We also experimented with top-𝑘(Fan et al., 2018) and nucleus sampling (Holtzman et al., 2019). As
seen in Figure A5, despite running exhaustive hyperparameter sweeps we did not observe signiﬁcant
performance improvements with these methods. We therefore use regular sampling with temperature
in our experiments. A few complete examples of model prompts and samples are provided in Appendix
F.

12

Competition-Level Code Generation with AlphaCode

4.5. Filtering

To accurately represent competitive programming contests and penalties, our formulation limits us
to just 10 submissions per problem no matter how many samples we draw. One powerful tool for
selecting these submissions is ﬁltering samples to only those that pass the example tests given in
the problem statement. Filtering removes approximately 99% of model samples, although the exact
amount depends on the problem and model, and ﬁltering can still leave tens of thousands of candidate
samples for many problems. Finding solutions that pass example tests is itself a diﬃcult problem, and
on approximately 10% of problems our models cannot ﬁnd a single such program. Indeed this easier
version of our setting is a classic program synthesis formulation, where the task is speciﬁed by a list
of given input/output pairs (Gulwani et al., 2017).

4.6. Clustering

Filtering using example tests can still leave thousands of candidate programs per problem. Randomly
picking from this pool wastes the limited submission budget on programs that are syntactically
diﬀerent but semantically equivalent. Semantically equivalent programs could be detected if we had
additional test inputs, by executing all remaining programs on these inputs and grouping programs
that produce the same outputs together into clusters. We could then avoid repeatedly picking from
the same clusters.

We trained a separate test input generation model, using the same architecture as our main models,
and initialised from the same GitHub pre-trained checkpoint. This model was trained to predict test
inputs from problem descriptions, using example, hidden, and generated test inputs as training data.
After training, this model was used to create new test inputs for unseen problems. Note that although
these created test inputs are not guaranteed to be valid, especially when problems have complex
constraints, imperfect and even invalid test inputs can still be useful for grouping sampled programs.

This learned test input generation model is diﬀerent from the mutation-based test generation process
used in Section 3.2.1 to augment our dataset. The latter requires correct solutions (which are not
available at test time) to ﬁlter out bad test cases.

After clustering on program behaviour we found that selecting one solution from each cluster from
largest to smallest performed best, perhaps because there are many ways solutions can be incorrect
while correct solutions tend to behave the same and therefore are grouped into larger clusters. If the
candidate solutions for a problem form less than 10 clusters (or more in the case of more than 10
submissions), after reaching the smallest cluster, we repeat from the ﬁrst cluster skipping samples
that have already been submitted.

5. Results

In this section we present experimental results that give insights into our model performance, and
evidence that guided our design decisions. We highlight the results obtained by evaluating on the
Codeforces platform (Section 5.1) and on CodeContests (Section 5.2), present a detailed study of
model performance on our dataset in Section 5.3, and conclude by comparing to published models in
the literature on the public APPS (Hendrycks et al., 2021) benchmark of programming problems in
Section 5.4. To ensure that our baseline models are comparable to past work we also compare our
decoder-only baseline directly to Chen et al. (2021) on the HumanEval benchmark in Appendix C.5.

13

Competition-Level Code Generation with AlphaCode

Contest ID
1591
1608
1613
1615
1617
1618
1619
1620
1622
1623
Average

Best
43.5%
43.6%
59.8%
60.5%
65.1%
32.2%
47.1%
54.0%
57.5%
20.6%
48.4%
Estimated
44.3%
46.3%
66.1%
62.4%
73.9%
52.2%
47.3%
63.3%
66.2%
20.9%
54.3%
Worst
74.5%
95.7%
75.0%
90.4%
82.3%
53.5%
88.1%
75.1%
81.6%
55.3%
77.2%

Table 4 | Estimated percent ranking of our system in 10 Codeforces competitions (lower is
better). For each contest, we show ranking using simulated time and incorrect submission penalties
(Estimated), as well as the best and worst possible rankings using minimum and maximum possible
time penalties as estimates, averaged over 3 evaluations. Percents are how many users performed
better than AlphaCode. Our system achieved an overall ranking of top 54.3% averaged across the 10
contests.

5.1. Codeforces competitions evaluation

Evaluating on programming competitions checks program correctness more thoroughly, compared to
evaluating on our dataset which has known weaknesses including false positives, accepting algorithmically ineﬃcient solutions, and handling problems with multiple acceptable outputs. Additionally,
evaluating in the real setting allows us to benchmark against the best performers on this task: human
competitors.

We evaluated our best system on all Codeforces competitions from 2021/12/01 to 2021/12/28 with
more than 5,000 participants per contest, a total of 10 competitions. The system was an ensemble
of 41B and 9B models with clustering, which performed best on our validation set but turned out
to be slightly worse than using the 41B model alone with clustering (see Appendix C.1 for more on
ensembling). For each contest, we simulated running AlphaCode live, generating samples for each
problem, ﬁltering with example tests,7 and then clustering to get candidate submissions. We submitted
these selected candidates to the Codeforces platform,8 and computed AlphaCode’s placement in each
contest. After the ﬁrst run, we repeated this procedure two more times to measure variance and
performance with more than 10 submissions. Sources of variance include problem distribution, model
training, sampling, ﬁltering, and clustering. See Appendix D for the exact evaluation procedure, and
Table A5 and Table A6 for full results.

Table 4 shows evaluation results across the 10 competitions. For each competition, we show the
estimated percentile ranking using a simulated penalty, and upper and lower bounds assuming zero
and maximum submission time penalties. The bounds represent how ranking depends on the number
of accelerators used to draw samples during competition. For the second and third runs, Table A6
shows the estimated percentile when not limiting to 10 submissions per problem (still taking into
account penalties for incorrect submission), which although not human-like does follow competition
rules. We found that the model still continued to solve problems when given more attempts, though
at a decreased rate. The model tends to solve the easier problems in competitions, but it does manage
to solve harder problems including one rated 1800.

Overall our system achieved an average ranking of top 54.3% limiting to 10 submissions per problem,
with an actual average of 2.4 submissions for each problem solved.9 When allowed more than 10
submissions per problem (the second and third evaluation), AlphaCode achieved a ranking of top

7For problems permitting multiple correct outputs, we change the example test outputs to be the most canonical, which
gives our approach a slight advantage in the evaluation. See Appendix D for more details.
8Submitted programs can be found on our 3 accounts on Codeforces: SelectorUnlimited, WaggleCollide, and AngularNumeric. Attention visualizations for these problems can be found here.
9Our estimated performance is closer to its upper bound than its lower bound, because human solutions (and our
solutions) are typically submitted early in the contest, especially for easier problems.

14

Competition-Level Code Generation with AlphaCode

Approach
Validation Set
Test Set
10@1k 10@10k 10@100k 10@1M 10@1k 10@10k 10@100k

9B
16.9%
22.6%
27.1%
30.1%
14.3%
21.5%
25.8%
41B
16.9%
23.9%
28.2%
31.8%
15.6%
23.2%
27.7%
41B + clustering
21.0%
26.2%
31.8%
34.2%
16.4%
25.4%
29.6%

Table 5 | Solve rates of our best systems on the validation set and test set .

48.8%, with an actual average of 28.8 submissions for each problem solved. Our 10 submissions per
problem result corresponds to an estimated Codeforces rating of 1238, which is within the top 28%
of users who have participated in a contest in the last 6 months (a small and selected subset of all
programmers). To the best of our knowledge, this is the ﬁrst time that a computer system has been
competitive with human participants in programming competitions.

5.2. CodeContests evaluation

As well as the Codeforces evaluation, we evaluated our model on the validation and test sets of
CodeContests. The test set is a superset of the competitions used in Section 5.1.10 The metrics on our
dataset are lower variance and easier to measure, since they do not involve submitting to an external
site. For CodeContests (both here and in Section 5.3), we focus on the two main metrics discussed
in Section 2.2:

• pass@k: The percentage of problems solved when we take 𝑘samples from the model for
each problem and submit all of them for evaluation on the hidden tests. If any solution in the
speciﬁed sample budget solves a problem, the problem is counted as solved. Therefore this
metric measures mostly the search aspect of the sampling process, and is used in Section 5.3.
• 10@k: The percentage of problems solved when we take 𝑘samples from the model for each
problem but can only submit 10 of them for evaluation on the hidden tests. This measures
factors including the ﬁltering process and how models behave at a very large number of samples.

The results are shown in Table 5. With up to a million samples per problem, we can solve 34.2% of
problems in our validation set; and with one hundred thousand samples, we solve 31.8% of problems
in our validation set, and 29.6% of problems in our test set. Because of the temporal split, no problem
in either set was seen by our model during training. Given the diﬃculty of these problems (since they
are problems given to the self-selected group of those who try competitive programming), this is a
substantial proportion of the dataset.

Diﬀerences in solve rates between the validation and test sets are caused by variation in problem
distributions (as the test set and validation set were collected in temporally disjoint periods), as well as
some overﬁtting. However, the diﬀerence in performance between the two sets remains limited. The
41B consistently outperforms the 9B model, and clustering consistently provides an improvement.

5.3. CodeContests ablations & results

This section contains results that support our design decisions described in Section 4. All results are
on the CodeContests validation set, with models ﬁne-tuned on the CodeContests training set and not
using clustering unless otherwise noted.

10Except one problem that does not have 5 test cases and is therefore not included in our test set.

15

Competition-Level Code Generation with AlphaCode

0.30

0.4

0.25

300M
1B
3B
9B
41B

300M
1B
3B
9B
41B

0.3

0.20

10@k

0.15

pass@k

0.2

0.10

0.1

0.05

0.00

0.0

100
101
102
103
104
105
106

100
101
102
103
104
105
106

Sample budget

Sample budget

(a) 10 attempts per problem
(b) Unlimited attempts per problem

Figure 6 | Solve rate scaling vs. number of samples. The solve rate scales approximately loglinearly with the number of samples, although this tapers oﬀslightly in the 10@k setting. The better,
larger-parameter models have higher scaling slopes in this log-linear plot.

5.3.1. Solve rates scale with respect to parameter count, compute, number of samples, and
dataset size

As would be expected, scaling up the number of model parameters or the size of the dataset greatly
improves model performance (see Figure A6 for scaling with dataset size). However, even when only
10 samples can be submitted, scaling up the total number of samples leads to massive improvements
in model solve rate.

Figure 6 shows how the model performance scales on the 10@𝑘and 𝑝𝑎𝑠𝑠@𝑘metrics with more
samples, i.e. as we increase 𝑘. The diﬀerence between the two metrics highlights the importance
of selecting which samples to submit. Figure 7 shows how performance scales with the amount of
compute used for training and for sampling. These scaling curves highlight a few interesting facts
about this problem domain and our models:

Solve rates scale log-linearly with more samples.
Both the 10@k and pass@k solve rates scale
approximately log-linearly with 𝑘, with the 10@k curve bending down slightly at high sample budgets.
The fact that sampling signiﬁcantly more than 10 still improves the 10@k solve rate shows how
important it is to suﬃciently explore the search space before committing to the ﬁnal 10 submissions
per problem. However, improving solve rate requires exponentially increasing amounts of samples
and the costs quickly become prohibitive.

Better models have higher slopes in the scaling curve.
Another observation from Figure 6 is that
larger models tend to have better model quality, reﬂected as better solve rate with the same number
of samples and higher slope in this log-linear scaling curve. Because of log-linear scaling, a better
model with a higher slope can reach the same solve rate with exponentially fewer samples than
worse models. This points to improving model quality as an eﬀective way to counter the exponential
explosion of sample budget required to reach a higher solve rate.

Solve rates scale log-linearly with more compute.
As shown in Figure 7(a), the solve rate also
scales approximately log-linearly with more training compute. Each point on the curves corresponds
to one model size. Figure 7(b) shows how solve rate scales with sampling compute, and highlights
that larger models take more compute to draw each sample, but they eventually outperform smaller

16

Competition-Level Code Generation with AlphaCode

0.30

0.30

10@1K
10@10K
10@100K
10@1M

0.25

300M
1B
3B
9B
41B

0.25

0.20

0.20

10@k

10@k

0.15

0.10

0.15

0.05

0.10

0.00

102
103
104

10
1
100
101
102
103
104
105
106

Training TPU-days

Sampling TPU-seconds per problem

(a) Solve Rate vs. Training Compute
(b) Solve Rate vs. Sampling Compute

Figure 7 | Solve rate scaling vs. Compute. The solve rate scales approximately log-linearly with the
training compute when we choose model sizes close to optimal for each compute allocation. Similarly,
as we increase the amount of compute we use for sampling the optimal model size increases.

Blocks
Seq. length
Hidden
Fan-Out
Samples /
Model
Enc.
Dec.
Enc.
Dec.
Size
Ratio
Params
TPU sec
10@10K

AlphaCode model
5
30
1536
768
1408
6
1.15B
4.74
17.3%
Decoder-only
40
2304
1408
6
1.17B
1.23
18.5%
Std MH attention
5
30
1536
768
1408
4.3
1.16B
0.37
17.0%

Table 6 | Architecture comparison. Architecture changes increase sampling speed without signiﬁcantly impacting the solve rate.

models even with the same sampling compute as the better quality of samples from the larger models
become the dominant factor for performance. These results present an interesting trade-oﬀbetween
how much of the available compute should be used to train a model compared to sampling from it.
Both ways of leveraging more compute demonstrate log-linear scaling.

5.3.2. Architecture changes to improve sampling speed

Because drawing more samples is important for improving performance, architecture changes that
increase sampling speed would also increase the overall solve rate within a certain compute budget.
Therefore, we made two architecture decisions: using (1) an encoder-decoder architecture with
asymmetric encoder and decoder structures and (2) the multi-query attention setup from Shazeer
(2019) which uses one shared attention head for keys and one for values each block.

To investigate the eﬀects of these decisions, we compared our base 1B parameter model against
the two alternatives that remove each of the changes. We pre-trained and ﬁne-tuned the standard
multi-head attention model in exactly the same way as our base 1B model. The decoder-only model
was trained with the same amount of compute. However, due to the signiﬁcantly longer decoder
sequence length (2304 tokens), with the same amount of training compute it consumes 50% more
loss tokens than training the encoder-decoder models.

Table 6 shows that our encoder-decoder model with multi-query attention signiﬁcantly improves the
sampling speed while keeping the sample quality at the same level as the more expensive alternatives.

17

Competition-Level Code Generation with AlphaCode

5.3.3. Choice of the pre-training dataset

Table 7 compares our base 1B model trained on our full GitHub dataset with equivalent models that
are pretrained on (1) the Python-only portion of GitHub, (2) the MassiveText generic text dataset
(Rae et al., 2021) which also includes a portion of GitHub or (3) not pre-trained at all. The pre-trained
models are then ﬁne-tuned and sampled in exactly the same way, except that the model pre-trained
on Python-only data is also ﬁne-tuned on Python-only data and only samples Python solutions.

As Table 7 shows, pre-training on the full GitHub dataset with all languages leads to signiﬁcantly
better results than pre-training either on Python alone, or on the MassiveText dataset that mostly
consists of natural language text. Any pre-training signiﬁcantly improves the results over training
from scratch on CodeContests.

Pre-training dataset
Solve rate
10@1K
10@10K
10@100K

No pre-training
4.5%
7.0%
10.5%
GitHub (Python only)
5.8%
11.1%
15.5%
MassiveText
9.7%
16.1%
21.2%
GitHub (all languages)
12.4%
17.3%
21.5%

Table 7 | Model solve rate with diﬀerent pre-training settings and datasets.

5.3.4. Model enhancements

As discussed in Section 4, we adopted training and model enhancements which signiﬁcantly improved
the solve rate relative to the standard encoder-decoder transformer setup. Table 8 presents the results
of a build-up ablation of the enhancements we added to AlphaCode, starting from the base setting
with no enhancements (beyond the multi-query attention change discussed in Section 5.3.2). We
added one new setting at a time, with the ﬁnal setting that corresponds to AlphaCode reported at the
bottom of the table. Each additional setting improves performance and combining the 5 enhancements
together increases the 10@100k solve rate from 15.2% to 24.1%, although the contribution depends
on the number of samples.

Solve rate
Fine-tuning setting
10@1K
10@10K
10@100K
10@1M

No Enhancements
6.7% (6.5-6.8)
10.4% (9.6-11.0)
15.2% (14.3-15.9)
19.6% (18.2-20.4)
+ MLM
6.6% (6.2-7.0)
12.5% (12.1-12.7)
17.0% (16.5-17.2)
20.7% (19.1-21.3)
+ Tempering
7.7% (7.2-8.5)
13.3% (12.5-13.8)
18.7% (18.0-19.2)
21.9% (20.7-22.6)
+ Tags and Ratings
6.8% (6.4-7.0)
13.7% (12.8-14.9)
19.3% (18.1-20.0)
22.4% (21.3-23.0)
+ Value
10.6% (9.8-11.1)
16.6% (16.4-16.9)
20.2% (19.6-20.7)
23.2% (21.7-23.9)
+ GOLD
12.4% (12.0-13.0)
17.3% (16.9-17.6)
21.5% (20.5-22.2)
24.2% (23.1-24.4)
+ Clustering
12.2% (10.8-13.4)
18.0% (17.3-18.8)
24.1% (23.2-25.0)
28.4% (27.5-29.3)

Table 8 | Build-up ablation for model enhancements. Eﬀect of each additional model enhancement
building up from No enhancements which is a plain ﬁne-tuned 1B encoder-decoder model trained with
the standard next token prediction loss. Numbers in parentheses represent 95% conﬁdence intervals.
For each setting we ﬁne-tuned and sampled from at least 3 diﬀerent models from the same pre-trained
checkpoint, and computed means and conﬁdence intervals using a combination of subsampling and
bootstrapping as discussed in Appendix A.3.

18

Competition-Level Code Generation with AlphaCode

% Problems with ≥1
Average 𝑝pass example tests
Average 𝑝pass example tests
Model
samples pass example tests
on all problems
on solved problems

300M
82.05%
0.39%
1.18%
1B
87.18%
0.59%
1.40%
3B
87.18%
0.49%
0.98%
9B
89.74%
0.76%
1.52%
41B
92.31%
0.73%
1.47%

Table 9 | Example test statistics. Example tests help us ﬁlter out more than 99% of model samples,
and as models get better with larger scales, they are more likely to ﬁnd samples that pass example
tests for more problems. One million samples were drawn per problem from each model.

5.3.5. Filtering & clustering

To solve problems within a realistic evaluation budget, we rely on ﬁltering and clustering to select a
small number of samples to evaluate from the large amount of model samples we generate.

Filtering using example tests.
Table 9 shows the percentage of model samples that pass example
tests and the percentage of problems where at least one sample passes example tests. Note that
these percentages are calculated based on the full set of samples, without ﬁrst ﬁltering out programs
that have syntax errors (see Section 6.2 for more on syntactic correctness of the samples). Overall
less than 1% of samples from our models pass example tests, though the percentage varies greatly
across problems, which means that ﬁltering using example tests removes more than 99% of the model
samples. On problems where our models do ﬁnd a correct solution, the fraction of samples that
pass example tests roughly doubles but still remains at a low level. The non-uniform distribution of
𝑝pass example tests across problems is highlighted more in Appendix C.4.

Another observation from Table 9 is that larger and better quality models produce samples more
likely to pass example tests, and pass example tests for signiﬁcantly more problems. With 106

samples, our largest 41B models can generate solutions that pass example tests for over 90% of
problems, a remarkable success as ﬁnding programs that satisfy I/O example constraints remains a
very challenging problem.

Clustering.
A solution has to pass hidden tests in addition to example tests, so we must further
select correct samples from those that pass all public tests. Filtering 99% of a million samples still
leaves thousands of samples per problem to select from. We cluster the remaining samples based on
their behaviour on generated test inputs, to make the most of the evaluation budget.

Figure 8 shows a comparison between (i) randomly picking model samples without ﬁltering, (ii)
ﬁltering and then randomly selecting from the ﬁltered samples, (iii) ﬁltering and then using clustering
to select samples, and (iv) allowing unlimited evaluation attempts, which gives us the upper bound
performance attainable with a perfect sample selection method. Filtering and clustering clearly enable
scaling, as otherwise the solve rate remains ﬂat. However there is still a large gap between them and
the theoretical upper bound.

5.4. Results on APPS

In addition to evaluating on Codeforces competitions and CodeContests, we performed evaluations
on the previously published APPS benchmark to directly compare to previous work. The APPS dataset
(Hendrycks et al., 2021) contains a total of 10,000 programming problems divided equally between
training and test sets. Because of missing information in the dataset, we could not apply our full

19

Competition-Level Code Generation with AlphaCode

0.35

0.30

pass@k
10@k with filtering + clustering
10@k with filtering
10@k no filtering

0.25

0.20

Solve rate

0.15

0.10

0.05

0.00

101
102
103
104
105
106

Sample budget

Figure 8 | Comparison of diﬀerent sample selection methods. We show random selection (“10@k
no ﬁltering”), ﬁltering using example tests (“10@k with ﬁltering”), clustering after ﬁltering (“10@k
with ﬁltering + clustering”), and perfect sample selection (“pass@k”).

method. We therefore followed the settings we used for pre-training on GitHub, and ﬁne-tuned our
pre-trained models on the APPS training set without using clustering, tags, ratings, value conditioning,
or prediction, and with sampling temperature 0.25 and nucleus sampling. Other settings were the
same as our main models.

Table 10 compares our model with existing large language models ﬁne-tuned on this dataset as
reported by Hendrycks et al. (2021), as well as the 1-shot performance of the Codex model reported
by Chen et al. (2021). A small 1B parameter model already outperforms the GPT-NEO baseline on
all diﬃculty levels, and outperforms Codex 12B on the interview and competition diﬃculty levels.
We highlight that AlphaCode still improves when increasing the number of samples per problem,
showing support for our claim of the importance of large scale sampling. Diﬀerences in performance
between APPS results and CodeContests could be attributed to dataset quality (e.g. the high APPS
false positive rate shown in Section 3.2.1), dataset size, missing components of AlphaCode, and
tuning for the problem distribution.

6. AlphaCode’s capabilities & limitations

We performed a detailed analysis of the capabilities and limitations of our models. In particular,
we ﬁnd that our models are not simply copying from the training set (Section 6.1) and our models
are sensitive to various changes in the problem descriptions and metadata used for conditioning
(Section 6.3 and 6.4), both of which indicate that we are not solving problems by exploiting obvious
weaknesses in the task structure.

We also analyze the characteristics of the solutions the model ﬁnds, for syntactic correctness, dead
code, and the types of problems it can solve (Section 6.2). We further show that using validation loss
as a proxy for model performance has several issues (Section 6.5). More analysis of our model and
approach are included in Appendix E, and an attention visualization as well as example problems
and solutions generated by the model can be found at https://alphacode.deepmind.com/. All
analysis results are reported without clustering unless otherwise noted.

20

---

*Source: arXiv:2203.07814 / Science*
