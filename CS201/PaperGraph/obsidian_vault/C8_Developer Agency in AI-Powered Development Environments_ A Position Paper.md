---
id: C8
title: "Developer Agency in AI-Powered Development Environments: A Position Paper"
domain: C
year: 2024
arxiv_id: "2501.02450"
confidence: verified
source: "arXiv:2501.02450"
node_type: paper
---

# Developer Agency in AI-Powered Development Environments: A Position Paper

**Domain**: [[domain_C|Human-AI Collaboration]] | **Year**: 2024 | **Confidence**: [x] verified


## Authors
[[author_Advait Sarkar|Advait Sarkar]]


## Keywords
- [[kw_developer agency|developer agency]]
- [[kw_AI development|AI development]]
- [[kw_over-reliance|over-reliance]]
- [[kw_autonomy|autonomy]]
- [[kw_CHI|CHI]]

## Abstract

(Abstract not available - see PDF content below)

## Paper Content

1

GCP: Guarded Collaborative Perception with
Spatial-Temporal Aware Malicious Agent Detection

Yihang Tao∗, Senkang Hu∗, Yue Hu, Haonan An, Hangcheng Cao, and Yuguang Fang, Fellow, IEEE

Fig. 1.
Illustration of security challenges and defense mechanisms
in CP. While CP systems are vulnerable to adversarial messages from
malicious agents, our proposed GCP framework provides comprehensive
protection through joint spatial-temporal consistency verification, effectively
safeguarding the system against various attack patterns.

Abstract—Collaborative perception significantly enhances autonomous driving safety by extending each vehicle’s perception range through message sharing among connected and
autonomous vehicles. Unfortunately, it is also vulnerable to
adversarial message attacks from malicious agents, resulting in
severe performance degradation. While existing defenses employ hypothesis-and-verification frameworks to detect malicious
agents based on single-shot outliers, they overlook temporal
message correlations, which can be circumvented by subtle yet
harmful perturbations in model input and output spaces. This
paper reveals a novel blind area confusion (BAC) attack that
compromises existing single-shot outlier-based detection methods.
As a countermeasure, we propose GCP, a Guarded Collaborative
Perception framework based on spatial-temporal aware malicious
agent detection, which maintains single-shot spatial consistency
through a confidence-scaled spatial concordance loss, while simultaneously examining temporal anomalies by reconstructing
historical bird’s eye view motion flows in low-confidence regions.
We also employ a joint spatial-temporal Benjamini-Hochberg test
to synthesize dual-domain anomaly results for reliable malicious
agent detection. Extensive experiments demonstrate GCP’s superior performance under diverse attack scenarios, achieving up to
34.69% improvements in AP@0.5 compared to the state-of-theart CP defense strategies under BAC attacks, while maintaining
consistent 5-8% improvements under other typical attacks. Code
will be released at https://github.com/yihangtao/GCP.git.

Index Terms—Connected and autonomous vehicle (CAV), collaborative perception, malicious agents, spatial-temporal detection.

arXiv:2501.02450v2  [cs.CV]  26 Apr 2026

I. INTRODUCTION
C
OLLABORATIVE perception (CP) supersedes singleagent perception by enabling information-sharing among
multiple connected and autonomous vehicles (CAVs), substantially enlarging a vehicle’s perception scope and accuracy
[1, 2, 3, 4, 5, 6, 7, 8]. The extended perception helps an ego
vehicle detect occluded objects that were originally difficult to
recognize with single-vehicle perception due to physical occlusions, thereby boosting the safety of autonomous driving. To
collaborate, one simple way for an ego CAV to receive helps is
to directly request early-stage raw data or late-stage detection
results from the neighboring CAVs, and then combine this
information with its own data to get the CP results. However,

Y. Tao, S. Hu, H. An, H. Cao, and Y. Fang are with Department of Computer
Science, City University of Hong Kong, Kowloon, Hong Kong. Email:
{yihang.tommy, senkang.forest, haonanan2-c}@my.cityu.edu.hk, {hangccao,
my.fang}@cityu.edu.hk. (* indicates equal contribution.)
Y. Hu is with the Department of Robotics, University of Michigan, Ann
Arbor, USA. Email: huyu@umich.edu.
This work was supported in part by the Hong Kong Innovation and
Technology Commission under InnoHK Project CIMDA, by the Hong Kong
SAR Government under the Global STEM Professorship, and by the Hong
Kong Jockey Club under JC STEM Lab of Smart City (Ref.: 2023-0108).

these methods are either bandwidth-consuming or vulnerable
to perception noise or malicious message attacks. The recent
development of deep learning has facilitated feature-level
fusion, where the collaborative CAVs send an intermediate
representation of deep neural network models to an ego CAV
for aggregation, enhancing the performance-bandwidth tradeoff in multi-agent perception.
Although CP has brought many benefits to CP systems, it
is inevitably attracting adversarial attacks due to the openness
of communication channels. While traditional authentication
methods (e.g., message and/or source authentication) can
verify message sources and data integrity, they cannot protect
against compromised legitimate agents who possess valid
credentials to share malicious messages. In an intermediate
fusion-based CP system, a malicious collaborator could send
an adversarial feature map with intricately crafted perturbation
to an ego agent, causing significant CP performance degradation after fusion. This is particularly dangerous because
the perturbed CP performance could drop far below the
single-agent perception, consequently resulting in catastrophic
driving decisions. Previous works have revealed diverse attacks
that can fool a CP system [9, 10, 11, 12]. For example,
Zhang et al.[12] introduced an online attack by optimizing a
perturbation on the attacker’s feature map in each LiDAR cycle
and reusing the perturbation over frames, which significantly

2

lowers the performance of CP systems.

concordance loss and an LSTM-AE-based temporal BEV
flow reconstruction. To the best of our knowledge, this
is the first work to protect CP systems from joint spatial
and temporal views.

• We conduct comprehensive experiments on diverse attack
scenarios with V2X-Sim [2] dataset. The results demonstrate that GCP achieves the state-of-the-art performance,
with up to 34.69% improvements in AP@0.5 compared to
the existing state-of-the-art defensive CP methods under
intense BAC attacks, while maintaining 5-8% advantages
under other adversarial attacks targeting at CP systems.

II. RELATED WORK

Existing CP defense methods are primarily unsupervised
outlier detection schemes based on the hypothesize-and-verify
paradigm. For instance, ROBOSAC [11] and CP-Guard [13]
employ sample consensus strategies to identify malicious
agents, while MADE [12] utilizes multi-test consistency
checks. Recently, CP-Guard+ [14] utilizes prior knowledge of
attack patterns for training, which accelerates the detection
speed yet imposes stronger assumption on the defender’s
knowledge. Furthermore, all these methods rely heavily on
spatial information within a single slot, ignoring temporal correlations across frames. Consequently, they remain vulnerable
to dynamic attacks that are spatially subtle but temporally
anomalous.

A. Collaborative Perception (CP)

To overcome the inherent limitations of single-agent perception, particularly restricted field-of-view (FoV) and occlusions,
collaborative perception (CP) has emerged as a promising
paradigm leveraging multi-agent fusion to enhance perception
accuracy [15, 16]. The evolution of CP systems has witnessed
various strategies, from early raw-data [1] and output-level
fusion [17], which faced challenges in communication overhead and information loss, to sophisticated intermediate-level
feature fusion. Notable advances include DiscoNet [18], which
employs a teacher-student framework to learn an optimized
collaboration graph, V2VNet [19], which leverages graph
neural networks for efficient aggregation, and When2com [20],
which learns to construct communication groups to decide
when to communicate. Where2comm [4] further advances the
field by introducing a confidence-aware attention mechanism
that simultaneously optimizes communication efficiency and
perception performance. Similarly, R-ACP [21] proposes a
robust task-oriented communication strategy that optimizes online self-calibration and efficient feature sharing by minimizing
the Age of Perceived Targets (AoPT) to ensure timely and
accurate perception. While improving performance-bandwidth
trade-offs, these developments exposed critical vulnerabilities
in CP systems regarding robustness against adversarial attacks,
which is the main focus of this paper.

B. Adversarial CP

We strongly believe incorporating temporal knowledge is
crucial for CP defense based on two key insights. ❶First,
adversarial attacks in real-world scenarios often exhibit distinct
temporal patterns. When malicious agents inject perturbations
intermittently to maintain stealth, these attacks manifest as
anomalous variations in the temporal domain of CP results.
For instance, both ROBOSAC [11] and MADE [12] observe
that attackers typically alternate between sending malicious
and benign messages across time slots to avoid detection.
This temporal characteristic provides an additional verification dimension: beyond examining spatial consistency in the
current frame, we can leverage historical clean messages from
collaborators as reliable references to identify suspicious temporal deviations. ❷Second, even when attackers continuously
inject perturbations, their impact on CP results inevitably
creates distinctive temporal patterns that differ from normal
CP behaviors. These patterns manifest in various aspects, such
as unnatural object motion trajectories, inconsistent detection
confidence variations, or abrupt changes in spatial feature
distributions across frames. Such temporal anomalies, while
potentially subtle in individual frames, become more apparent
when analyzed over extended long sequences.
To address the aforementioned challenges, in this paper, we
first reveal a novel adversarial attack targeted at CP, namely,
the blind area confusion (BAC) attack, which generates subtle
and targeted perturbation in an ego CAV’s less confident areas
to bypass the single-shot outlier-based detection methods.
Besides, to overcome the limitations of previous malicious
agent detection methods, we propose GCP, a defensive CP
system against adversarial attackers based on knowledge from
both the spatial and temporal domains. Specifically, GCP leverages a confidence-scaled spatial concordance loss and Longshort-term-memory autoencoder (LSTM-AE)-based BEV flow
reconstruction to check the spatial and temporal consistency
jointly. To sum up, our main contributions are three-fold:

• We reveal a novel attack, dubbed as blind area confusion
(BAC) attack, which is targeted at CP systems by generating subtle and dangerous perturbation in an an ego
CAV’s less confident areas. The attack can significantly
degrade existing state-of-the-art single-shot outlier-based
malicious agent detection methods for CP systems.

• We develop GCP, a novel spatial-temporal aware CP
defense framework, under which malicious agents can be
jointly detected by utilizing a confidence-scaled spatial

The vulnerabilities in CP systems can be broadly categorized into systematic and adversarial challenges, each presenting unique threats to system reliability and safety. On the
system front, inherent issues such as communication delays,
synchronization problems, and localization errors have been
addressed by recent works. CoBEVFlow [22] introduced an
asynchrony-robust CP system to compensate for relative motions to align perceptual information across different temporal
states, while CoAlign [23] developed a comprehensive framework specifically targeting at unknown pose errors through
adaptive feature alignment. However, beyond these system
challenges lies a more insidious threat, namely, adversarial
vulnerabilities introduced by malicious agents within the collaborative system. These adversaries can compromise system
integrity by injecting subtle adversarial noise into shared
intermediate representations, potentially causing catastrophic
failures in critical scenarios. Initial investigations by Tu et al.

3

in Eq. 1: i) the times ti−1, ti−2, · · · , ti−k are not necessary
to be equally distributed, and the time intervals between two
consecutive times can be irregular. ii) When the length of
historical times k equals 0, the i-th CAV outputs the CP
results without referring to the temporal contexts of messages
of collaborative CAVs, which degrades to the settings used in
most existing works like ROBOSAC [11] and MADE [12].

[10] demonstrated how untargeted adversarial attacks could
compromise detection accuracy in intermediate-fusion CP systems through feature perturbation. Zhang et al. [9] advanced
this research by incorporating sophisticated perturbation initialization and feature map masking techniques for more
realistic, targeted attacks in real-world settings. Nevertheless,
these attack methods lack sophistication in attack region
selection and output perturbation constraints, making them
potentially detectable by conventional defense mechanisms
while highlighting the need for more robust security measures.

However, this collaborative process is vulnerable when
malicious agents Mti ⊆N exist. These agents can encode
raw observations into initial feature maps and then generate adversarial perturbations δ by maximizing the distance between
CP results and ground truth (GT):

C. Defensive CP

arg max
δ
L(Ytj
δ , Ytj
gt),
s.t.
∥δ∥≤∆i,
(2)

where L is the perception loss function, Ytj
δ and Ytj
gt denote
the perturbed CP output and the ground truth at time tj,
respectively, and ∆i is the maximum allowable perturbation
amplitude. Finally, they transmit these perturbed features to
the victim to degrade perception accuracy.

B. Adversarial Threat Model

We adopt the same threat model assumptions as in previous
works [11, 12, 13]. Specifically, we consider an internal
attacker who has compromised at least one legitimate CAV and
gained white-box access to the shared CP model architecture,
parameters, and the feature maps transmitted among vehicles.
This access is inherent to the compromised vehicle’s legitimate
participation in the CP system. The attacker aims to mislead
the victim’s perception by sending malicious feature maps
while maintaining stealthiness. We also assume the attacker
adheres to the system’s communication protocol regarding
timing and does not manipulate artificial delays or jitter,
focusing instead on manipulating the semantic content of the
transmitted features.

In response to emerging threats, the research community
has developed various defensive strategies, primarily focusing
on output-level malicious agent detection through hypothesisand-verification frameworks. ROBOSAC [11] and CP-Guard
[13] employ sample consensus strategies to systematically
identify and exclude potentially malicious agents. MADE
[12] utilizes a multi-test framework leveraging both match
loss and collaborative reconstruction loss to ensure robust
consistency. Zhang et al. [9] further utilize occupancy maps for
discrepancy detection. While these unsupervised outlier-based
detection methods show promise, they remain vulnerable to
sophisticated attacks with subtle perturbations. Recently, CPGuard+ [14] trains a feature classifier by pre-simulating attack
patterns (CP-GuardBench). However, this approach relies on
the strong assumption that the defender possesses prior knowledge of the attacker’s perturbation mechanism to construct the
training dataset. This contrasts with other methods that operate
without such prior knowledge or access to simulated attack
data. Furthermore, existing approaches generally overlook
the temporal information, limiting their effectiveness against
dynamic threats. To address these limitations, we propose a
comprehensive defense framework that considers both spatial
and temporal aspects, offering a more robust solution to CP
security challenges.

C. Blind Area Confusion Attack

III. ATTACK METHODOLOGY

A. Model of CP

Consider a scenario with N CAVs, where the CAV set is
denoted as N. CAVs exchange collaboration messages with
each other, and each CAV can maintain up to K historical
messages from its collaborators. For the i-th CAV, we denote
its raw observation at time ti as Oti
i , and use Φi
enc to represent
its pre-trained feature encoder that generates intermediate
feature map Fti
i
= Φi
enc(Oti
i ). The collaboration message
transmitted from the m-th CAV to the i-th CAV at time ti is
denoted as Fti
m→i. Based on the received messages at current
timestamp and historical k (0 ≤k ≤K) timestamps, the i-th
CAV uses a feature aggregator f i
agg(·) to fuse these feature
maps and adopts a feature decoder Φi
dec(·) to get the final CP
output, which is expressed as:

Yti
i = Φi
dec

f i
agg

{Fti
m→i, Fti−1
m→i, · · · , Fti−k
m→i}N
m=1

, (1)

where Yti
i is the CP result of the i-th CAV at time ti. There
are two important notes in terms of the formulation described

For previous outlier-based methods like ROBOSAC [11]
and MADE [12], the core idea is to check the spatial consistency between ego CAV and CP outcomes. However, they only
clip the perturbation at the input level, and the output detection
errors are significant and randomly distributed, which can be
easily detected by outlier-based defense methods. Intuitively,
we want to emphasize that a good attack targeted at the CP
system should satisfy two conditions: (i) both the input and
output perturbation should not exceed a certain level; (ii) the
output perturbation should be more distributed in the regions
where the victim vehicle (i.e., agent or CAV) is less sensitive
so that the victim vehicle can hardly discern whether the
unseen detections could benefit from collaboration or just fake
ones. Following the above ideas, we design a novel blind area
confusion (BAC) attack as shown in Figure 2. The steps are
elaborated below.
Message Request. Following our threat model, a malicious
CAV i can participate in the CP system before being identified.
During this initial phase, it masquerades as a benign agent
to establish communication with the victim’s ego vehicle.

4

blind spots. The malicious CAV first generates two types
of perception results: single perception Ytk
s
= Φi
dec
 
Ftk
e→i


using only the victim’s transmitted features, and CP Ytk
c
=
Φi
dec
 
f i
agg
 
Ftk
i→i, Ftk
e→i

using both local and the victim’s
features. Let Ytk
s ∩Ytk
c
denote the matched bounding boxes
between single and CP results. Based on this matching, the
system partitions all detections into two categories: victimonly detections Ytk
vic = Ytk
s
(marked in purple), and nonvictim detections Ytk
nvic = Ytk
c \ (Ytk
s ∩Ytk
c ) (marked in
blue). This differential detection process reveals the spatial
distribution of the victim’s unique detections, providing crucial
insights into its perception strengths and potential blind spots,
which form the foundation for subsequent targeted perturbation generation.

Fig. 2. Overview of the proposed blind area confusion (BAC) attack. The
malicious agent first establishes communication with the victim ego CAV to
obtain collaborative messages, then infers the victim’s blind regions through
differential detection analysis and region segmentation. Finally, it generates
adversarial perturbations guided by the inferred confidence mask to confuse
the victim’s perception defense system.

Algorithm 1 Blind Region Segmentation (BRS)

Blind Region Segmentation. The differential detection map
is processed through an adaptive region growing algorithm
to partition the BEV detection map into confident area (CA)
and blind area (BA), as shown in Algorithm 1. The algorithm first uses Cluster() to determine the victim grid e by
finding the grid point with the minimum total distance to
all victim-detected objects, establishing a perception-centric
coordinate system. It then initializes seed grids from Ytk
vic
as CA seeds (value 1) and Ytk
nvic as BA seeds (value -1).
When Ytk
nvic is empty, which occurs in limited perception
range scenarios, the grid farthest from e is selected as the BA
seed to reflect natural perception degradation with distance.
The region growing process utilizes two priority queues (Qca,
Qba) to manage confident and blind area expansion. The
GetNeighbors() function implements an adaptive neighbor
selection mechanism:

Input: Mtk
c
(initial confidence mask), Ytk
vic (victim detections),
Ytk
nvic (unseen detections), Dtk
i
(BEV detection map)
Output: Mtk
c
(binary confidence mask)
1: procedure BRS(Dtk
i )
2:
e ←Cluster(Ytk
vic)
▷Victim Grid
3:
if Ytk
nvic = ∅then
4:
Ytk
nvic ←arg max

g∈Dtk
i
Dist(g, e)


,
(3)

Ks =

Kbase · exp

−γd · Dist(s, e)

Dnorm

where Dist(s, e) computes the Euclidean distance between
grid s and victim grid e, Dnorm =
√

5:
end if
6:
Mtk
c {Dtk
i {Ytk
vic}} ←1; Mtk
c {Dtk
i {Ytk
nvic}} ←−1
7:
Qca, Qba ←Queue{m}, ∀Mtk
c {m} = 1 or −1
8:
while Qca ̸= ∅or Qba ̸= ∅do
▷Region Growing
9:
for Q ∈{Qca, Qba} and Q ̸= ∅do
10:
s ←Q.Pop()
11:
for j ∈GetNeighbors(s, e) and Mtk
c {j} = 0 do
12:
Mtk
c {j} ←[Q = Qca]?1 : −1
13:
Q.Append(j)
14:
end for
15:
end for
16:
end while
17:
Mtk
c {m} ←0, ∀Mtk
c {m} = −1
▷Binary Mask
18:
return Mtk
c
19: end procedure

H2 + W 2 is the
normalization factor based on BEV detection map dimensions
(height H and width W), Kbase = 6 establishes a hexagonallike growth pattern, and γd = 0.3 controls the decay rate.
This distance-adaptive design ensures denser expansion near
the victim and sparser expansion in distant regions, naturally
modeling spatial perception reliability [24]. The expansion
proceeds simultaneously via queue operations. Initialized by
Queue{m}, priority queues Qca and Qba drive breadth-first
growth. Grids are iteratively processed using Queue.Pop(),
where unassigned neighbors inherit the label (1 for CA, -1 for
BA) and are added via Queue.Append(). Finally, converting
BA labels to 0 generates the binary mask Mtk
c , which effectively delineates the victim’s perception boundaries.

Perturbation Optimization. BAC perturbation optimization aims to make the perturbation mostly distributed in the
victim-blind area of the BEV detection map while adaptively
controlling the perturbation magnitude. This optimization
problem can be formulated as:

arg max
δbsi
L(Ytk
δ ⊙Wδ, Ytk
gt ⊙Wδ),

s.t.
Wδ = wδMδ + Sδ,

(4)

Sδ = 1 −σ
 
|Ytk
δ −Ytk
gt| −∆o

,

Specifically, the malicious CAV requests feature messages
Ftk
e→i from the victim’s ego agent at timestamp tk. These
messages contain valuable information about the victim’s
perception capabilities and limitations, which will be exploited
in subsequent attack stages. Note that in our low frame rate
setting (e.g., 10 FPS), the attack generation and feature fusion
can be completed within the same frame, eliminating the need
for temporal prediction compensation that would be required
in high FPS scenarios.
Differential Detection. After obtaining a victim’s messages, the malicious CAV i performs a comparative analysis
between independent and CP results to infer the victim’s

∥δbsi∥≤∆i,

5

Fig. 3.
Overview of the proposed GCP framework. GCP performs joint spatial-temporal consistency verification through two key components: (1) a
confidence-scaled spatial concordance loss that adaptively evaluates detection consistency, and (2) an LSTM-AE-based temporal BEV flow reconstruction that
captures motion patterns in CP.

scores for each grid cell in the BEV detection map, evaluating
the consistency between the ego CAV’s observations and
messages from the i-th neighboring CAV at the current time
slot. Second, it performs temporal verification by analyzing the
past K frames from both the ego CAV and the i-th neighboring
CAV to generate a BEV flow map for low-confidence detections, employing an LSTM-AE-based temporal reconstruction
to verify motion consistency. The framework culminates in a
comprehensive spatial-temporal multi-test that combines both
consistency metrics to make final decisions on malicious agent
detection. The following sections (IV-B to IV-D) provide the
detailed descriptions of these core components.

B. Confidence-Scaled Spatial Concordance Loss

where L(·) denotes the optimization loss function, Mδ =
1 −Mtk
c {Dtk
i {Ytk
δ }} is the inverted confidence mask, σ(·)
is the sigmoid activation function, | · | computes element-wise
absolute values, ∆i bounds the input perturbation magnitude,
∆o bounds the output perturbation magnitude, and wδ is a
positive weighting parameter.
This optimization formulation incorporates several innovative design principles for effective and stealthy attacks. First,
instead of a simple loss function, it employs a weighted
loss function where ⊙denotes element-wise multiplication.
The weight Wδ is carefully designed to guide the spatial
distribution of perturbations, ensuring more targeted attacks.
Second, the optimization adopts a dual-weight mechanism: the
spatial guidance weight Mδ directs perturbations towards the
victim’s blind areas, while the adaptive suppression weight
Sδ automatically reduces weights when output perturbations
become too large. The adaptive suppression is implemented
through a sigmoid function, which provides smooth transitions
when the prediction deviates from the ground truth beyond the
threshold ∆o. This design ensures that the attack remains effective while avoiding generating easily detectable anomalies.
Additionally, the input perturbation magnitude is constrained
by ∆i to maintain physical feasibility and attack stealthiness.
After the optimization, the malicious CAV i incorporates the
optimized perturbation δbsi into its intermediate BEV feature
before transmission to the victim CAV.

IV. GCP FRAMEWORK

A. Overall Architecture

Existing spatial consistency checking methods typically
compare the difference between the ego CAV’s perception
and CP without considering the ego CAV’s varying perception
reliability across different spatial regions. This indiscrimination could result in mistaking true blindspot detection
complemented by other CAVs as malicious signals. This
occurs because the ego CAV’s perception reliability naturally
decreases in occluded areas and at sensor range boundaries
[24]. When other CAVs provide valid detections in these lowreliability regions, traditional methods may incorrectly flag
them as inconsistencies by failing to account for the ego
CAV’s spatially varying perception capabilities. To address
this limitation, we propose a novel confidence-scaled spatial
concordance loss (CSCLoss) that incorporates the ego CAV’s
detection confidence when checking messages from the i-th
CAV at time tk. Let Ytk
e
denote the ego CAV’s detected
bounding boxes and Ytk
e,i represent the collaboratively detected bounding boxes at timestamp tk. We first construct a
weighted bipartite graph between these two sets, where edges
represent matching costs based on classification confidence

In this paper, we propose a robust framework to guard
CP systems through spatial-temporal aware malicious agent
detection. As illustrated in Fig. 3, GCP operates through a dualdomain verification process. First, it computes a confidencescaled spatial concordance loss by estimating confidence

6

and intersection-over-union (IoU) scores. Since |Ytk
e | may
differ from |Ytk
e,i|, we pad the smaller set with empty boxes
to ensure the one-to-one matching. The optimal matching
O{Ytk
e , Ytk
e,i}N
n=1 is then obtained using the Kuhn-Munkres
algorithm [25]. To compute the CSCLoss, we first estimate a
spatial confidence map Ctk
e using the ego CAV’s feature map
Fti
e→e:
Ctk
e = Φconf
 
Fti
e→e

∈[0, 1]H×W ,
(5)

where Φconf(·) is a detection decoder that generates confidence scores for each grid cell in the BEV map [4]. The
CSCLoss is then calculated by combining the optimal matching and confidence map:

M

Ytk(j)
e
, Ytk(j)
e,i
; c

· Ctk(j)
e

X

Lcsc
 
Ytk
e , Ytk
e,i

=
X

|Ctk
e |
,

c∈C

j∈O

(6)
where Ctk(j)
e
is the confidence score for the grid cell containing bounding box Ytk(j)
e
, c ∈C represents a prediction class,
M(·) computes the matching cost between two boxes:

M (y1, y2; c) = ReLU (p1 −p2) + ϕ (1 −IoU (z1, z2)) , (7)

where p1 and p2 are the class c posterior probabilities for boxes
y1 and y2, respectively, z1 and z2 are their spatial coordinates,
and ϕ is a weighting coefficient. Typically, when temporal
context is not applicable, a spatial anomaly can be assumed
to be detected when Lcsc exceeds a threshold α.

Algorithm 2, the matching process is a chain-based process,
which means that we first find the best-matched bounding
boxes Btk−1
e,i
in Ytk−1
e,i
for Otk
e,i, then we keep searching for
the best matched bounding boxes Btk−2
e,i
in Ytk−2
e,i
for Btk−1
e,i ,
until the K-th frame has been matched. During this process,
not all the bounding boxes in Otk
e,i in Ytk
e,i can be matched up
to the K-th frame, we call the successfully matched bounding
boxes as the candidate BEV flow Itk
c . LSTM-AE is used to
reconstruct it for further temporal consistency check. As for
the unmatched BEV flow Itk
u , they will result in additional
time anomaly penalties. To maintain computational efficiency,
we cache the historical matching chains for each tracked flow.
This significantly reduces the computational complexity as
previously established matches can be directly reused without
re-computation, making the chain matching process highly
efficient in practice. Note that consecutive K frames are not
always completely available for chain matching. There are two
cases when the frame cache is not enough for chain matching.
❶Case 1: The ego CAV may have not collected up to K
frames of neighboring CAVs at the early stage. In this case,
we only use spatial consistency to check the malicious agent
until the cached messages are up to K frames.
❷Case 2: Certain frames of neighboring CAVs have been
identified as malicious messages and are thereby discarded.
In this case, we use Kalman Filter (KF) [26] for interpolation
(Please refer to Appendix A). We also set a maximum consecutive interpolation limit of L to avoid the cumulative error.
Once the consecutive interpolation frames exceed the limit L,
all the cached frames will be refreshed.

C. Low-Confidence BEV Flow Matching

D. Temporal BEV Flow Reconstruction

For the generated candidate BEV flow, we further analyze
their temporal characteristics based on the current and past
K-frame messages from the i-th CAV. As shown in Fig 3,
there are three key components of BEV flow reconstruction:
LSTM encoder, LSTM decoder, and temporal reconstruction
loss estimator.
LSTM encoder. Each candidate BEV flow itk
c ∈Itk
c can be
represented as itk
c = [ok−K, ok−K+1, · · · , ok]⊤∈R(K+1)×8,
the LSTM encoder further encodes the high-dimensional input
sequence into a low-dimensional hidden representation using
the following equation:

Hk = σo ([Hk−1, ok]) ⊙tanh(Ck),
(9)

where σo(·) = σ(wo(·) + bo) represents the output gate with
activation function σ(·), weight wo, and bias bo. Hk−1 and ok
represent the concatenation of the hidden state and the current
input, respectively. Ck represents the input gate calculated by:

Bird’s Eye View (BEV) flow [22] represents the consecutive motion vectors of detected objects across consecutive
frames in the top-down perspective. By observing the flow of
bounding boxes in the BEV space over time, we can capture
the temporal dynamics and motion patterns of objects in the
scene. For computational efficiency, we only perform temporal
matching on two specific types of low-confidence BEV flows:
(i) those with low detection scores from ego CAV’s own view,
and (ii) ego CAV’s unseen detections that the collaborative
agents complement. While temporal matching could be applied
to all detected boxes, focusing on these low-confidence cases
is particularly crucial as it is difficult for an ego CAV to
judge whether they represent true objects or perturbations
caused by malicious agents merely through single-shot spatial
checks, especially when the added perturbation is subtle. To
address this challenge, we utilize temporal characteristics to
analyze their anomaly patterns. We first match the detected
low-confidence bounding boxes of Ytk
e,i to those in the past
K frames Ytk−1
e,i
, Ytk−K+1
e,i
, · · · , Ytk−K
e,i
. In the BEV detection
map, each detected bounding box can be represented as a set
of 4 corner points:

ˆCk = tanh(wc[Hk−1, ok] + bc),
(10)

oj = [x1
j, y1
j , · · · , x4
j, y4
j ]⊤∈R8, ∀oj ∈Ytk
e,i,
(8)

Ck = σf ([Hk−1, ok]) ⊙Ck−1 + σg ([Hk−1, ok]) ⊙ˆCk, (11)

where σf(·) is the forget gate, σg(·), ˆCk, Ck represent the
input gate. The output vector will be repeated K + 1 times to
yield the final encoded feature vector:

where {(xk
j , yk
j )}4
k=1 represents the locations of 4 corners of
bounding box oj in BEV detection map. Given the currentframe low-confidence bounding boxes set Otk
e,i ∈Ytk
e,i, the
ego CAV will generate BEV flow Itk
e,i by iteratively matching
the bounding boxes in each historical frame. As shown in

Hk = Φlstm
enc {itk
c } ∈R1×M,
(12)

7

K+1 times
z
}|
{
Hk ⊕Hk ⊕. . . ⊕Hk ∈R(K+1)×M,
(13)

Hbf =

where M is the latent encoded feature dimension, Φlstm
enc {·}
is the LSTM encoding network model containing M computation units following Eq. 9 to Eq. 11, and ⊕represents the
concatenation operation along the horizontal dimension.
LSTM decoder. The LSTM Decoder consists of a 3-layer
network with K LSTM cell units. Each LSTM cell processes
each R1×M encoded feature. These LSTM units generate a
R(K+1)×M output vector learned from the encoded feature,
which is further multiplied with a RM×8 vector output by
a TimeDistributed (TD) layer [27]. The TimeDistributed layer
maintains the temporal structure by applying a fully connected
layer to each time step output. Finally, the LSTM decoder
generates a reconstructed vector ˆitk
c with the same size as the
input vector following:

ˆitk
c = Φlstm
dec {Hbf} · TD{Hbf} ∈R(K+1)×8,
(14)

Fig. 4. Architecture of LSTM-AE-based BEV flow reconstruction. The
input BEV flow vector consists of 8-dimensional features representing corner
points of detected objects. The encoded latent features are repeated K + 1
times before decoding, followed by a TimeDistributed layer for temporalaware reconstruction of object motion patterns.

where Φlstm
dec {·} is the LSTM decoding network model, TD{·}
represents the TimeDistributed Layer funciton.
Temporal reconstruction loss estimator. To evaluate the
loss between the input vector and output vector, we use Mean
Absolute Error (MAE) as a metric, given by:

K
X

Ltr(itk
c ,ˆitk
c ) = 1

K

i=1
|ok−i −ˆok−i|,
(15)

hypothesis test H = {H0, H1}, where H0 : LST ∼Pnormal
represents the null hypothesis that the combined score follows
the distribution of normal agents, and H1 : LST ≁Pnormal
represents the alternative hypothesis. Since this distribution is
typically intractable, we compute conformal p-values using the
calibration set S:

ˆp = 1 + |{s ∈S : s ≥LST }|

1 + |S|
,
(18)

where ˆok−i ∈ˆitk
c
is the reconstructed BEV flow vector at
time tk−i. The final temporal anomaly score is the sum of the
candidate BEV flow reconstruction loss and the unmatched
BEV flow penalty, given by:

Lta =
X

Ltr(itk
c ,ˆitk
c ) + κp|Itk
u |,
(16)

i
tk
c ∈I
tk
c

where | · | represents the cardinality of the set. Following the
BH procedure, we sort the p-values in ascending order ˆp(1) ≤
ˆp(2) ≤... ≤ˆp(m) for m hypothesis tests. Let j be the largest
index satisfying:

where κp is a constant penalty coefficient for the unmatched
BEV flow.

ˆp(j) ≤j

mαbh,
(19)

E. Joint Spatial-Temporal Benjamini-Hochberg Test

Then, agent i is classified as malicious when its p-value is
less than or equal to ˆp(j), where αbh controls the desired false
detection rate.

V. EXPERIMENTS

A. Experimental Setup

To jointly utilize spatial and temporal anomaly results for
malicious agent detection, we employ the Benjamini-Hochberg
(BH) procedure [12, 28]. The BH procedure effectively controls the False Discovery Rate (FDR), the expected proportion
of false positives among all rejected null hypotheses, which
is crucial in CP where false accusations can severely impact
system reliability. For each CAV i under inspection, we first
compute a weighted combination of spatial and temporal
scores:
LST = ωsLcsc + ωtLta,
(17)

Datasets. We conduct experiments using two data sets,
namely, V2X-Sim dataset [2] and V2X-Flow dataset. V2XSim dataset is used for training and evaluating different CP
models and CP defense methods, while V2X-Flow dataset is
used for pre-training LSTM-AE in our GCP.
❶V2X-Sim Dataset. V2X-Sim [2] is a simulated dataset
generated by the CARLA simulator [29]. The dataset contains
10,000 frames of synchronized multi-view data captured from
6 different connected agents (5 vehicles and 1 RSU), including
LiDAR point clouds (32-beam, 70m range) and RGB images,
along with 501,000 annotated 3D bounding boxes containing
object attributes. The data is split into training, validation, and
test sets with a ratio of 8:1:1.

where ωs and ωt are learnable weights. The BH procedure
controls FDR through a step-up process that adapts its rejection threshold based on the distribution of observed p-values.
Unlike traditional single hypothesis testing, the BH procedure
maintains FDR control at level αbh even under arbitrary
dependencies between tests, making it particularly suitable for
CP where agents’ behaviors may be correlated due to shared
environmental factors or coordinated attacks. We formulate the

8

Algorithm 2 Chain-based BEV Flow Matching (BFM)

haviors spread through the network over time, similar to
virus propagation. The evolution of the number of malicious
messages Nt follows:

dNt

dt = γNt(1 −
Nt
λNaT ),
(21)

Input: Otk
e,i (low-confidence detections), {Y
tk−j
e,i
}K
j=1 (cached detections)
Output: Itk
c , Itk
u (candidate and unmatched BEV flows)
1: procedure BFM(Otk
e,i)
2:
Itk
c , Itk
u ←{}, {}
▷BEV Flow Initialization
3:
if Otk
e,i = ∅then return Itk
c , Itk
u
4:
end if
5:
for oj ∈Otk
e,i do
6:
ij ←[oj], curr ←oj, matched ←True
7:
for k ←K −1 down to 0 do
▷Chain Match
8:
bmax ←arg max

b∈Ytk
e,i
IoU(b, curr)

9:
cost ←ReLU(pcurr −pb) + ϕ(1 −IoU(b, curr))
10:
if cost < τ then
11:
ij ←ij ⊕bmax, curr ←bmax
12:
else
13:
matched ←False
14:
break
15:
end if
16:
end for
17:
if matched then
▷BEV Flow Concatenation
18:
Itk
c ←Itk
c ∪{ij}
19:
else
20:
Itk
u ←Itk
u ∪{ij}
21:
end if
22:
end for
23:
return Itk
c , Itk
u
24: end procedure

❷V2X-Flow Dataset. To facilite the pre-training of LSTMAE used in our GCP, we construct a V2X-Flow dataset based
on V2X-Sim. We separate the annotations of the V2X-Sim
dataset to generate the BEV flow of each bounding box using
the chain-based BEV flow matching method mentioned in
Section IV-C.
Attack Settings. To simulate realistic stealthy attacks, we
model temporal patterns where malicious agents attack intermittently. Given Na total agents, m malicious agents, and time
horizon T, we define the attack ratio λ as the proportion of malicious messages in the total traffic. We evaluate three modes
where the total malicious load λNaT is distributed among
m attackers via a truncated normal distribution N( λNaT

m
, σ2),
reflecting varying attack capabilities:

For adversarial perturbation generation, we evaluate our
method against three representative white-box attacks: Projected Gradient Descent (PGD) [32], Carlini & Wagner
(C&W) [33], and Basic Iterative Method (BIM) (Please refer
to Appendix B for implementation details). Moreover, we
introduce our proposed BAC attack to exploit the vulnerabilities in CP systems specifically. To maintain real-time
attack capability, the BAC mask generation adopts a slow
update strategy without requiring per-frame updates. In our
experiments, we set the mask update rate to 0.5 FPS.
Implementation Details. Our GCP is implemented with PyTorch. Each agent’s locally captured LiDAR or camera data
is first encoded into a BEV feature map with size 16 × 16
and 512 channels. The encoder network adopts a ResNetstyle architecture with 5 layers of feature encoding, with
output feature dimensions progressively increasing from 32
to 512 channels while spatial dimensions decrease from 256
to 16. The decoder consists of three convolutional blocks with
channel dimensions progressively decreasing from 512 to 64.
The detection head follows a two-branch design (classification and regression) with standard convolutional layers, batch
normalization, and ReLU activation. It uses an anchor-based
detection decoder [34] with multiple anchor sizes per location.
The fusion method is V2VNet [19]. The LSTM-AE model
consists of 2 LSTM layers with hidden dimension 32. We train
the model using Adam optimizer (lr=0.001) with the default
history sequence length 5. All experiments are conducted on a
server with 2 Intel(R) Xeon(R) Silver 4410Y CPUs, a single
NVIDIA RTX A5000 GPU, and 512 GB RAM.
Baselines and Evaluation Metrics. We compare our method
with state-of-the-art defenses ROBOSAC [11], MADE [12],
and CP-Guard [13]. We also introduce a simple Gated Late
Fusion as a baseline, which aggregates proposal boxes from all
connected agents and filters them using a consistency voting
mechanism. Reference settings include Upper-bound (clean
CP), Lower-bound (ego-only), and No Defense. Evaluation
metrics include Average Precision (AP@0.5, AP@0.7) for
accuracy and Frames Per Second (FPS) for efficiency.

B. Quantitative Results

❶Random attack process (R-mode) [11, 12]. This mode
follows a uniform random distribution in which malicious
messages are randomly distributed over the time period T,
with λNaT total malicious messages from m malicious agents,
simulating unpredictable attack patterns.
❷Poisson attack process (P-mode) [30]. This mode models
bursty attack patterns (e.g., DDoS attacks) in which intense
activities occur in short durations. At each time step t ∈T,
the number of malicious messages Nt follows a Poisson
distribution:

P (Nt = k) = (λNa)kexp (−λNa)

k!
,
(20)

Benchmark Comparison. Table I evaluates GCP on V2XSim. First, while No Defense suffers severe degradation, all defenses show effectiveness. Notably, CP-Guard and Gated Late
Fusion are competitive against conventional attacks. However,
despite MADE slightly outperforming ROBOSAC, all baselines are vulnerable to our BAC attack. In contrast, GCP excels.
For conventional attacks, it achieves consistent improvements
(0.47%-0.79% AP@0.5). The advantage is pronounced under
BAC attack, where GCP surpasses CP-Guard, Gated Late
Fusion, MADE, and ROBOSAC by 9.42%, 6.26%, 12.64%,
and 8.53% respectively, maintaining near-upper-bound performance. Across R/P/S-Modes, GCP demonstrates consistent

where k denotes the number of malicious messages at time
step t, and λNa represents the mean rate.
❸Susceptible-Infectious process (S-mode) [31]. This mode
simulates progressive attack scenarios where malicious be
9

TABLE I
COMPARATIVE RESULTS UNDER DIFFERENT ATTACK METHODS AND MODES. SETTINGS: m = 2, λ = 0.25, ∆i = ∆o = 0.5, ITERATIONS=10.
BOLD/UNDERLINED: BEST/SECOND-BEST (EXCL. UPPER-BOUND). ↓: SIGNIFICANT DROP.

Method
Mode
PGD attack
C&W attack
BIM attack
BAC attack
AP@0.5
AP@0.7
AP@0.5
AP@0.7
AP@0.5
AP@0.7
AP@0.5
AP@0.7
Upper-bound
—
80.52
78.65
80.52
78.65
80.52
78.65
80.52
78.65

GCP (Ours)

Avg.
77.54
76.51
76.81
75.56
77.40
76.46
76.64
75.64
R
76.88
75.78
76.90
75.16
76.72
75.74
75.80
74.96
P
77.37
76.34
77.13
76.20
77.38
76.43
76.29
75.09
S
78.36
77.41
76.41
75.32
78.11
77.22
77.82
76.87

CP-Guard

Avg.
76.54
75.24
75.82
74.45
76.12
74.68
67.22↓
63.88↓
R
76.12
74.98
75.45
73.88
76.35
74.88
67.78
63.95
P
76.25
74.55
75.92
74.38
76.45
75.38
66.88
64.55
S
77.25
76.18
76.08
75.08
75.55
73.78
66.98
63.15

Gated Late Fusion

Avg.
71.18
68.82
70.52
67.98
71.08
68.75
70.38
68.02
R
70.65
68.12
70.58
67.58
70.42
68.08
69.62
67.42
P
71.02
68.64
70.78
68.48
71.05
68.72
70.05
67.52
S
71.95
69.62
70.18
67.72
71.68
69.42
71.48
69.12

MADE

Avg.
77.07
75.75
76.28
74.87
76.61
75.06
64.00↓
54.48↓
R
76.67
75.56
75.81
74.28
76.82
75.28
65.63
56.92
P
76.76
75.04
76.29
74.81
76.96
75.88
65.11
55.95
S
77.79
76.66
76.39
75.42
76.06
74.01
61.28
50.58

ROBOSAC

Avg.
73.31
71.54
73.90
72.16
73.48
71.70
68.11↓
64.77↓
R
74.64
72.93
74.52
72.78
74.67
73.19
68.68
64.82
P
73.98
72.84
73.63
72.45
73.63
72.07
67.25
65.48
S
71.31
68.84
73.54
71.26
72.14
69.85
68.40
64.02

No Defense

Avg.
36.65↓
35.92↓
17.70↓
15.14↓
38.27↓
37.28↓
54.83↓
45.11↓
R
36.50
35.64
18.03
15.52
37.07
36.18
55.19
43.99
P
39.01
38.35
18.92
16.23
36.59
35.47
55.03
45.07
S
34.45
33.78
16.14
13.66
41.16
40.19
54.26
46.28
Lower-bound
—
64.08
61.99
64.08
61.99
64.08
61.99
64.08
61.99

robustness where baselines degrade, validating our spatialtemporal defense against diverse attacks.

TABLE III
IMPACT OF THE NUMBER OF MALICIOUS AGENTS (m) ON DETECTION
PERFORMANCE (UNDER PGD ATTACK).

TABLE II
PERFORMANCE COMPARISON (AP@0.5 / AP@0.7) UNDER INDEPENDENT
VS. COLLUDING ATTACKS (m = 2).

# Mal. (m)
AP@0.5
AP@0.7
No Defense
GCP (Ours)
No Defense
GCP (Ours)
0 (Benign)
80.52
78.42
78.65
77.51
1
50.32
78.32
48.42
77.05
2 (Default)
36.65
77.54
35.92
76.51
3
22.11
73.08
21.56
71.88

Attack
Method
Independent
Colluding
AP@0.5
AP@0.7
AP@0.5
AP@0.7

PGD
No Defense
36.65
35.92
34.35
33.66
GCP (Ours)
77.54
76.51
74.51
73.53

C&W
No Defense
17.70
15.14
16.42
14.05
GCP (Ours)
76.81
75.56
73.89
72.68

TABLE IV
ABLATION STUDY. GCP VS. VARIANTS (GCP-S/T). SETTINGS:
m = 2, λ = 0.25, ∆i = ∆o = 0.5.

BIM
No Defense
38.27
37.28
35.88
34.91
GCP (Ours)
77.40
76.46
74.65
73.72

BAC
No Defense
54.83
45.11
51.92
42.85
GCP (Ours)
76.64
75.64
74.28
73.35

Method
PGD attack
BAC attack
AP@0.5
AP@0.7
AP@0.5
AP@0.7
Upper-bound
80.52
78.65
80.52
78.65
GCP (Ours)
77.54
76.51
76.64
75.64
GCP-S
77.21
76.10
69.23
62.38
GCP-T
70.16
67.33
66.01
58.79
No Defense
36.65
35.92
54.83
45.11

Robustness under Different Attack Configurations. We
first evaluate stability under varying intensities and attacker
counts. Table V shows that under intense attacks (m = 3,
λ = 0.5), GCP maintains stability with only 5.6% degradation,
significantly outperforming baselines. It also shows consistent
performance (<0.5% variation) across different perturbation
budgets. Furthermore, Table III confirms resilience to increasing attacker counts. In benign scenarios (m = 0), GCP matches
the Upper-bound, ensuring minimal impact. As m increases,
GCP maintains robust performance (around 77% for m ≤2)
while No Defense degrades catastrophically. This confirms
GCP’s reliability in highly compromised networks.
Robustness against More Attack Strategies. We further
evaluate GCP against two advanced strategies. ❶Coordinated
Attacks: As shown in Table II, even when agents optimize
a joint adversarial objective, GCP remains highly effective
(AP@0.5 > 73%). ❷LiDAR Spoofing Attacks: We implement
a physically realistic Ray-Casting attack [9] that injects synthesized points by calculating intersections between LiDAR rays
and ghost vehicle meshes. Although this generates plausible
raw data perturbations, the resulting ghost boxes are often

sparse and intermittent due to optimization constraints. GCP
effectively identifies these anomalies, achieving an 84.5%
defense success rate (at K = 5).
Overhead Analysis. We analyze the computational efficiency
on a single NVIDIA RTX A5000 GPU using the V2X-Sim test
set under the default attack setting (m = 2, λ = 0.25, ∆i =
∆o = 0.5). As detailed in Table VI, the average total defense
latency per frame is around 26.0 ms, which is dominated by
the temporal LSTM-AE inference. This latency corresponds
to a processing speed of around 38.6 FPS, as compared with
other methods in Table VII. Since this speed significantly
exceeds the typical 10-20 Hz sensor frequency (e.g., LiDAR),
GCP can verify collaborative messages within the generation
cycle. This ensures a 0-frame detection delay (where attacks
at frame t are identified immediately), preventing perception
bottlenecks. Furthermore, GCP is resource-efficient, incurring
zero bandwidth overhead and negligible GPU memory cost

10

Fig. 5.
Visualization of 3D detection results on V2X-Sim dataset. Attack settings: number of malicious agents = 2; attack ratio = 0.25; input/output
perturbation budget = 0.5. Scene ID: 8, Frame ID: 81, 65, 30, 47 (from top to bottom). Red boxes are predictions while the green ones are GT.

TABLE V
PERFORMANCE UNDER DIFFERENT ATTACK INTENSITIES. SETTINGS: MODERATE (m = 2, λ = 0.25) VS. INTENSE (m = 3, λ = 0.5).
BOLD/UNDERLINED: BEST/SECOND-BEST (EXCL. UPPER-BOUND). ↓: SIGNIFICANT DROP.

Method
BAC Attack
∆i = 0.5, ∆o = 0.5
∆i = 0.5, ∆o = 0.1
∆i = 1.0, ∆o = 0.5
∆i = 1.0, ∆o = 0.1
AP@0.5
AP@0.7
AP@0.5
AP@0.7
AP@0.5
AP@0.7
AP@0.5
AP@0.7
Upper-bound
—
80.52
78.65
80.52
78.65
80.52
78.65
80.52
78.65

GCP (Ours)
Moderate
75.80
74.96
75.23
74.02
75.58
74.28
75.51
74.69
Intense
70.22
68.57
70.03
68.36
70.07
68.59
69.77
68.20

CP-Guard
Moderate
72.15
68.25
69.88
65.55
67.55
61.25
69.25
64.55
Intense
63.55
57.88
64.55
58.25
62.15
54.25
64.15
56.95

Gated Late Fusion
Moderate
69.62
67.42
69.12
66.58
69.42
66.82
69.35
67.18
Intense
64.48
61.65
64.32
61.48
64.35
61.62
64.08
61.32

MADE
Moderate
65.63
56.92
65.80
57.26
65.39
55.56
66.93
56.79
Intense
51.94↓
40.87↓
54.22↓
41.38↓
53.21↓
40.61↓
51.04↓
36.25↓

ROBOSAC
Moderate
68.68
64.82
66.26
62.38
64.12
58.08
65.64
61.42
Intense
60.09
54.88
61.20
55.07
59.08
51.14
61.02
53.87

No Defense
Moderate
55.19↓
43.99↓
65.05
55.44
55.92↓
42.91↓
59.54
47.27↓
Intense
35.63↓
25.35↓
47.82↓
32.55↓
33.40↓
17.98↓
41.85↓
23.24↓

C. Ablation Studies

(+118 MB), making it ideal for resource-constrained deployment. For extreme constraints, further optimizations like model
distillation or FP16 quantization could be applied.

Generalizability to Different CP Model. We extend the evaluation to include DiscoNet [18] and When2com [20] besides
the default V2VNet. DiscoNet utilizes distilled collaboration
while When2com employing handshake communication for
collaboration. Table VIII (averaged over three attack modes)
shows that GCP consistently protects these diverse CP models,
demonstrating its broad applicability and robustness across
different collaborative fusion frameworks.

Ablation on Defense Components. We evaluate the contributions of our defense components. First, we compare GCP
with its spatial-only (GCP-S) and temporal-only (GCP-T)
variants. As shown in Table IV, both components contribute
to the overall defense. For PGD attacks, the spatial component is dominant, with GCP-S achieving 77.21% AP@0.5,
comparable to the full model. However, for the challenging
BAC attack, neither component alone suffices, with significant
drops in GCP-S (7.41%) and GCP-T (10.63%). This confirms
the necessity of our joint spatial-temporal design. Notably,
GCP-S alone outperforms ROBOSAC [11] and MADE [12],
validating our CSCLoss design.

11

Fig. 6.
Comparative results under different cached frame length and consecutive KF interpolation times. Attack settings: m = 2, λ = 0.25;
∆i = ∆o = 0.5. (a)-(d): AP@0.7 under different attacks and interpolation budgets; (e)-(h): AP@0.7 under different attacks and cached frame length.

TABLE VI
RESOURCE CONSUMPTION AND RUNTIME BREAKDOWN.

TABLE VIII
GENERALIZABILITY OF GCP ACROSS DIFFERENT CP MODELS.

CP Model
Method
AP@0.5
AP@0.7
No Def.
GCP
No Def.
GCP

DiscoNet

PGD
37.05
78.35
36.25
77.28
C&W
17.88
77.62
15.28
76.35
BIM
38.69
78.21
37.62
77.25
BAC
55.42
77.45
45.58
76.42

When2com

Resource Consumption
Metric
No Defense
GCP (Ours)
Overhead
GPU Memory (MB)
28,020
28,138
+118
GPU Utilization (%)
0.6
0.9
+0.3
CPU Utilization (%)
10.4
6.7
-3.7
Runtime Breakdown
Module
Time (ms)
Equiv. FPS
Spatial Defense (CSCLoss)
3.2
Temporal Defense (Flow Gen.)
1.8
Temporal Defense (LSTM-AE)
21.0
Total Defense Latency
26.0
38.6

PGD
23.65
49.65
22.05
46.85
C&W
11.45
49.18
9.32
46.28
BIM
24.68
49.52
22.88
46.82
BAC
35.32
49.08
27.65
46.35

TABLE VII
ATTACK AND DEFENSE SPEED (FPS). SETTINGS:
m = 2, λ = 0.25, ∆i = ∆o = 0.5.

Attack Method
Attack
Defense Speed (FPS)
Speed (FPS)
MADE
ROBOSAC
GCP
PGD attack
25.3
55.8
27.1
38.6
C&W attack
18.2
38.5
21.6
30.2
BIM attack
24.8
54.6
28.7
37.5
BAC attack
22.4
82.8
44.6
47.6

Ablation on Hyper-Parameters. We analyze the frame cache
length (K) and consecutive KF interpolation times (L). As
shown in Fig. 6, moderate values yield optimal performance.
The cache size peaks around K = 5 (AP@0.5 of 76.87%
under BAC S-mode), balancing context and noise. For interpolation times, performance is optimal at L = 3 (77.42%
AP@0.5 for BIM), as excessive interpolation may introduce
cumulative errors.

D. Qualitative Results

Visualization of Attack Pipeline. Fig. 7 illustrates the complete pipeline of the BAC attack on the V2X-Sim dataset.
The malicious agent analyzes detection results from the victim’s local perception (Fig. 7(a)) to identify blind spots. By

comparing with CP results (Fig. 7(b)), it pinpoints regions
where the victim relies heavily on collaborative messages.
This differential analysis yields an initial seed map (Fig. 7(c)),
which is refined into a BAC confidence mask (Fig. 7(d)) via
blind region segmentation to guide targeted perturbations.
Visualization of Defense Performance. We visualize the collaborative 3D detection performance in Fig. 5. While conventional attacks (PGD, C&W, BIM) generate obvious perturbations that are easily detected, our BAC attack optimizes outputspace perturbations in blind regions, demonstrating superior
stealthiness and causing significant degradation in baselines
like ROBOSAC and MADE. In contrast, GCP maintains robust
detection through spatial-temporal consistency verification,
achieving results comparable to the upper bound.
Visualization of BEV Flow Reconstruction Loss. Fig. 8
analyzes the BEV flow reconstruction loss distribution. Normal agents (Agent 3 and 4) show consistently low losses
(mean ≈0.01) and stable temporal consistency. In contrast,
agents under BAC attack (Agent 0 and 2) exhibit significantly
higher maximum losses (> 0.4) and more unmatched flows.
These high-loss outliers and temporal inconsistencies serve as

12

Fig. 7.
Visualization of the BAC attack pipeline on V2X-Sim dataset. (a) Detection results using only victim vehicle’s local perception; (b) Enhanced
detection results through CP; (c) Initial BAC seed map generated from differential detection results; (d) Refined BAC confidence mask obtained through blind
region segmentation. Red boxes are predictions while the green ones are GT.

Fig. 9.
Statistical dependency analysis and FDR control validation. It
validates the PRDS assumption via positive correlations and demonstrates the
effectiveness of the chosen operating point.

Fig. 8. BEV flow reconstruction loss distribution. Agent 0 and 2 are under
BAC attack (∆o = 0.5) while agent 3 and 4 are normal. Scene ID: 8, Frame
ID: 61.

Therefore, investigating the feasibility of such stealthy threats
and developing corresponding defense mechanisms remains an
important direction for future research.

VI. CONCLUSION

reliable indicators for detecting stealthy attacks.
Visualization of Statistical Validity. We validate the statistical assumptions on a small calibration set comprising
389 agent-frame samples from V2X-Sim Scene 8. The BH
procedure requires the test statistics to satisfy Positive Regression Dependence on a Subset (PRDS). As shown in Fig. 9,
we observe consistently non-negative cross-agent correlations
(r ∈[0.02, 0.51]) due to shared field-of-views, alongside positive temporal autocorrelation from physical continuity. This
positive dependence structure satisfies the PRDS condition,
ensuring the validity of FDR control. Fig. 9(d) further confirms
that our chosen operating point (αbh = 0.36) achieves 100%
attack recall while maintaining conservative FDR control,
effectively balancing safety and benign performance.

E. Limitation and Future Work

Currently, we focus on defending against untargeted adversarial attacks without temporal optimization. While sophisticated “slow-drift” attacks could potentially evade detection
by generating temporally consistent long-term trajectories,
such methods are currently unexplored in the context of realtime multi-agent systems due to the optimization difficulities.

In this paper, we have devised a novel blind area confusion
(BAC) attack to show that collaborative perception (CP) systems are vulnerable to malicious attacks even with existing
outlier-based CP defense mechanisms. The key innovation
of the BAC attack lies in the blind region segmentationbased local perturbation optimization. To counter such attacks,
we have proposed our GCP, a robust CP defense framework
utilizing spatial and temporal contextual information through
confidence-scaled spatial concordance loss and LSTM-AEbased temporal BEV flow reconstruction. These components
are integrated via a spatial-temporal Benjamini-Hochberg test
to generate reliable anomaly scores for malicious agent detection. Extensive experiments demonstrate GCP’s superior
robustness against diverse attacks while preserving benign
performance. These results validate the efficacy of spatialtemporal analysis in guarding CP systems. We believe our
work establishes a solid foundation for secure multi-agent
collaboration in future autonomous driving applications.

13

REFERENCES

[20] Y.-C. Liu, J. Tian, N. Glaser, and Z. Kira, “When2com: Multi-agent
perception via communication graph grouping,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), June 2020.
[21] Z. Fang, J. Wang, Y. Ma, Y. Tao, Y. Deng, X. Chen, and Y. Fang,
“R-ACP: Real-time adaptive collaborative perception leveraging robust
task-oriented communications,” IEEE Journal on Selected Areas in
Communications, June 2025.
[22] S. Wei, Y. Wei, Y. Hu, Y. Lu, Y. Zhong, S. Chen, and Y. Zhang,
“Asynchrony-Robust Collaborative Perception via Bird’s Eye View
Flow,”
in
Advances
in
Neural
Information
Processing
Systems
(NeurIPS), vol. 36, 2023, pp. 28 462–28 477.
[23] Y. Lu, Q. Li, B. Liu, M. Dianati, C. Feng, S. Chen, and Y. Wang,
“Robust Collaborative 3d Object Detection in Presence of Pose Errors,”
in IEEE International Conference on Robotics and Automation (ICRA),
2023, pp. 4812–4818.
[24] J. Yin, J. Shen, C. Guan, D. Zhou, and R. Yang, “LiDAR-Based Online
3D Video Object Detection With Graph-Based Message Passing and
Spatiotemporal Transformer Attention,” in IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2020, pp. 11 492–
11 501.
[25] X. Gao, Z. Chen, J. Pan, F. Wu, and G. Chen, “Energy Efficient Scheduling Algorithms for Sweep Coverage in Mobile Sensor Networks,” IEEE
Transactions on Mobile Computing, vol. 19, no. 6, pp. 1332–1345, 2020.
[26] J. Liu, Y. Zhang, X. Zhao, Z. He, W. Liu, and X. Lv, “Fast and
Robust LiDAR-Inertial Odometry by Tightly-Coupled Iterated Kalman
Smoother and Robocentric Voxels,” IEEE Transactions on Intelligent
Transportation Systems, vol. 25, no. 10, pp. 14 486–14 496, 2024.
[27] Y. Wei, J. Jang-Jaccard, F. Sabrina, W. Xu, S. Camtepe, and
A. Dunmore, “Reconstruction-based LSTM-Autoencoder for Anomalybased DDoS Attack Detection over Multivariate Time-Series Data,”
arXiv:2305.09475, 2023.
[28] Y. Benjamini and Y. Hochberg, “Controlling the False Discovery Rate:
A Practical and Powerful Approach to Multiple Testing,” Journal of the
Royal statistical society: series B (Methodological), vol. 57, no. 1, pp.
289–300, 1995.
[29] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun,
“CARLA: An open urban driving simulator,” in Proceedings of the 1st
Annual Conference on Robot Learning, 2017, pp. 1–16.
[30] Y. Xiang, K. Li, and W. Zhou, “Low-Rate DDoS Attacks Detection and
Traceback by Using New Information Metrics,” IEEE Transactions on
Information Forensics and Security, vol. 6, no. 2, pp. 426–437, 2011.
[31] H. Ahn, J. Choi, and Y. H. Kim, “A Mathematical Modeling of StuxnetStyle Autonomous Vehicle Malware,” IEEE Transactions on Intelligent
Transportation Systems, vol. 24, no. 1, pp. 673–683, 2023.
[32] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards
Deep Learning Models Resistant to Adversarial Attacks,” in International Conference on Learning Representations (ICLR), 2018.
[33] N. Carlini and D. Wagner, “Towards Evaluating the Robustness of Neural
Networks,” in IEEE Symposium on Security and Privacy (SP), May
2017, pp. 39–57.
[34] W. Luo, B. Yang, and R. Urtasun, “Fast and Furious: Real Time Endto-End 3D Detection, Tracking and Motion Forecasting with a Single
Convolutional Net,” in IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), Jun. 2018, pp. 3569–3577.
[35] A. Kurakin, I. Goodfellow, and S. Bengio, “Adversarial examples in the
physical world,” arXiv:1607.02533, 2017.

[1] Q. Chen, S. Tang, Q. Yang, and S. Fu, “Cooper: Cooperative Perception
for Connected Autonomous Vehicles Based on 3D Point Clouds ,”
in IEEE International Conference on Distributed Computing Systems
(ICDCS), Jul. 2019, pp. 514–524.
[2] Y. Li, D. Ma, Z. An, Z. Wang, Y. Zhong, S. Chen, and C. Feng, “V2XSim: Multi-Agent Collaborative Perception Dataset and Benchmark for
Autonomous Driving,” IEEE Robotics and Automation Letters, vol. 7,
no. 4, pp. 10 914–10 921, 2022.
[3] S. Hu, Z. Fang, H. An, G. Xu, Y. Zhou, X. Chen, and Y. Fang, “Adaptive
Communications in Collaborative Perception with Domain Alignment
for Autonomous Driving,” arXiv:2310.00013, 2024.
[4] Y. Hu, S. Fang, Z. Lei, Y. Zhong, and S. Chen, “Where2comm:
Communication-Efficient Collaborative Perception via Spatial Confidence Maps,” in Advances in Neural Information Processing Systems
(NeurIPS), 2022.
[5] S. Hu, Z. Fang, Y. Deng, X. Chen, Y. Fang, and S. Kwong, “Towards
Full-scene Domain Generalization in Multi-agent Collaborative Bird’s
Eye View Segmentation for Connected and Autonomous Driving,”
arXiv:2311.16754, 2024.
[6] Z. Fang, S. Hu, H. An, Y. Zhang, J. Wang, H. Cao, X. Chen, and
Y. Fang, “PACP: Priority-Aware Collaborative Perception for Connected
and Autonomous Vehicles,” IEEE Transactions on Mobile Computing,
vol. 23, no. 12, pp. 15 003–15 018, 2024.
[7] Y. Tao, S. Hu, Z. Fang, and Y. Fang, “Direct-CP: Directed Collaborative Perception for Connected and Autonomous Vehicles via Proactive
Attention,” arXiv:2409.08840, 2024.
[8] R. Xu, X. Xia, J. Li, H. Li, S. Zhang, Z. Tu, Z. Meng, H. Xiang, X. Dong,
R. Song, H. Yu, B. Zhou, and J. Ma, “V2v4real: A real-world large-scale
dataset for vehicle-to-vehicle cooperative perception,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), June 2023, pp. 13 712–13 722.
[9] Q. Zhang, S. Jin, R. Zhu, J. Sun, X. Zhang, Q. A. Chen, and Z. M. Mao,
“On Data Fabrication in Collaborative Vehicular Perception: Attacks and
Countermeasures,” in 33rd USENIX Security Symposium, Aug. 2024, pp.
6309–6326.
[10] J. Tu, T. Wang, J. Wang, S. Manivasagam, M. Ren, and R. Urtasun,
“Adversarial Attacks On Multi-Agent Communication,” in IEEE/CVF
International Conference on Computer Vision (ICCV), 2021, pp. 7748–
7757.
[11] Y. Li, Q. Fang, J. Bai, S. Chen, F. Juefei-Xu, and C. Feng, “Among
Us: Adversarially Robust Collaborative Perception by Consensus,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision (ICCV), October 2023, pp. 186–195.
[12] Y. Zhao, Z. Xiang, S. Yin, X. Pang, S. Chen, and Y. Wang, “Malicious
Agent Detection for Robust Multi-Agent Collaborative Perception,”
arXiv:2310.11901, 2024.
[13] S. Hu, Y. Tao, G. Xu, Y. Deng, X. Chen, Y. Fang, and S. Kwong, “Cpguard: malicious agent detection and defense in collaborative bird’s eye
view perception,” in Proceedings of the Thirty-Ninth AAAI Conference
on Artificial Intelligence and Thirty-Seventh Conference on Innovative
Applications of Artificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence, ser. AAAI’25. AAAI Press,
2025.
[14] Anonymous, “CP-Guard+: A New Paradigm for Malicious Agent
Detection and Defense in Collaborative Perception,” in Submitted to
The Thirteenth International Conference on Learning Representations
(ICLR), 2024, under review.
[15] Y. Han, H. Zhang, H. Li, Y. Jin, C. Lang, and Y. Li, “Collaborative
Perception in Autonomous Driving: Methods, Datasets and Challenges,”
IEEE Intelligent Transportation Systems Magazine, vol. 15, no. 6, pp.
131–151, Nov. 2023, arXiv:2301.06262 [cs].
[16] S. Hu, Z. Fang, Y. Deng, X. Chen, and Y. Fang, “Collaborative Perception for Connected and Autonomous Driving: Challenges, Possible
Solutions and Opportunities,” Jan. 2024, arXiv:2401.01544.
[17] W. Zeng, S. Wang, R. Liao, Y. Chen, B. Yang, and R. Urtasun, “DSDNet:
Deep Structured Self-driving Network,” in European Conference on
Computer Vision (ECCV), 2020, pp. 156–172.
[18] Y. Li, S. Ren, P. Wu, S. Chen, C. Feng, and W. Zhang, “Learning
Distilled Collaboration Graph for Multi-Agent Perception,” in Advances
in Neural Information Processing Systems (NeurIPS), vol. 34, 2021, pp.
29 541–29 552.
[19] T.-H. Wang, S. Manivasagam, M. Liang, B. Yang, W. Zeng, and R. Urtasun, “V2VNet: Vehicle-to-Vehicle Communication for Joint Perception
and Prediction,” in European Conference on Computer Vision (ECCV),
2020, pp. 605–621.

14

Yihang Tao received the B.S. degree from the
School of Information Science and Engineering,
Southeast University, Nanjing, China, in 2021 and
received his M.S. degree from the School of
Computer Science, Shanghai Jiao Tong University,
Shanghai, China, in 2024. Currently, he is pursuing
his PhD degree in the Department of Computer
Science at City University of Hong Kong. His current research interests include world model, spatial
intelligence, and autonomous driving. He serves as
the PC member for CVPR, ICML, ICLR, etc.

Senkang Hu received his B.S. degree in electronic
and information engineering from Beijing Institute
of Technology, Beijing, China, in 2022. He is pursuing his PhD in the Department of Computer Science
at City University of Hong Kong, Hong Kong. His
research interests include connected and autonomous
driving, vehicle-to-vehicle collaborative perception,
and LLMs.

Yuguang Fang (S’92, M’97, SM’99, F’08) received
his MS from Qufu Normal University, China in
1987, his PhD from Case Western Reserve University, Cleveland, Ohio, USA, in 1994, and another
PhD from Boston University, Boston, Massachusetts,
USA, in 1997. He joined the Department of Electrical and Computer Engineering at University of
Florida in 2000 as an assistant professor, then was
promoted to associate professor in 2003, full professor in 2005, and distinguished professor in 2019,
respectively. Since August 2022, he has been a
Global STEM Scholar and Chair Professor with the Department of Computer
Science at City University of Hong Kong.
Prof. Fang received many awards, including the US NSF CAREER Award
(2001), US ONR Young Investigator Award (2002), 2018 IEEE Vehicular
Technology Outstanding Service Award, IEEE Communications Society’s
AHSN Technical Achievement Award (2019), CISTC Technical Recognition
Award (2015) and WTC Recognition Award (2014), and 2010-2011 UF Doctoral Dissertation Advisor/Mentoring Award. He held multiple professorships,
including the Changjiang Scholar Chair Professorship (2008-2011), Tsinghua
University Guest Chair Professorship (2009-2012), University of Florida
Foundation Preeminence Term Professorship (2019-2022), and University
of Florida Research Foundation Professorship (2017-2020, 2006-2009). He
served as the Editor-in-Chief of IEEE Transactions on Vehicular Technology (2013-2017) and IEEE Wireless Communications (2009-2012) and
serves/served on several editorial boards of journals, including Proceedings
of the IEEE (2018-2023), ACM Computing Surveys (2017-present), ACM
Transactions on Cyber-Physical Systems (2020-present), IEEE Transactions
on Mobile Computing (2003-2008, 2011-2016, 2019-present), IEEE Transactions on Communications (2000-2011), and IEEE Transactions on Wireless
Communications (2002-2009). He served as the Technical Program CoChair of IEEE INFOCOM’2014. He is a Member-at-Large of the Board of
Governors of IEEE Communications Society (2022-2024) and the Director of
Magazines of IEEE Communications Society (2018-2019). He is a fellow of
ACM, IEEE, and AAAS.

Yue Hu is currently a postdoctoral researcher in
the Department of Robotics, University of Michigan,
Ann Arbor, USA. She received her PhD from the
Cooperative Medianet Innovation Center at Shanghai
Jiao Tong University in 2024. She received her MS
and BE degrees in information engineering from
Shanghai Jiao Tong University, Shanghai, China, in
2020 and 2017, respectively. Her research interests
include multi-agent collaboration, communication
efficiency, and 3D vision.

Haonan An received his BE degree in Telecommunication Engineering from Huazhong University
of Science and Technology, Wuhan, China, in 2023,
and his MS in Signal Processing from the School
of Electrical and Electronic Engineering, Nanyang
Technological University, Singapore, in 2024. He
is pursuing his Ph.D. at the City University of
Hong Kong. His research interests include digital
watermarking, AI security, and autonomous driving.

Hangcheng Cao is currently a postdoctoral fellow in
the Department of Computer Science, City University of Hong Kong. He obtained the Ph.D. degree in
the College of Computer Science and Electronic Engineer, Hunan University, China, in 2023. He studied
as a joint PhD student in the School of Computer
Science and Engineering, Nanyang Technological
University, Singapore, in 2022. He has published
papers in IEEE S&P, ACM Ubicomp/IMWUT, IEEE
INFOCOM, IEEE ICDCS, IEEE TMC, ACM ToSN,
IEEE Communications Magazine, Information Fusion, etc. His research interests lie in the area of IoT security.

APPENDIX B
IMPLEMENTATION OF ADVERSARIAL ATTACKS

APPENDIX A
KF-BASED BEV FLOW INTERPOLATION

In this paper, we evaluate our proposed GCP and other
baselines by implementing two adversarial attacks:

• Projected Gradient Descent (PGD) Attack [32]: PGD
attack introduces a random initialization step to the adversarial example generation process. The mathematical
expression for PGD is initiated by adding uniformly
distributed noise to the original input:

Given the state transition equations for intermittent BEV
flow, we can directly apply these to the Kalman filter (KF)
framework for both prediction and state update, and thereby
interpolate the missing values. Assume that system state and
observation noises are additive white Gaussian noises and that
the state transition and observation models are linear, we first
define the state vector and the state transition matrix of BEV
flow:

F0
k = Fk + Uniform(−∆, ∆),
(31)

oj = [x1
j, y1
j , · · · , x4
j, y4
j ]⊤∈R8,
(22)

where ∆is a predefined perturbation limit. Subsequent
iterations adjust the adversarial example by moving in the
direction of the gradient of the loss function:

Ft+1
k
= Π∆{Ft
k + α · sign(∇Ft
kL(Ft
k, y))},
(32)

where oj represents the BEV flow state vector, containing 4
corner points of a bounding box in the BEV detection map,
with (xk
j , yk
j ) being the coordinates of the k-th corner point.
Then, the state transition equation can be represented as:

oj,k+1 = Fkoj,k + wk,
(23)

where t denotes the iteration index, α is the step size and
Π∆represents the projection operation that confines the
perturbation within the allowable range. This procedure is
typically repeated for a predefined number of iterations,
with settings such as T = 15 and α = 0.1 often used.

where Fk is the state transition matrix constructed based on
the equations provided, assuming no external control inputs
besides the physical model-based linear relationships:


,
(24)

Fk =
I4
∆tI4
04
I4

• Carini & Wagner (C&W) Attack [33]: The C&W
attack focuses on identifying the minimal perturbation
δ that leads to a misclassification, formulated as the
following optimization problem:

min
δ
|δ|p + c · f(Fk + δ),
(33)

where I4 represents 4 × 4 identity matrix, 04 is 4 × 4 zero
matrix, and ∆t is the time difference between frame k and
frame k −1. Based on the state transition matrix above, there
are two stages for KF, namely, the prediction stage and the
update stage. In the prediction stage, both the state and the
error covariance are predicted, given as:

where the function | · |p measures the size of the perturbation using the Lp norm, while c is a scaling constant
that adjusts the weight of the misclassification function
f, which is designed to increase the likelihood of misclassification:

ˆoj,k+1|k = Fkˆoj,k|k,
(25)

f(F′
k) = max(max
i̸=t Z(F′
k)i −Z(F′
k)t, −κ),
(34)

Pk+1|k = FkPk|kFT
k + Qk,
(26)

where Z(F′
k) outputs the logits from the model for the
perturbed input, with t indicating the target class, and κ
serving as a confidence parameter to ensure robustness in
the adversarial example.

where Qk is the process noise covariance matrix, which needs
to be set based on practical scenarios. In the update stage,
assuming the observation vector zk directly reflects all state
variables, we have the observation model as follows:

zk = Hkoj,k + vk,
(27)

• Basic Iterative Method (BIM) Attack [35]: The BIM
attack incrementally adjusts an initial input by applying
small but cumulative perturbations, based on the sign
of the gradient of the loss function with respect to the
input, aiming to maximize the prediction error in a model
while ensuring the perturbations remain within specified
bounds:

where Hk is the observation matrix, which can be simplified
to the identity matrix I8, if all state variables are directly
observable. The KF update process includes updating Kalman
gain, state estimate, and error covariance as follows:

Ft+1
k
= Clip∆{Ft
k + α · sign(∇Ft
kL(Ft
k, y))},
(35)

Kk = Pk+1|kHT
k (HkPk+1|kHT
k + Rk)−1,
(28)

ˆoj,k+1|k+1 = ˆoj,k+1|k + Kk(zk −Hkˆoj,k+1|k),
(29)

Pk+1|k+1 = (I8 −KkHk)Pk+1|k,
(30)

where t represents the iteration index, α is the size of
the step, ∆defines the maximum allowable perturbation,
and Clip∆is a function that restricts the values within a
∆boundary around the original features Fk. The initial
setting is F0
k = Fk. This procedure is iterated either
a preset number of times or until a specific stopping
condition is reached.

where Kk is the Kalman gain, ˆoj,k+1|k+1 is the state estimate,
Pk+1|k+1 is the error covariance, and Rk is the observation
noise covariance matrix.

---

*Source: arXiv:2501.02450*
