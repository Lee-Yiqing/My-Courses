---
id: B12
title: "A Systematic Literature Review on Large Language Models for Code Generation"
domain: B
year: 2024
arxiv_id: "2405.00235"
confidence: verified
source: "arXiv:2405.00235"
node_type: paper
---

# A Systematic Literature Review on Large Language Models for Code Generation

**Domain**: [[domain_B|LLM Code Generation]] | **Year**: 2024 | **Confidence**: [x] verified


## Authors
[[author_Yuxiang Wei|Yuxiang Wei]], [[author_Chunqiu Steven Xia|Chunqiu Steven Xia]], [[author_Lingming Zhang|Lingming Zhang]]


## Keywords
- [[kw_systematic literature review|systematic literature review]]
- [[kw_code generation|code generation]]
- [[kw_LLM|LLM]]
- [[kw_SLR|SLR]]

## Abstract

(Abstract not available - see PDF content below)

## Paper Content

Blockchain Price vs. Quantity Controls

Abdoulaye Ndiaye[0009−0000−7466−6444]

New York University, New York NY 10012, USA
andiaye@stern.nyu.edu
https://www.abdoulayendiaye.com/

Abstract. This paper studies the optimal transaction fee mechanisms
for blockchains, focusing on the distinction between price-based (P) and
quantity-based (Q) controls. By analyzing factors such as demand uncertainty, validator costs, cryptocurrency price ﬂuctuations, price elasticity
of demand, and levels of decentralization, we establish criteria that determine the selection of transaction fee mechanisms. We present a model
framed around a Nash bargaining game, exploring how blockchain designers and validators negotiate fee structures to balance network welfare
with proﬁtability. Our ﬁndings suggest that the choice between P and
Q mechanisms depends critically on the blockchain’s speciﬁc technical
and economic features. The study concludes that no single mechanism
suits all contexts and highlights the potential for hybrid approaches that
adaptively combine features of both P and Q to meet varying demands
and market conditions.

Keywords: blockchain, transaction fees, EIP-1559, EIP-4844, price elasticity, decentralization, validator costs, demand uncertainty, cryptocurrency volatility

1
Introduction

arXiv:2405.00235v1  [econ.GN]  30 Apr 2024

Transaction fee mechanisms play a crucial role in maintaining blockchains’ network stability, eﬃciency, and user satisfaction. These policies can be broadly
categorized into three main types, each with distinct characteristics and implications for the network’s economic and operational dynamics.
Quantity Controls (Q): This approach involves setting a maximum limit on
resource usage, such as the block size limit. Bitcoin’s fee policies are a prime
example of this strategy, where the block size limit is the primary mechanism
controlling the volume of transactions processed in each block. By capping the
block size, the network eﬀectively manages transaction throughput through a
ﬁrst-price auction.
Price Controls (P): Under this mechanism, a minimum price per unit of
resource usage is set, often dynamically adjusted to reﬂect current network conditions. Ethereum’s EIP-1559 and EIP-4844 illustrate cases where a base fee is
determined algorithmically, rising or falling based on the network’s congestion

2
A. Ndiaye

levels. This approach allows the quantity of transactions to adjust in response
to demand changes.

Fixed Unit Price (∅): Some networks implement a ﬁxed fee structure without
adjusting for market conditions. This straightforward approach, often adopted
at the beginning of the life-cycle of blockchains, can simplify transactions for
users but may lack the ﬂexibility needed to address ﬂuctuating network demand
eﬀectively.

The selection between these transaction fee mechanisms—whether a blockchain
opts for quantity control (Q), price control (P), or a ﬁxed price approach (∅)—is
inﬂuenced by several factors. Key among these is the interplay between the objectives of welfare-maximizing blockchain designers and proﬁt-maximizing validators. Blockchain designers typically aim to enhance overall network eﬃciency
and user satisfaction, while validators are incentivized to maximize their earnings
from transaction fees and block rewards.

The central question in this paper is: What speciﬁc conditions or characteristics of a blockchain aﬀect the decision between implementing a price control
or a quantity control mechanism? By analyzing various blockchain architectures
and economic factors, such as uncertainty in user demand, uncertainty in costs
to a validator to process transactions, cryptocurrency price ﬂuctuations, price
elasticity of demand, and levels of decentralization, we establish criteria that
determine the selection of transaction fee mechanisms.

Table 1 summarizes those criteria. First, in environments with high demand
uncertainty, exempliﬁed by blockchains with various use cases, following a price
control is preferred to adjust the block size to match ﬂuctuations in demand. Second, a quantity control is favored for blockchains with a consensus mechanism,
such as Proof of Work (PoW), where there is a signiﬁcant positive correlation
between demand uncertainty and marginal costs (hash rate). The reason is that
if the block size limit was allowed to adjust, validators may face higher costs
when demand—and hence their workload—increases. Third, when token prices
ﬂuctuate widely, implementing quantity controls helps avoid base fees that are
too high and leads to more stable transaction fees denominated in the native token. Fourth, blockchains characterized by a high elasticity of demand for block
space, such as those with faster blocks or quicker conﬁrmations, beneﬁt from
price controls, which allow more ﬂexible and responsive fee adjustments. Fifth,
quantity controls are adequate in highly decentralized networks with low validator bargaining power, as they become easier to enforce.

The optimal choice between an EIP-1559 type policy (P) or a traditional
block size limit (Q) is determined by the relative balance of these ﬁve economic
factors that emerge from the blockchain technical speciﬁcities. Both Bitcoin
and Ethereum blockchains face signiﬁcant uncertainties in user demand. The
marginal cost for Ethereum validators has been essentially constant since the
Proof of Stake (PoS) upgrade. In addition, Ethereum features faster blocks and
is arguably less decentralized than Bitcoin. Therefore, our results help explain
Ethereum’s adoption and planned adoption of price control mechanisms such as

Blockchain Price vs. Quantity Controls
3

EIP-1159 and EIP-4844. At the same time, they help explain why Bitcoin still
uses a sole block size limit as quantity control.

FACTORS
EXAMPLE
OUTCOME

High demand uncertainty
Diﬀerent use cases
P

+ Corr. btw demand uncertainty and MC
PoW
Q

Token price ﬂuctuations
Fees in native token
Q

High elasticity of inclusion in next block Faster blocks or conﬁrmation
P

Low validator bargaining power
High decentralization
Q

Table 1: Summary of factors leading to a P or Q equilibrium between ”welfaremaximizing” blockchain designers and ”proﬁt-maximizing” validators.

1.1
Literature Review

This research builds upon the foundational studies on price versus quantity controls initiated by [8] The choice of selecting a supply function under uncertain
conditions has been extensively discussed in the work of [4] My methodology
closely resembles the analysis by [6], who investigates the implications of these
choices from a macroeconomic standpoint However, my approach diverges by
focusing on the unique challenges blockchain designers face, who must balance
multiple technical and strategic objectives, such as setting block size limit The
results of this study contribute to the development of transaction fee mechanisms.
The microeconomic mechanism design perspective on TFMs has seen substantial growth, particularly with the contributions of [1], who analyze ”credible
mechanisms” that resist manipulation by designers. These are particularly relevant in the blockchain sphere, where [7] demonstrates that Ethereum’s EIP-1559
and related models meet these criteria, oﬀering a myopically credible solution
for validators and users. This is further supported by the ﬁndings of
[3], ensuring that the TFMs underlying the price controls studied here are incentivecompatible.
From a broader macroeconomic angle, this paper applies results in
[5] to
enhance our understanding of the tradeoﬀs in choosing the families of blockchain
TFMs.

Outline: The paper is organized as follows: Section 2 provides intuition on the
economics of price and quantity controls with perfect enforcement. Section 3
presents how each factor aﬀects the choice of controls in the blockchain context
where protocol designers cannot fully enforce policies. Section 4 takes stock of

4
A. Ndiaye

the results, discusses the general choice of a supply function for block space, and
concludes the paper.

2
Prices vs. Quantity Controls with Perfect Enforcement

In this section, we analyze the welfare implications of price and quantity controls
amid demand uncertainty and social cost uncertainty in a general setting to
illustrate the idea of [8].
Consider a general private beneﬁt of block space q, denoted by B(q). These
private beneﬁts may reﬂect the utility users derive from generating a block of
size q that matches their total demand. Social costs could include costs beyond
those incurred by validators, such as centralization costs. Uncertainty in private
beneﬁts and social costs are introduced through the B(q, Ψ) and C(q, η). From
the perspective of the blockchain designer, the value of setting the quantity ¯q in
advance is given by:
max
¯q∈R+ EΨ,η[B(¯q, Ψ) −C(¯q, η)].
(1)

Alternatively, if the blockchain designer sets the price p, in advance, quantities
adjust ex-post to match demand:

B1(qadj(p, Ψ), Ψ) = p.
(2)

The value of this price control takes quantity adjustments into account:

max
p∈R+ EΨ,η[B(qadj(p, Ψ), Ψ) −C(qadj(p, Ψ), η)],
(3)

For intuition on the choice of instruments between a minimum price (base
fee) and a maximum quantity (block size limit), consider a setting where there is
no uncertainty in the cost function but with a 50% chance demand is high and a
50% chance demand is low Figure 1 illustrates this example with a quantity limit
(top panel) and price control (bottom panel) under such demand uncertainty.
Consider, without loss of generality, that the block size limit qmax is equal to
block space demand when demand is low but is binding when demand is high, as
depicted in the top panel of the ﬁgure. In this context, the concept of deadweight
loss comes into play. Deadweight loss refers to the loss in total social surplus due
to an ineﬃcient market outcome—it occurs when supply and demand are not in
equilibrium. Here, the deadweight loss is represented by the shaded area between
the high demand curve, the marginal cost curve, and the block size limit. This
area is twice the deadweight loss that results from the block size limit.
Now, consider a scenario where the blockchain designer introduces a minimum price that exceeds the low-demand market price, as shown in the bottom
panel of the ﬁgure. In this scenario, the dark-shaded area represents twice the
deadweight loss that results from this price control. It can be observed that the
deadweight loss from the price control is lower than the deadweight loss from
the block size limit. This situation arises whenever the uncertainty in demand
exceeds the uncertainty in marginal costs.

Blockchain Price vs. Quantity Controls
5

private demand

Price

marginal
social cost

high demand
low demand
size limit
DWL

Block space

Price

base price
DWL

Block space

Fig. 1: Welfare improvement from price controls under demand uncertainty. Top
panel: deadweight loss of a quantity limit. Bottom panel: deadweight loss of price
control.

6
A. Ndiaye

Formally, the demand curve can be approximated around a quantity limit ¯q
as:
B1(q, Ψ) ≈B′ + β(Ψ) + B′′ · (q −¯q),
(4)

and the marginal cost is approximated by:

C1(q, η) ≈C′ + η(η) + C′′ · (q −¯q),
(5)

where it is assumed that there is uncertainty around a ﬁxed demand curve
B1(q) ≈B′ + B′′ · (q −¯q) and marginal cost curve C1(q) ≈C′ + C′′ · (q −¯q) such
that E[β] = E[η] = 0.
The comparative advantage of a price control over a quantity control, denoted
by ∆, can be expressed as:

∆≡E[(B(˜q(Ψ), Ψ) −C(˜q(Ψ), η)) −(B(¯q, Ψ) −C(¯q, η))].
(6)

This comparative advantage of price control is:

∆∝B′′

C′′2 + 1

C′′ ,
(7)

and if |B′′| > C′′, a price control improves welfare over a quantity control.
This is the main result of [8]. When demand is more uncertain than marginal
cost, price controls can lead to quantity adjustments that better match demand,
while marginal costs do not vary much. This result determines when price controls lead to welfare improvements over quantity controls.

2.1
EIP-1559: The Ethereum P Transaction Fee Mechanism

Inspired by Weitzman’s work about environmental regulation, [2] introduced a
revised pricing mechanism, EIP-1559, for the Ethereum blockchain. The system
has a target block size, currently set at qtarget = 15M gas (the unit of block
size), and a maximum block size of qmax = 2qtarget The minimum gas price, pt,
is adjusted based on the formula

pt = pt−1 · (1 + dqt−1 −qtarget

qtarget
)
(8)

where the adjustment parameter d is set by default for the minimum price to
double in 8 blocks when blocks are full, i.e., d = 1

8.
In the Ethereum blockchain, each transaction indexed by j has an associated
gas limit qj, computed based on a ﬁxed fee schedule px Transaction senders
pay an amount in ETH, the native currency of the blockchain, equal to qj ·
min{pt + δj, c}, where δj is the tip and c is the fee cap, with c ≥pt The base fee
revenue, PN
j=1 qjpt, s burned, mainly for reasons related to oﬀ-chain incentives
of validators [7], while the eﬀective tips are transferred to the validator.

Blockchain Price vs. Quantity Controls
7

2.2
Modeling Key Diﬀerences between Blockchains Fees and
Environment Regulation

Blockchains, particularly in permissionless systems, present unique challenges
that diﬀerentiate them from traditional economic regulation under uncertainty,
as illustrated above. These challenges stem primarily from the decentralized
nature of blockchains and the diverse stakeholders involved, each with diﬀerent
objectives and inﬂuences on market equilibrium.

Diverse Stakeholders: The key actors in a blockchain ecosystem include developers, validators (proposers or builders), and users. Each group holds varying
degrees of power and inﬂuence over the network’s operations and policies.

Absence of Central Authority: Unlike a government, a blockchain designer cannot
unilaterally enforce policies. Validators, who play a critical role in processing
transactions and creating new blocks, must be incentivized to follow proposed
changes, which may not always align with their interests.

Uncertainty: Blockchains face multiple sources of uncertainty that aﬀect their
operation and the feasibility of diﬀerent transaction fee mechanisms. These include ﬂuctuations in user demand, the variable costs faced by validators, and
the volatile prices of cryptocurrencies.
To address these challenges, we propose in [5] a model that conceptualizes
the decision-making process regarding transaction fee mechanisms as a Nash bargaining game between blockchain designers and validators. The model is structured around the following components:

Decisions: Blockchain designers must commit ex-ante to either a ﬁxed base fee
(P-setting) or a ﬁxed blockspace (Q-setting) before the full extent of uncertainties is realized.

Bargaining Model of Decentralization: The bargaining game is formalized by the
following optimization problem:

max
P,Q E[Social Beneﬁt(Ψ, η)]1−βE[Validator Proﬁts(Ψ, η)]β
(9)

where β ∈[0, 1] represents the bargaining power of validators.

3
How Diﬀerent Factors Aﬀects the P vs. Q choice

In this section, we discuss the contribution of each factor in the choice of price
or quantity controls.

8
A. Ndiaye

3.1
Uncertainty in User Demand

Deﬁnition and Examples: In our model, user demand for blockchain transactions can be deﬁned by the equation:

q = Ψ

pε
(10)

Where q represents the resource quantity used by transactions, Ψ captures
factors inﬂuencing demand such as transaction utility or network activity, and
p is the price per transaction.
Demand uncertainty is quantitatively expressed through the variance of Ψ,
denoted as V ar[Ψ]. High variability in Ψ indicates high uncertainty in user demand. This variability can be attributed to several causes:

Diﬀerent Use Cases: Blockchains can serve various applications with distinct
demand patterns.

Adoption Phase: As the technology matures and gains wider acceptance, demand
for block space can increase and be less volatile.

Cycles: Economic and speculative cycles can cause signiﬁcant ﬂuctuations in
activity levels on the blockchain.

Economic Implications: To address the challenges posed by high demand
uncertainty, one intuitive solution is to allow the block size to adjust with a
base fee (P) and better match the ﬂuctuating demand. When demand spikes,
increasing the block size can help accommodate more transactions, alleviating
congestion.

3.2
Uncertainty in Validator Costs

Validator costs play an important role in the operation of blockchains. These
costs can vary signiﬁcantly depending on the consensus mechanism.

Deﬁnition and Examples The marginal cost of transaction validation, denoted by C′(q), is represented by η. In practice, the covariance between Ψ (user
demand uncertainty) and η (validator marginal costs) is often positive, implying
that the costs to validators also tend to rise as demand increases. This relationship can be mathematically expressed as:

Cov[Ψ, η] > 0
(11)

For example, in PoW blockchains like Bitcoin, the hash rate—a proxy for
computational eﬀort and energy consumption—typically increases with higher
transaction demand, reﬂecting a positive and high covariance. Conversely, in
PoS systems, the marginal cost of block production is relatively constant and
less dependent on ﬂuctuating transaction volumes since the cost to validators is
mainly the ﬁxed cost of staking.

Blockchain Price vs. Quantity Controls
9

Economic Implications: A large positive covariance between validator marginal
costs and user demand favors quantity controls Q In particular, under a price
control mechanism (P), validators may face higher costs exactly when demand—and
hence their workload—increases. This scenario could lead to ineﬃciencies where
blocks become more expensive to produce precisely when they are most needed.

3.3
Cryptocurrency Price Fluctuations

Cryptocurrencies are notoriously volatile, which presents unique challenges for
blockchain transaction fee mechanisms.

Deﬁnition and Examples: In economic terms, people generally value their
wealth in stable currencies like the dollar or in terms of real goods rather than in
the native tokens of blockchains. One measure of price volatility for Ethereum,
for instance, is the variance of its token V ar[$ET H]. This variability means that
transaction fees’ real (USD) cost can ﬂuctuate widely, even if the fee in native
tokens remains constant.

Economic Implications: Large ﬂuctuations in cryptocurrency prices tilt the
balance against P mechanisms and towards Q mechanisms During periods when
the value of a cryptocurrency like Ethereum is high, the corresponding USD
value of base fees in a P mechanism can become prohibitively expensive. Another interpretation of this result is that if blockchain designers care about implementing a P mechanism, they should consider allowing fees to be paid in USD
and stablecoins.

3.4
Price Elasticity of Demand

Deﬁnition and Examples: The price elasticity of demand for inclusion in
the next block refers to the responsiveness of the number of transactions (i.e.,
block space used) to changes in the base fee. Mathematically, the exponent ε
of our demand function denotes the price elasticity of demand. A high value of
ε suggests that users are highly sensitive to changes in transaction costs. For
instance, fast blockchains with short block times or conﬁrmation times, such as
Ethereum, exhibit higher price elasticities. The estimate of εethereum ≈12.6 in
[5] suggests that even minor adjustments in the base fee can lead to signiﬁcant
changes in the demand for block space.

Economic Implications: A high price elasticity of demand for inclusion in
the next block ampliﬁes gains from P controls. A higher elasticity facilitates
faster adjustments in block size in response to changes in demand, amplifying
the beneﬁts of dynamically matching block space to user needs.

10
A. Ndiaye

3.5
Decentralization

Deﬁnition and Examples: Decentralization is an aspect of blockchains, often measured by the distribution of power among participants in the network.
Validator bargaining power, denoted by β, inversely correlates with the level of
decentralization within the network. A lower value of β implies higher decentralization, indicating reduced power concentration among validators.

Economic Implications: In a more decentralized blockchain, the reduced bargaining power of validators makes it easier for blockchain designers to implement
and enforce their preferred policies, such as adjustments to block size. This fact
is particularly relevant when designing mechanisms to adjust block size in response to ﬂuctuating demand dynamically, ensuring the network can eﬃciently
respond to user needs without undue inﬂuence from a concentrated group of
validators.

4
Conclusion

This paper has examined various factors inﬂuencing the decision between pricebased (P) and quantity-based (Q) transaction fee mechanisms in blockchain
systems. Below is a summary table that encapsulates the main ﬁndings:
The choice between implementing a policy akin to Ethereum’s EIP-1559 (P)
or opting for a traditional block size limit (Q) hinges on the blockchain’s technical speciﬁcities. We found that high demand uncertainty and faster blockchains
should favor price controls, while highly decentralized and PoW blockchains
should favor simple block size limits. Furthermore, EIP-1559-type mechanisms,
when needed, should compute base fees in USD due to cryptocurrency price
volatility. The relative balance between ﬁve factors -demand uncertainty, validator cost uncertainty, token price volatility, demand elasticity, and decentralizationdetermines the optimal transaction fee mechanism, which aligns the objectives
of welfare-maximizing blockchain designers and proﬁt-maximizing validators.
The optimal choice between a price-based and a quantity-based transaction
fee mechanism is not one-size-ﬁts-all but depends on a blockchain’s speciﬁc technical characteristics and the economic environment in which it operates. Future
research could further quantify these choices and explore optimal supply schedules that incorporate the advantages of both P and Q mechanisms, with the
potential to oﬀer more ﬂexible and robust fee structures as blockchain technology evolves and matures.

References

1. Akbarpour, M., Li, S.: Credible auctions: A trilemma. Econometrica 88(2), 425–467
(2020)
2. Buterin,
V.:
Blockchain
resource
pricing.
URL:
https://ethresear.
ch/uploads/default/original X 2 (2018)

Blockchain Price vs. Quantity Controls
11

3. Chung, H., Shi, E.: Foundations of transaction fee mechanism design. In: Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA).
pp. 3856–3899. SIAM (2023)
4. Klemperer, P.D., Meyer, M.A.: Supply function equilibria in oligopoly under uncertainty. Econometrica: Journal of the Econometric Society pp. 1243–1277 (1989)
5. Ndiaye, A.: Why bitcoin and ethereum diﬀer in transaction fees: A theory of
blockchain fee policies. Available at SSRN (2023)
6. Reis, R.: Inattentive producers. The Review of Economic Studies 73(3), 793–821
(2006)
7. Roughgarden, T.: Transaction fee mechanism design. ACM SIGecom Exchanges
19(1), 52–55 (2021)
8. Weitzman, M.L.: Prices vs. quantities. The Review of Economic Studies 41(4), 477–
491 (1974)

---

*Source: arXiv:2405.00235*
