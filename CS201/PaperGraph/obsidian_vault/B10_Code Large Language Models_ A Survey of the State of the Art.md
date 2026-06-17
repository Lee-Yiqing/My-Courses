---
id: B10
title: "Code Large Language Models: A Survey of the State of the Art"
domain: B
year: 2024
arxiv_id: "2401.08672"
confidence: verified
source: "arXiv:2401.08672 / ACM Computing Surveys"
node_type: paper
---

# Code Large Language Models: A Survey of the State of the Art

**Domain**: [[domain_B|LLM Code Generation]] | **Year**: 2024 | **Confidence**: [x] verified


## Authors
[[author_Zhi Jin|Zhi Jin]], [[author_Xinyu Dai|Xinyu Dai]], [[author_Jialu Liu|Jialu Liu]], [[author_Xing Hu|Xing Hu]], [[author_Bonian Dai|Bonian Dai]], et al.


## Keywords
- [[kw_survey|survey]]
- [[kw_code LLM|code LLM]]
- [[kw_code generation|code generation]]
- [[kw_code understanding|code understanding]]
- [[kw_state of the art|state of the art]]

## Abstract

(Abstract not available - see PDF content below)

## Paper Content

Concept Alignment

Sunayana Rane ∗
Department of Computer Science
Princeton University
Princeton, NJ 08540
srane@princeton.edu

Polyphony J. Bruna ∗
Department of Cognitive and Information Sciences
University of California Merced
Merced, CA 95343
pbruna@ucmerced.edu

Christopher Kello
Department of Cognitive and Information Sciences
University of California Merced
Merced, CA 95343
ckello@ucmerced.edu

Ilia Sucholutsky
Department of Computer Science
Princeton University
Princeton NJ, 08540
is2961@princeton.edu

Thomas L. Grifﬁths
Department of Psychology
Department of Computer Science
Princeton University
Princeton NJ, 08540
tomg@princeton.edu

Abstract

arXiv:2401.08672v1  [cs.LG]  9 Jan 2024

Discussion of AI alignment (alignment between humans and AI systems) has focused on value alignment, broadly referring to creating AI systems that share
human values. We argue that before we can even attempt to align values, it is
imperative that AI systems and humans align the concepts they use to understand
the world. We integrate ideas from philosophy, cognitive science, and deep learning to explain the need for concept alignment, not just value alignment, between
humans and machines. We summarize existing accounts of how humans and machines currently learn concepts, and we outline opportunities and challenges in
the path towards shared concepts. Finally, we explain how we can leverage the
tools already being developed in cognitive science and AI research to accelerate
progress towards concept alignment.

1
Introduction

Alignment of AI with humans is a critically important vision, and a complex technical and societal
challenge [12]. Most discussion of AI alignment focuses on value alignment, which has been deﬁned
broadly and often in the public discourse, for example as “AI that does what it morally ought to
do, as deﬁned by the individual or society,” [20] or “AI systems [that] act in accordance with our
ethical principles and objectives” [60]. What this means, however, is poorly delineated even for
those creating these deﬁnitions. Value alignment is discussed more frequently in news articles,
blogs, and opinion pieces [28, 60] than it is in the academic literature (although see [9, 23, 22,
26]). This important goal is difﬁcult to break down into a tractable path to implementation. Any
attempt at a general, all-encompassing solution is immediately riddled with thorny and seemingly
insurmountable questions: What exactly is a value for an AI? To whose values should AI be aligned?

∗Equal contribution.

Preprint. Under review.

How do these values change in a given context? Does the AI even understand what it means for a
human to have values?

These are difﬁcult questions that require a deeper look at AI alignment. It is important to remember
that even humans disagree strongly, and often violently, on values. That is one reason why we
have democratic governments that reﬂect a changing set of societal values, and a legal system to
periodically redeﬁne the practical boundaries of how those values translate to the real world, with all
their edge cases and exceptions, case by case. We also have values as deﬁned in religious systems,
economic systems, and even educational systems, and we all have a societal role in shaping our
world such that these values are fairly balanced between opposing factions whose values clash. This
process is complicated and the results are always imperfect, but we do at least have a process, and
it is important that the process be a democratic one that is accessible to everyone, and enriched by
a broad and diverse range of perspectives. However, creating value alignment among humans is
arguably considerably easier than between humans and machines for one simple reason: humans
already have a lot in common in how they conceive of the world around them.

In this paper we argue that before we can even attempt human-AI value alignment, it is important
to pursue human-AI alignment at the conceptual level, such that AI systems understand the world in
terms of the same concepts humans use to understand the world. We call this concept alignment. The
importance of concept alignment can be recognized by understanding where it fails among humans.
The process of scientiﬁc discovery is awash with the confrontation and resolution of incommensurate
ways of viewing the world: conceptual frameworks that oppose one another to the extent that it
becomes impossible to draw meaningful comparisons between them or to assess the one by the
standards of the other. They are more than simply different, they cut across the same conceptual
space in different ways [61], making it impossible to agree on concepts let alone values.

For example, an Aristotelian physicist and a Newtonian physicist, although sharing the term “motion,” could not engage in a productive dialogue on the topic because Aristotelian physics is
grounded in explaining changes of quality whereas Newtonian physics is grounded in explaining
projectile mechanics. These two systems of thought are incommensurable because it is not possible
to evaluate the claims of one system within the paradigm of the other, which make radically different
ontological assumptions about the world [33]. Our scientiﬁc language is, in fact, littered with the
remnants of past worldviews – lest we forget that “atom,” from the ancient Greek “atomos,” means
indivisible (due to the long-held belief that atoms were the fundamental unit of matter) and that
“planet,” from the ancient Greek “planetes,” means wanderer (due to their apparent motion against
the backdrop of the supposedly ﬁxed stars). Two physicists with such different worldviews would
face signiﬁcant challenges in agreeing on a set of values to guide their science.2

We also see signiﬁcant differences in concepts between children and adults, such as how children
understand liquids before acquiring the correct conservation principle [46]. When faced with two
glass containers, one short and stout and the other tall and thin, ﬁlled with the same quantity of
liquid, young children will believe that the taller glass contains more liquid. This is despite the fact
that the children have witnessed the same quantity of liquid being poured into the two containers.
Two people, a young child and an adult, will believe two contradictory facts (that the taller glass
contains more liquid / that the amount of liquid contained by the glasses is the same) about the same
set of circumstances due to a fundamental conceptual difference – the adult has access to the concept
of volume while the child does not. As a consequence, it might be hard for an adult and a child to
agree on the fair allocation of juice, let alone more complex problems of value alignment.

In the context of contemporary AI advancements, it is unclear to what extent our current AI technologies’ conceptual understandings of the world we inhabit are similar to our own. We have many
reasons to believe that there are substantial, potentially catastrophic differences. For example, AI
classifying black people as non-human primates [38], a self-driving Uber that killed a jaywalking
pedestrian because it didn’t understand the concept of crossing the street without using a crosswalk
[21], and an AI that learns the word label “wolf” to mean the snow in the background of images
of wolves [54]. Despite using the same language labels like “wolf” and “pedestrian,” the AI is fundamentally misaligned with our understanding of the concepts those words represent. How can we
begin to make progress towards value alignment when the concepts we use in our discussions about
it are often so grossly misaligned with the concepts the AI system is using to represent the world?

2For example, they might disagree on their scientiﬁc goals (i.e., which scientiﬁc questions are worth pursing).

2

To inform the debate on how to achieve value alignment between humans and AI, we draw from
philosophy to motivate why concept alignment is necessary for safer and more effective AI, and we
explain how the ﬁelds of cognitive science and AI research are already developing the tools needed
to accelerate progress towards concept alignment.

2
Goals for concept alignment

Achieving alignment is difﬁcult, even between humans. So what is good enough? What should we
aim for in concept alignment research?

One very concrete goal is:

Concept alignment will lead to natural language communication with AI systems
that is functionally equivalent to natural language communication among humans.

It’s commonly understood, especially in the age of deep learning, that AI outputs are, to humans,
often seemingly “weird,” and outside the distribution of what one would expect of human behavior
for the same task. Often an AI will ﬁnd a solution to a simple problem [68] in a way that is outside
the range of known solutions that humans would produce. AlphaGo [58] famously did so very
successfully, defeating one of the world’s best players in a game of Go by using a move the player
said no human would have made [40]. This type of behavior may be highly desirable when we are
seeking innovation and discovery. However, for cases in which AI is to be integrated into society,
such as self-driving cars that make the type of mistakes human drivers wouldn’t normally make, AI
that behaves in unpredictable and uninterpretable ways is often catastrophic and disturbing. One
goal in concept alignment research is to produce AI that will be easier to interact with because we
will understand it and it will understand us. The natural language that we use to communicate with
it will have a sufﬁciently shared meaning for it to be functionally equivalent to communicating and
collaborating with a human, and this means fewer non-human-like, “weird” mistakes.

3
The challenge for shared concepts

Is an AI’s use of natural language (like English) enough to convince us that the meaning behind the
words it uses is the same? When ChatGPT [1] produces words, do you trust that its understanding
of those words is the same as your own? Is the meaning behind each word in ChatGPT’s English the
same as the meaning behind each word you use when chatting with a friend or colleague? Can you
make the same assumptions about the meaning of the AI’s words as you do about another human?

The notion that language underspeciﬁes the world was formalized in 20th century epistemic critiques in analytic philosophy, spearheaded by W. V. O. Quine. Quine’s “naturalized epistemology”
offered a rebuttal to rationalist, Cartesian foundationalist epistemology, which grounded human
knowledge in a priori truths and epistemic certainty.3 Instead, Quine argues that knowledge is
ultimately grounded in the relationship between and development of scientiﬁc theories and the empirical evidence that supports them.4

An upshot of grounding knowledge in empirical evidence for Quine is his thesis of semantic indeterminacy, the thesis that translation, meaning, and reference are indeterminate. A primary argument
for semantic indeterminacy comes from Quine’s argument for the inscrutability of reference (also
known as “ontological relativity”) [50]. This argument has since been picked up by ﬁelds such as
cognitive science, where it is often better known as the “gavagai argument.”

The gavagai argument asks us to imagine a linguist tasked with translating an unknown foreign
language is faced with translating the term “gavagai” after witnessing a native speaker utter the term
whilst pointing into the distance wherein there sits a rabbit among the grass. Quine argues that there
is an inﬁnite number of possible referents of the term “gavagai” (e.g., “rabbit,” “rabbit (temporal)
stage,” “undetached rabbit part”), all of which perfectly describe any evidence that the linguist may
gather from the behavior of the native speaker. The argument illustrates that language fundamentally
underdetermines meaning.

3For Quine’s argument against the possibility of an a priori foundation for knowledge, see [49]
4Coherentist approaches like this were also defended by the philosopher of science Otto Neurath [42] and
by the epistemologist Laurence BonJour [6], among others.

3

Quine’s indeterminacy theses raise a historic, philosophical challenge to the simple notion that mere
participation in a language game affords one access to the concepts that one’s interlocutor intends.
We argue that it is not the performance of language itself but a shared conceptual structure that
should ground our determination of mutual understanding, and it is this very conceptual structure
that we must aim to align between humans and AI models like ChatGPT.

4
How do humans learn concepts?

In cognitive science, as in philosophy, language and concepts are deeply intertwined. Human concept learning is a frequently-debated subject in cognitive science and developmental psychology,
and there are many open questions about how human children acquire and understand novel concepts. One prominent theory, proposed by the developmental psychologist Susan Carey and known
as Quinian bootstrapping [10], posits that words are symbolic placeholders that children ﬁrst learn
and which gradually ﬁll with meaning over time. Carey argues that the placeholder words are ﬁlled
with meaning primarily through analogical reasoning [10], while others have argued that they may
be ﬁlled through a variety of other mechanisms [29, 5].

Human concept learning is therefore also inextricably linked to the symbol grounding problem [24,
57], identiﬁed in cognitive science, philosophy, and robotics as the challenge of linking symbols
(like words) to the real-world phenomena they represent.

If we think of words as placeholders for conceptual knowledge, veritable buckets to be ﬁlled by
sensory and higher-level experiences (like thoughts, feelings, and memories) that deﬁne a concept,
then the concept alignment problem becomes suddenly more concrete and accessible. Although
two people probably have slightly different sensory information tied to the word “apple” (you may
immediately think of a red, sweet one while I think of a green, tart one), their understanding of it is
aligned enough to be able to communicate effectively about it [37]. In their discourse they can safely
trust that their companion understands what they mean by the word “apple” because the concept is
grounded by similar sensorimotor, socio-cultural, and linguistic experiences on both sides. How do
we achieve this for AI?

Even between humans, grounding is often insufﬁcient to ensure that we have aligned notions of a
complex concept, like love or democracy, for example. But even in those cases, grounded language
is a good starting point for concept-level alignment, because it allows us to discuss and come to
a clearer understanding. Shared experience and shared grounded language forms the basis of our
collective ability to understand the same concepts as someone else and make certain assumptions
about what someone means when they use a word.

Can we make those same assumptions about AI? Not yet, but we argue that we can get there by
leveraging empirical tools already being developed in cognitive science and AI research.

5
How do machines learn concepts?

Probing machine understanding of concepts has been the domain of subﬁelds like interpretability,
multimodal learning, and language grounding [62, 14]. Each of these ﬁelds have developed methods
of measuring to what extent words or labels are successfully linked to the real-world phenomena they
describe.

Neural networks form distributed representations, and it has proven difﬁcult to map those to humanunderstandable concepts. In the context of deep neural networks, these “representations” are usually
vectors of learned weights and biases in high-dimensional latent spaces. Researchers often measure
“representational alignment” between multiple models or models and humans by using tools such
as representational similarity analysis [32] and centered kernel alignment [31] which compare the
representational geometries of two neural systems. Much of the literature in deep representation
learning focuses on improving model performance by increasing the representational alignment of a
student model with a skilled teacher (which can be a human or another model) using methods such
as contrastive learning [11] and knowledge distillation [27]. For example, analyzing representations
might help us discover if the vector pairs for “man”-“king” and “woman”-“queen” are similarly
positioned relative to each other in latent space [64]. This representational analysis would suggest
that a language model may have recovered some degree of human-like conceptual structure about the

4

phenomena those words represent. However, while such representational correspondences provide
clues to what is going on at the conceptual level, they do not guarantee that models are actually using
human-understandable concepts in a human-like way [45]. Analyzing representations can, however,
give us insight into how a neural network solves a speciﬁc problem, and help us begin to understand
the relationship between representational alignment and concept alignment.

Within interpretability, which started with simple tools like saliency maps for convolutional neural
networks in computer vision [59, 4], the goal has always been to characterize machine representations or outputs in human-understandable terms. Saliency maps show which pixels of the original
image are the most salient in the classiﬁcation decision made by the model, thus linking the visual
input modality to the textual modality of the classiﬁcation label. For years, adversarial machine
learning was the prime example of machine intelligence gone wrong, where changing a few imperceptible (to humans) pixels could completely fool an AI model into miscategorizing an image [34].
Slowly, tools were developed to make models robust against such attacks. Interpretability work
showed that robustiﬁed classiﬁers had more interpretable saliency maps [63, 17], suggesting that
having a more human-like (interpretable) conceptual representation meant improved model performance and protection from adversarial attacks. Other interpretability work has made the importance
of concept-level alignment even more clear by showing that concept-level interpretability methods
are simply more effective than pixel-based interpretability methods. There are even developed tools
like Concept Activation Vectors [30] to probe a model’s conceptual understanding.

The trend towards generalist models operating in many modalities [51, 52, 15] in recent years is
beneﬁcial to concept alignment research, because concept alignment is fundmentally multimodal.
Many generalist, multimodal models have already provided proof by strong performance that a
version of language-based bootstrapping is also incredibly effective for training deep multimodal
AI models. Joint vision and language models like CLIP, DALL-E, and Imagen [56], are already
providing visual grounding to large text encoders. This brings us one step closer to shared grounded
language with AI, because it allows for a kind of sensory meaning of what an AI system means
when it says “apple.” This type of proof, grounded in a sensory modality like vision, is a ﬁrst step
towards human-like grounding.

For example, the generative model Imagen, which takes in text prompts and generates images to
match, uses a ﬁxed T5 encoder as the LLM component, and then learns to map the correct visual
information to an already rich textual embedding space of words. Other models like Parti [67]
start with a pretrained text encoder but then also ﬁnetune the LLM component further when trained
jointly with visual input. There is an analogy here to Quinian bootstrapping, where the words
are placeholders (although not empty ones, when they have pretrained text embeddings), and the
meaning of the words is slowly ﬁlled and enriched through information from the other modalities,
in this case through vision.

6
How do humans align on concepts?

Concept alignment between humans is not a static process. Shared, conceptually-groundedlanguage
gives human agents the ability to make reasonable assumptions about each others’ intended meanings. Humans further hone their concept alignment through interaction. While language is perhaps
the most important way that humans align on concepts, there are important subtleties in this process
occurring under the surface of verbal interactions between human agents.

The past several decades has seen the emergence of behavioral and neuroscientiﬁc evidence in cognitive science for the role of interaction in human communication and comprehension [44, 66]. A
cluster of effects that have been studied throughout the years, such as “behavioral matching,” “imitation,” “mimicry,” “entrainment,” “repetition,” “accommodation,” “coordination,” and “joint-action,”
is currently coalescing into a single, coherent cognitive phenomenon referred to in the cognitive
science literature as “alignment” [53].

The term “alignment” in this context is drawn from the interactive alignment paradigm [48], an inﬂuential theory in cognitive science which argues that successful human communication is facilitated

5

by aligning linguistic processing across multiple scales.5 This approach has since been extended by
the analysis of human concept alignment qua interpersonal synergies, an analysis of communicative
coordination using dynamical systems theory [36, 13, 18, 16]. Interactive adaptation of this kind has
been shown to be crucial for resolving misunderstanding and more accurately ﬁne-tuning aligned
conceptual understandings during real-time problem solving and coordination [19].

Drawing on these paradigms, a range of measures have been developed in cognitive science for
measuring human alignment. For example, Abney et al. (2015) develop a measure of synergy
in coordination that predicts performance in a cooperative task during problem solving [3]. This
ﬁnding indicates that measures of synergy might be useful for evaluating concept alignment between
humans and AI.

Additionally, maximizing information exchange between two complex systems (i.e., increasing the
ability for the dynamics of each system to affect, and be affected by, the other, also known as
“resonance” [65]) may be a key driver of concept alignment. Complexity matching, a measure
of such information exchange,6 has been shown to increase between participants in an afﬁliative
conversation and decrease between partners in an argumentative conversation [2]. This provides
empirical evidence for a connection between information exchange and concept alignment, making
measures such as complexity matching another promising measure of concept alignment.

We argue that in addition to grounded linguistic meaning, human-AI alignment must be dynamic and
adaptive in order to ﬁne-tune shared conceptual understanding. Interaction serves to align diverse
experiences and perspectives between agents and create paths for shared meaning to take hold. Creating AI systems that are capable of engaging in and reciprocating this sort of multiscale, interactive
adaptation with their interlocutors will be an important step towards human-aligned AI that reliably
meets the standard for concept alignment that we would reasonably expect of a human agent.

7
How might humans and machines align on concepts?

It is clear that the standard for proof of human-AI alignment must be even higher than it is for humanhuman alignment, because we can make more assumptions about another human’s understanding of
a shared word than we can about, say, ChatGPT. We have more reason to believe an AI might
be misaligned due to the lack of this shared embodiment, shared experience, and shared linguistic
grounding.

To develop a standard for measuring, steering, and improving concept alignment, we must start
with the seeds laid already within the AI research space, often under areas such as interpretability,
language grounding, grounded robotics, and multimodal models.

Promisingly, it is clear that bootstrapping from LLM representations and then grounding in other
modalities is a powerful combination for AI systems. Newer robotics models like PaLM-E [15] start
with LLM representations and then enrich those with sensory information that the robot experiences.
The sensory information improves the word representations, and the rich existing knowledge embedded in the LLM helps the robot bootstrap instead of relying solely on sensorimotor experience as
a source of all its knowledge. This method is extremely successful, and PaLM-E achieved state-ofthe-art performance on many robot learning tasks. Interestingly, PaLM-E also became an excellent
vision-and-language model, apparently beneﬁting from the addition of sensorimotor modalities to
improve its performance in the vision-language space. This is a ﬁrst step towards the kind of concept
alignment that we need – ﬁrst bootstrapping using pretrained LLMs and then ﬁlling words with conceptual meaning grounded in other modalities should lead to more veriﬁably human-like concepts
and better performance. Different modalities provide ways for an AI system to demonstrate, through
an image or a sound, that it understands a concept in a human-like way.

There is clearly still work to be done in designing AI systems that are interpretable and grounded at
the concept level. Further inquiry along these lines is critical to achieve concept alignment. Techni
5Lexical entrainment (that interlocutors will use or repeat similar words throughout the course of a conversation, [7, 8]) and syntactic priming (that interlocutors will use or repeat similar syntactic structures throughout
the course of a conversation, [47]) are key examples of alignment in this sense.
6Complexity matching is measured using time series data, and thus may be implemented in any modality
from which a time series can be extracted.

6

cal tools for probing the effects of language indeterminacy must be developed to measure, stress-test,
and validate claims of concept alignment.

We argue that this work should be done in conversation with and guided by the cognitive science
research in concept formation and alignment. What degree of shared conceptual understanding is
required, or even commonly present, between interacting humans is something we can only fully
characterize through empirical studies of human agents. We can then design AI to have, at a lower
bound, this degree of shared conceptual understanding. This is a complex and iterative process
requiring ongoing studies and engineering across modalities.

For example, say we determine through child development studies that visual recognition of an
“apple” is usually learned in the ﬁrst 18 months of development, that sensorimotor knowledge of
holding, biting, and tasting an apple is usually learned within the ﬁrst 36 months, and that reliably
using the word correctly in a linguistic context is learned within the ﬁrst ﬁve years. In addition, we
may ﬁnd that most adult humans think of a red, sweet apple when they think of apples, but that a
large proportion also think of Granny Smiths, and a smaller proportion of yellow delicious apples. In
cognitive science research, prototype theory [55] has helped us extensively explore such ideas about
the prototypicality of particular instantiations of a concept – whether a red apple is more prototypical
than a green one, for example. Cognitive science helps us set a standard for what a fellow human
might assume our shared understanding of the concept “apple” would be, a standard which can then
be applied to machines. This standard will change for more complex concepts, but it will always be
grounded in what can be reasonably expected of humans. This isn’t a perfect process of course; for
example, a fellow human who hails from a geographic region without access to apples might lack this
shared understanding, and they would still be able to interactively communicate this, which is why
the interactive and adaptive component is so important. However, aiming for shared understanding
in a way that approximates shared understanding between humans is a good place to start. The
beauty of concept alignment is that it allows us to reliably set goals for AI systems and tractably
measure performance against these goals, because concepts are intrinsically human-understandable
and can be tractably mapped to standards expected of a reasonable human agent. The standards for
concept understanding and alignment expected of a reasonable human agent should continue to be
characterized by a vital ongoing dialogue between cognitive science research and AI research.

A human agent’s understanding of concepts is also ﬁne-tuned through interaction [39], and humans
can even use different understandings of the same concept based on context (for example, if I explain
to you that I’m talking about a green apple, it’s easy for you to make the switch even if you were
thinking about a red one before). It is important that we also get AI systems to the stage of concept
alignment where they can reliably interact with humans to ﬁne-tune their conceptual understandings
as a reasonable human would. Fine-tuning through interaction can be informed by work in humanin-the-loop-learning (HILL) in the machine learning literature [35], and work in cognitive alignment
in the cognitive science literature [41, 25]. Among AI researchers, one popular way of approaching
interaction currently is reinforcement learning from human feedback (RLHF) [69, 43], which asks
human participants to rank instances of an AI agent’s behavior and uses this feedback to train a
reward model to optimize the AI agent’s policy. However, RLHF currently operates at the behavioral
level. We don’t know exactly how policy changes resulting from the human feedback ﬁlter through
to the AI systems’ understanding of concepts, and speciﬁcally whether RLHF in its current form
encourages the AI to form human-like concepts at all. RLHF is an interesting start, particularly
because it incorporates human preferences through interaction. However, more work along these
lines, focused at the concept-level, is necessary to interactively improve concept alignment through
human feedback. Verifying and improving alignment through conversation and nonverbal cues is
critical to successful human collaboration. Concept alignment between humans and AI would be
incomplete without this capacity for interactive ﬁne-tuning and improvement.

8
Conclusion: Where do we go from here?

We have introduced the idea of concept alignment between humans and AI, explained why it is an
important prerequisite to discussions of value alignment, and charted a path forward towards concept
alignment. In particular, we argue that:

7

• Concept alignment across sensory modalities that are part of the human experience will
lead to more human-aligned natural language representations (vision, textual language,
speech/audio, olfaction, touch, vestibular and limbic systems, etc.)

• Language is a vital, but not sufﬁcient, part of concept alignment.
Joint training in generalist models can use concept alignment veriﬁcation as a feedback
loop for improving and grounding natural language representations, aligning them with
shared meaning expected of a reasonable human agent.

• Concept alignment will lead to natural language communication with AI systems that is
functionally equivalent to natural language communication among humans.

Concept alignment, and the shared grounded language that will emerge from it, will lay the groundwork for value alignment, behavior alignment, and other important subgoals that are an integral step
on the journey towards AI alignment.

Acknowledgements

The authors would like to thank the Diverse Intelligences Summer Institute (DISI) and funding
from the Templeton World Charity Foundation (https://doi.org/10.54224/20333) for making this
collaboration possible.

References

[1] https://openai.com/blog/chatgpt/, 2022. ChatGPT, OpenAI.

[2] Drew H Abney, Alexandra Paxton, Rick Dale, and Christopher T Kello. Complexity matching
in dyadic conversation. Journal of Experimental Psychology: General, 143(6):2304, 2014.

[3] Drew H Abney, Alexandra Paxton, Rick Dale, and Christopher T Kello. Movement dynamics reﬂect a functional role for weak coupling and role structure in dyadic problem solving.
Cognitive processing, 16:325–332, 2015.

[4] Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim.
Sanity checks for saliency maps. Advances in neural information processing systems, 31, 2018.

[5] Jacob Beck. Can bootstrapping explain concept learning? Cognition, 158:110–121, 2017.

[6] Laurence BonJour. The Structure of Empirical Knowledge. Harvard University Press, Cambridge, Mass., 1985.

[7] Susan E Brennan and Herbert H Clark. Conceptual pacts and lexical choice in conversation.
Journal of experimental psychology: Learning, memory, and cognition, 22(6):1482, 1996.

[8] Susan E Brennan et al. Lexical entrainment in spontaneous dialog. Proceedings of ISSD,
96:41–44, 1996.

[9] Daniel S Brown, Jordan Schneider, Anca Dragan, and Scott Niekum. Value alignment veriﬁcation. In International Conference on Machine Learning, pages 1105–1115. PMLR, 2021.

[10] Susan Carey.
The Origin of Concepts.
Oxford Series in Cognitive Development. Oxford
University Press, USA, 2009.

[11] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework
for contrastive learning of visual representations.
In International conference on machine
learning, pages 1597–1607. PMLR, 2020.

[12] B. Christian. The Alignment Problem: Machine Learning and Human Values. WW Norton,
2020.

[13] Rick Dale, Riccardo Fusaroli, Nicholas D Duran, and Daniel C Richardson.
The selforganization of human interaction.
In Psychology of learning and motivation, volume 59,
pages 43–95. Elsevier, 2013.

[14] Finale Doshi-Velez and Been Kim. Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608, 2017.

8

[15] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter,
Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied
multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

[16] Nicholas D Duran and Rick Dale. Perspective-taking in dialogue as self-organization under
social constraints. New Ideas in Psychology, 32:131–146, 2014.

[17] Christian Etmann, Sebastian Lunz, Peter Maass, and Carola-Bibiane Schönlieb.
On the
connection between adversarial robustness and saliency map interpretability. arXiv preprint
arXiv:1905.04172, 2019.

[18] Riccardo Fusaroli, Drew H Abney, Bahador Bahrami, Christopher T Kello, and Kristian Tylén.
Conversation, coupling and complexity: Matching scaling laws predict performance in a joint
decision task. In Poster presented at the 35th annual conference of the cognitive science society,
2013.

[19] Riccardo Fusaroli, Joanna R ˛aczaszek-Leonardi, and Kristian Tylén. Dialog as interpersonal
synergy. New Ideas in Psychology, 32:147–157, 2014.

[20] Iason Gabriel. Artiﬁcial intelligence, values, and alignment. Minds and machines, 30(3):411–
437, 2020.

[21] Richard
Gonzales.
Feds
say
self-driving
uber
suv
did
not
recognize
jaywalking
pedestrian
in
fatal
crash.
https://www.npr.org/2019/11/07/777438412/feds-say-self-driving-uber-suv-did-not-recognize-ja
2019. National Public Radio (NPR).

[22] Dylan Hadﬁeld-Menell, Smitha Milli, Pieter Abbeel, Stuart J Russell, and Anca Dragan. Inverse reward design. Advances in neural information processing systems, 30, 2017.

[23] Dylan Hadﬁeld-Menell, Stuart J Russell, Pieter Abbeel, and Anca Dragan. Cooperative inverse
reinforcement learning. Advances in neural information processing systems, 29, 2016.

[24] Stevan Harnad. The symbol grounding problem. Physica D: Nonlinear Phenomena, 42(13):335–346, 1990.

[25] Robert D Hawkins, Michael Franke, Michael C Frank, Adele E Goldberg, Kenny Smith,
Thomas L Grifﬁths, and Noah D Goodman. From partners to populations: A hierarchical
bayesian account of coordination and convention. Psychological Review, 130(4):977, 2023.

[26] Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob
Steinhardt. Aligning ai with shared human values. arXiv preprint arXiv:2008.02275, 2020.

[27] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network.
arXiv preprint arXiv:1503.02531, 2015.

[28] Matthew Hutson. Can we stop runaway a.i.? https://www.newyorker.com/science/annals-of-artificial-int
2023. The New Yorker.

[29] Zoltan Jakab. How to improve on quinian bootstrapping-a response to nativist objections. In
Proceedings of the Annual Meeting of the Cognitive Science Society, volume 35, 2013.

[30] Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, et al.
Interpretability beyond feature attribution: Quantitative testing with concept activation vectors
(tcav). In International conference on machine learning, pages 2668–2677. PMLR, 2018.

[31] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of
neural network representations revisited. In International conference on machine learning,
pages 3519–3529. PMLR, 2019.

[32] Nikolaus Kriegeskorte, Marieke Mur, and Peter A Bandettini.
Representational similarity
analysis-connecting the branches of systems neuroscience. Frontiers in systems neuroscience,
page 4, 2008.

[33] Thomas S. Kuhn. The essential tension. Philosophy of Science, 45(4):649–652, 1978.

[34] Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial machine learning at scale.
arXiv preprint arXiv:1611.01236, 2016.

[35] Jiwei Li, Alexander H Miller, Sumit Chopra, Marc’Aurelio Ranzato, and Jason Weston. Dialogue learning with human-in-the-loop. In 5th International Conference on Learning Representations, ICLR 2017, 2017.

9

[36] Max M Louwerse, Rick Dale, Ellen G Bard, and Patrick Jeuniaux. Behavior matching in
multimodal communication is synchronized. Cognitive science, 36(8):1404–1426, 2012.
[37] Gary Lupyan, Ryutaro Uchiyama, Bill Thompson, and Daniel Casasanto. Hidden differences
in phenomenal experience. Cognitive Science, 47(1):e13239, 2023.
[38] Ryan Mac.
Facebook apologizes after a.i. puts ‘primates’ label on video of black men.
https://www.nytimes.com/2021/09/03/technology/facebook-ai-race-primates.html,
2021. The New York Times.
[39] Jean Matter Mandler. The foundations of mind: Origins of conceptual thought. Oxford University Press, 2004.
[40] Cade
Metz.
How
google’s
ai
viewed
the
move no
human
could
understand.
https://www.wired.com/2016/03/googles-ai-viewed-move-no-human-understand/,
2016. WIRED.
[41] Sonia K Murthy, Thomas L Grifﬁths, and Robert D Hawkins. Shades of confusion: Lexical
uncertainty modulates ad hoc coordination in an interactive communication task. Cognition,
225:105152, 2022.
[42] Otto Neurath. Protocol Statements, page 91–99. Springer Netherlands, Dordrecht, 1983.
[43] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to
follow instructions with human feedback. Advances in Neural Information Processing Systems,
35:27730–27744, 2022.
[44] Alexandra Paxton, Rick Dale, and Daniel C Richardson. Social coordination of verbal and
nonverbal behaviours. Interpersonal coordination and performance in social systems, page
259, 2016.
[45] JC Peterson, D Chen, and TL Grifﬁths. Parallelograms revisited: Exploring the limitations of
vector space models for simple analogies. cognition, 205, article 104440, 2020.
[46] J Piaget and B Inhelder. Systems of reference and horizontal–vertical coordinates. The Child’s
Conception of Space (1967), pages 375–418, 1967.
[47] Martin J Pickering and Holly P Branigan. Syntactic priming in language production. Trends
in cognitive sciences, 3(4):136–141, 1999.
[48] Martin J Pickering and Simon Garrod. Toward a mechanistic psychology of dialogue. Behavioral and brain sciences, 27(2):169–190, 2004.
[49] Willard V. O. Quine. Two dogmas of empiricism. Philosophical Review, 60(1):20–43, 1951.
[50] W.V.O. Quine. Ontological relativity. Journal of Philosophy, 65(7):185–212, 1968.
[51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable
visual models from natural language supervision. In International conference on machine
learning, pages 8748–8763. PMLR, 2021.
[52] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark
Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on
Machine Learning, pages 8821–8831. PMLR, 2021.
[53] Marlou Rasenberg, Asli Özyürek, and Mark Dingemanse. Alignment in multimodal interaction: An integrative framework. Cognitive science, 44(11):e12911, 2020.
[54] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. " why should i trust you?" explaining
the predictions of any classiﬁer.
In Proceedings of the 22nd ACM SIGKDD international
conference on knowledge discovery and data mining, pages 1135–1144, 2016.
[55] Eleanor H Rosch. Natural categories. Cognitive psychology, 4(3):328–350, 1973.
[56] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural
Information Processing Systems, 35:36479–36494, 2022.
[57] John R Searle. Minds, brains, and programs. Behavioral and brain sciences, 3(3):417–424,
1980.

10

[58] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van
Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al.
Mastering the game of go with deep neural networks and tree search.
nature,
529(7587):484–489, 2016.
[59] K Simonyan, A Vedaldi, and A Zisserman. Deep inside convolutional networks: visualising
image classiﬁcation models and saliency maps. In Proceedings of the International Conference
on Learning Representations (ICLR). ICLR, 2014.
[60] SingularityGroup. from risk to reward: the role of ai alignment in shaping a positive future.
https://www.su.org/blog/from-risk-to-reward-the-role-of-ai-alignment-in-shaping-a-positive2023.
[61] Charles Taylor. Rationality. In Martin Hollis and Steven Lukes, editors, Rationality and Relativism, pages 87–105. MIT Press, 1982.
[62] Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J Zico Kolter, Louis-Philippe Morency, and
Ruslan Salakhutdinov. Multimodal transformer for unaligned multimodal language sequences.
In Proceedings of the conference. Association for Computational Linguistics. Meeting, volume
2019, page 6558. NIH Public Access, 2019.

[63] Dimitris Tsipras, Shibani Santurkar, Logan Engstrom, Alexander Turner, and Aleksander
Madry. Robustness may be at odds with accuracy. arXiv preprint arXiv:1805.12152, 2018.
[64] Ekaterina Vylomova, Laura Rimell, Trevor Cohn, and Timothy Baldwin. Take and took, gaggle and goose, book and read: Evaluating the utility of vector differences for lexical relation
learning. arXiv preprint arXiv:1509.01692, 2015.

[65] Bruce J West, Elvis L Geneston, and Paolo Grigolini.
Maximizing information exchange
between complex networks. Physics Reports, 468(1-3):1–99, 2008.

[66] Thalia Wheatley, Adam Boncz, Ivan Toni, and Arjen Stolk. Beyond the isolated brain: The
promise and challenge of interacting minds. Neuron, 103(2):186–188, 2019.
[67] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for
content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.

[68] Ziqian Zhong, Ziming Liu, Max Tegmark, and Jacob Andreas. The clock and the pizza: Two
stories in mechanistic explanation of neural networks. arXiv preprint arXiv:2306.17844, 2023.

[69] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei,
Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences.
arXiv preprint arXiv:1909.08593, 2019.

11

---

*Source: arXiv:2401.08672 / ACM Computing Surveys*
