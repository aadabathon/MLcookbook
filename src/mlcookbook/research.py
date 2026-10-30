"""A curated AI history and links to explicitly selected local readings.

The dates describe publications or named project events, not a claim that an
idea had a single inventor. Ancient logic and mechanical calculation are
precursors; the Dartmouth proposal (1955) and meeting (1956) name a field.
This selected timeline ends with the 2025 Qwen3 report in Adam's collection.
It is neither an exhaustive world history nor a feed of current AI news.

The catalog is ordinary data. Opening it performs no network request, PDF
parsing, model loading, or scientific-library import. Local PDFs stay external.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

_PAPERS = [
    {
        "id": "transformer", "title": "Attention Is All You Need", "year": 2017,
        "filename": "attention is all you need.pdf", "kind": "Paper",
        "url": "https://arxiv.org/abs/1706.03762",
        "summary": "The Transformer architecture: attention, feed-forward layers, positional information and parallel sequence processing.",
    },
    {
        "id": "matrix-calculus", "title": "The Matrix Calculus You Need For Deep Learning", "year": 2018,
        "filename": "The matrix calculus you need for deep learning.pdf", "kind": "Explainer",
        "url": "https://arxiv.org/abs/1802.01528",
        "summary": "Terence Parr and Jeremy Howard explain derivatives, Jacobians and the chain rule behind neural-network training.",
    },
    {
        "id": "illustrated-transformer", "title": "The Illustrated Transformer", "year": 2018,
        "filename": "The Illustrated Transformer.pdf", "kind": "Explainer",
        "url": "https://jalammar.github.io/illustrated-transformer/",
        "summary": "Jay Alammar's visual explanation of Transformer components. The date refers to the article, not the later PDF print export.",
    },
    {
        "id": "gpt3", "title": "Language Models are Few-Shot Learners", "year": 2020,
        "filename": "languages models are few-shot learners.pdf", "kind": "Paper",
        "url": "https://arxiv.org/abs/2005.14165",
        "summary": "The GPT-3 study evaluates task examples supplied in a prompt without updating model weights for each task.",
    },
    {
        "id": "gqa", "title": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "year": 2023,
        "filename": "GQA-- training generalized Multi-Query transformer models from multi-head checkpoints.pdf", "kind": "Paper",
        "url": "https://arxiv.org/abs/2305.13245",
        "summary": "Grouped-query attention shares key/value heads between query heads to balance quality and decoder inference cost.",
    },
    {
        "id": "qwen3", "title": "Qwen3 Technical Report", "year": 2025,
        "filename": "qwen3 technical report.pdf", "kind": "Paper",
        "url": "https://arxiv.org/abs/2505.09388",
        "summary": "The Qwen3 report covers dense and mixture-of-experts models, training and thinking/non-thinking modes.",
    },
]

_ENTRIES = [
    {
        "id": "aristotle", "year": -350, "date": "c. 350 BCE", "era": "Foundations",
        "title": "Aristotle: formal logic and syllogisms", "people": "Aristotle",
        "summary": "Prior Analytics studies syllogisms: conclusions that follow from structured premises.",
        "significance": "Making the structure of an argument explicit creates a way to inspect its validity. This selected lineage begins with a surviving formal account; human traditions of reasoning extend much further.",
        "connections": ["Distinguish a rule's logical validity from whether its premises describe the data."],
        "sources": [{"title": "Prior Analytics, translated by A. J. Jenkinson", "url": "https://classics.mit.edu/Aristotle/prior.html"}],
        "paper_ids": [],
    },
    {
        "id": "pascal", "year": 1642, "date": "1642", "era": "Foundations",
        "title": "The Pascaline mechanical calculator", "people": "Blaise Pascal",
        "summary": "Pascal develops a mechanical calculator that automates addition through wheels and carries.",
        "significance": "Mechanical calculation shows how a repeatable human procedure can be embodied in a machine. It brings arithmetic, representation and physical mechanism together.",
        "connections": ["Accelerators also embody selected operations; efficient arithmetic alone does not define intelligence."],
        "sources": [{"title": "Whipple Museum: mechanical calculation", "url": "https://www.whipplemuseum.cam.ac.uk/explore-whipple-collections/calculating-devices/mechanical-calculation"}],
        "paper_ids": [],
    },
    {
        "id": "bayes", "year": 1763, "date": "1763", "era": "Foundations",
        "title": "Bayes's essay on probability and evidence", "people": "Thomas Bayes; communicated by Richard Price",
        "summary": "Bayes's posthumous essay addresses inference about an unknown probability from observations.",
        "significance": "Probability supplies a language for learning from incomplete evidence. Later Bayesian methods have a much larger history than this single essay.",
        "connections": ["Read probabilistic predictions as conditional on data and assumptions, rather than as logical certainties."],
        "sources": [{"title": "An Essay towards solving a Problem in the Doctrine of Chances", "url": "https://www.york.ac.uk/depts/maths/histstat/essay.htm"}],
        "paper_ids": [],
    },
    {
        "id": "lovelace", "year": 1843, "date": "1843", "era": "Foundations",
        "title": "Lovelace and the programmable Analytical Engine", "people": "Ada Lovelace; Charles Babbage; Luigi Menabrea",
        "summary": "Lovelace's notes on the proposed Analytical Engine describe sequences of operations and possibilities beyond numerical tables.",
        "significance": "The notes separate a general program from a machine built for one calculation. They also question what a machine can originate; the complete proposed engine was not built in their lifetimes.",
        "connections": ["A compute graph expresses operations and dependencies separately from the device executing them."],
        "sources": [{"title": "Sketch of the Analytical Engine: notes by the translator (1843)", "url": "https://www.yorku.ca/pclassic/Lovelace/lovelace.htm"}],
        "paper_ids": [],
    },
    {
        "id": "boole", "year": 1854, "date": "1854", "era": "Foundations",
        "title": "Boolean algebra and The Laws of Thought", "people": "George Boole",
        "summary": "The Laws of Thought develops an algebraic treatment of logical propositions and probability.",
        "significance": "Symbolic algebra makes reasoning rules amenable to formal manipulation. The later use of Boolean algebra in switching circuits is a separate engineering development.",
        "connections": ["Boolean conditions appear in dataset filters, program control and digital circuits."],
        "sources": [{"title": "An Investigation of the Laws of Thought (original text)", "url": "https://www.gutenberg.org/ebooks/15114"}],
        "paper_ids": [],
    },
    {
        "id": "pearson-pca", "year": 1901, "date": "1901", "era": "Foundations",
        "title": "Pearson and the geometry behind PCA", "people": "Karl Pearson",
        "summary": "Pearson studies lines and planes that best fit a cloud of points, part of the mathematical history of principal components.",
        "significance": "Representing data in fewer dimensions connects statistics with geometry. This historical paper is a landmark in a method with multiple contributors, not the whole origin of dimensionality reduction.",
        "connections": ["The PCA recipe exposes scores, loadings and explained variance; scaling changes the geometry."],
        "sources": [{"title": "On Lines and Planes of Closest Fit to Systems of Points in Space", "url": "https://doi.org/10.1080/14786440109462720"}],
        "paper_ids": [],
    },
    {
        "id": "turing-computability", "year": 1936, "date": "1936", "era": "Foundations",
        "title": "Turing's model of computation", "people": "Alan Turing",
        "summary": "On Computable Numbers models finite mechanical procedures and establishes limits on what such procedures can decide.",
        "significance": "A general mathematical account of computation makes programs and their limitations precise. Computability says what can be done in principle, not how quickly or economically.",
        "connections": ["Separate whether a model can execute from its latency, memory demand and practical hardware cost."],
        "sources": [{"title": "On Computable Numbers, with an Application to the Entscheidungsproblem", "url": "https://www.cs.virginia.edu/~robins/Turing_Paper_1936.pdf"}],
        "paper_ids": [],
    },
    {
        "id": "shannon-switching", "year": 1938, "date": "1938", "era": "Foundations",
        "title": "Shannon's algebra of switching circuits", "people": "Claude Shannon",
        "summary": "Shannon's published analysis represents relay and switching networks with algebraic equations.",
        "significance": "The work links logical structure with circuit design and simplification. This date identifies the published paper rather than treating the thesis and publication as one event.",
        "connections": ["Logic-level equivalence and physical implementation cost are different questions in accelerator design."],
        "sources": [{"title": "A Symbolic Analysis of Relay and Switching Circuits", "url": "https://paradise.caltech.edu/ist4/lectures/shannon38.pdf"}],
        "paper_ids": [],
    },
    {
        "id": "mcculloch-pitts", "year": 1943, "date": "1943", "era": "Foundations",
        "title": "The McCulloch-Pitts model neuron", "people": "Warren McCulloch; Walter Pitts",
        "summary": "A mathematical model of threshold-like neurons studies the logical computations that networks can express.",
        "significance": "The paper connects neural inspiration and formal computation through simplified threshold elements. It sets up a distinction between the computations a network can express and the way its connections are learned.",
        "connections": ["Neural-network graphs connect primitive operations; model expressivity and the procedure that learns weights remain distinct."],
        "sources": [{"title": "A Logical Calculus of the Ideas Immanent in Nervous Activity", "url": "https://doi.org/10.1007/BF02478259"}],
        "paper_ids": [],
    },
    {
        "id": "hebb", "year": 1949, "date": "1949", "era": "Foundations",
        "title": "Hebbian learning and cell assemblies", "people": "Donald O. Hebb",
        "summary": "The Organization of Behavior proposes that coordinated neural activity can strengthen connections and organize cell assemblies.",
        "significance": "Learning becomes a question about changes within a network, rather than only the rules supplied by a programmer. Hebbian ideas and modern backpropagation are different learning mechanisms.",
        "connections": ["Keep inference through fixed weights separate from training that changes those weights."],
        "sources": [{"title": "The Organization of Behavior (1949 edition)", "url": "https://pure.mpg.de/pubman/item/item_2346268_3/component/file_2346267/Hebb_1949_The_Organization_of_Behavior.pdf"}],
        "paper_ids": [],
    },
    {
        "id": "turing-imitation", "year": 1950, "date": "1950", "era": "Early AI",
        "title": "Turing's imitation game", "people": "Alan Turing",
        "summary": "Computing Machinery and Intelligence develops the imitation game and discusses learning machines and objections to machine intelligence.",
        "significance": "The paper turns a broad question about thinking into a concrete conversational experiment. Its discussion of learning machines helps frame intelligence as something to investigate through stated tests.",
        "connections": ["Choose an evaluation that measures the task of interest; fluent text is not evidence for every capability."],
        "sources": [{"title": "Computing Machinery and Intelligence", "url": "https://doi.org/10.1093/mind/LIX.236.433"}],
        "paper_ids": [],
    },
    {
        "id": "dartmouth", "year": 1956, "date": "1955–1956", "era": "Early AI",
        "title": "Dartmouth and the name artificial intelligence", "people": "John McCarthy; Marvin Minsky; Nathaniel Rochester; Claude Shannon",
        "summary": "The August 1955 proposal names artificial intelligence and proposes the Dartmouth summer research project held in 1956.",
        "significance": "Dartmouth is a conventional founding landmark for AI as an organized field, bringing earlier work on logic, learning and computation under a shared research agenda.",
        "connections": ["AI includes symbolic reasoning, search and learning; its history is broader than neural language models."],
        "sources": [{"title": "A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence (1955)", "url": "https://www-formal.stanford.edu/jmc/history/dartmouth/dartmouth.html"}],
        "paper_ids": [],
    },
    {
        "id": "perceptron", "year": 1958, "date": "1958", "era": "Early AI",
        "title": "Rosenblatt's perceptron", "people": "Frank Rosenblatt",
        "summary": "Rosenblatt's perceptron paper studies a system that learns connections for recognizing patterns.",
        "significance": "The perceptron makes adaptation of numerical parameters central to recognition. Its architecture and capabilities differ from the multilayer networks used today.",
        "connections": ["The Neural Networks recipe uses a multilayer perceptron; architecture, optimization and validation each affect its result."],
        "sources": [{"title": "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain", "url": "https://web.stanford.edu/class/psych209a/ReadingsByDate/01_30/Rosenblatt58Perceptron.pdf"}],
        "paper_ids": [],
    },
    {
        "id": "samuel", "year": 1959, "date": "1959", "era": "Early AI",
        "title": "Samuel's learning checkers program", "people": "Arthur Samuel",
        "summary": "Samuel describes checkers programs that use experience to improve evaluation and play.",
        "significance": "The paper explicitly studies machine learning through a concrete task. Search, stored experience and learned evaluation can contribute different parts of a system's performance.",
        "connections": ["Compare a learned component with a fixed baseline, and record how much data and computation produced the improvement."],
        "sources": [{"title": "Some Studies in Machine Learning Using the Game of Checkers", "url": "https://doi.org/10.1147/rd.33.0210"}],
        "paper_ids": [],
    },
    {
        "id": "lisp", "year": 1960, "date": "1960", "era": "Early AI",
        "title": "Lisp and symbolic computation", "people": "John McCarthy",
        "summary": "McCarthy's published Lisp paper describes recursive functions and computation over symbolic expressions.",
        "significance": "Symbol manipulation gives AI researchers a way to represent and transform structured expressions. The publication date here follows the 1960 paper; Lisp's development began earlier.",
        "connections": ["A graph representation is data that a program can inspect and transform, independently of the numerical kernels it describes."],
        "sources": [{"title": "Recursive Functions of Symbolic Expressions and Their Computation by Machine", "url": "https://www-formal.stanford.edu/jmc/recursive.html"}],
        "paper_ids": [],
    },
    {
        "id": "dendral", "year": 1965, "date": "1965 onward", "era": "Early AI",
        "title": "DENDRAL: scientific reasoning with domain knowledge", "people": "Joshua Lederberg; Edward Feigenbaum; Bruce Buchanan and collaborators",
        "summary": "The DENDRAL project applies chemical knowledge and search to reasoning about molecular structures from experimental evidence.",
        "significance": "Specialized knowledge can constrain an otherwise large hypothesis space. The linked history is the project authors' retrospective, rather than a claim that the complete system appeared in one year.",
        "connections": ["A domain-specific system needs evidence and constraints appropriate to its scientific problem."],
        "sources": [{"title": "DENDRAL: A Case Study of the First Expert System for Scientific Hypothesis Formation (authors' retrospective)", "url": "https://deepblue.lib.umich.edu/bitstream/handle/2027.42/30758/0000409.pdf"}],
        "paper_ids": [],
    },
    {
        "id": "eliza", "year": 1966, "date": "1966", "era": "Early AI",
        "title": "ELIZA and scripted conversation", "people": "Joseph Weizenbaum",
        "summary": "ELIZA transforms user input with scripts and pattern-matching rules to conduct text conversations.",
        "significance": "A conversational interface can invite people to infer understanding beyond the mechanism's evidence. This program is a different approach from a trained neural language model.",
        "connections": ["Inspect the mechanism behind an answer and evaluate factual reliability separately from conversational style."],
        "sources": [{"title": "ELIZA: A Computer Program for the Study of Natural Language Communication between Man and Machine", "url": "https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf"}],
        "paper_ids": [],
    },
    {
        "id": "kmeans", "year": 1967, "date": "1967", "era": "Statistical learning",
        "title": "MacQueen's k-means clustering paper", "people": "James MacQueen",
        "summary": "MacQueen describes methods using k means for classifying and summarizing multivariate observations.",
        "significance": "The paper is one landmark in a clustering method with several earlier and parallel contributors. A compact partition can reveal structure without supplying ground-truth class labels.",
        "connections": ["The k-means recipe makes feature scaling, cluster count and initialization explicit."],
        "sources": [{"title": "Some Methods for Classification and Analysis of Multivariate Observations", "url": "https://projecteuclid.org/euclid.bsmsp/1200512992"}],
        "paper_ids": [],
    },
    {
        "id": "nearest-neighbor", "year": 1967, "date": "1967", "era": "Statistical learning",
        "title": "Nearest-neighbor classification", "people": "Thomas Cover; Peter Hart",
        "summary": "Cover and Hart analyze nearest-neighbor classification and its relation to the optimal classification error.",
        "significance": "Prediction can depend on comparisons with stored observations. Statistical guarantees come with assumptions and do not remove the cost of storing and searching examples.",
        "connections": ["Distance definitions and feature scales determine which examples count as similar."],
        "sources": [{"title": "Nearest Neighbor Pattern Classification", "url": "https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf"}],
        "paper_ids": [],
    },
    {
        "id": "lighthill", "year": 1973, "date": "1973", "era": "Early AI",
        "title": "The Lighthill report and the debate over AI", "people": "James Lighthill; John McCarthy and other respondents",
        "summary": "Lighthill's published survey criticizes the generality and scaling of AI methods; researchers dispute his conclusions.",
        "significance": "The exchange documents how evidence, combinatorial growth and expectations become contested. AI's periods of reduced confidence had multiple causes; this report is not a complete explanation of them.",
        "connections": ["Scaling claims need measured workloads, baselines and stated limits rather than demonstrations alone."],
        "sources": [
            {"title": "Artificial Intelligence: A General Survey (original report)", "url": "https://rodsmith.nz/wp-content/uploads/Lighthill_1973_Report.pdf"},
            {"title": "McCarthy's response to the Lighthill report", "url": "https://www-formal.stanford.edu/jmc/reviews/lighthill/lighthill.html"},
        ],
        "paper_ids": [],
    },
    {
        "id": "xcon", "year": 1980, "date": "1980", "era": "Early AI",
        "title": "R1/XCON enters computer manufacturing", "people": "John McDermott; Digital Equipment Corporation collaborators",
        "summary": "R1, later known as XCON, is used to configure VAX computer systems from explicit domain rules.",
        "significance": "The 1981 authors' account dates manufacturing use to January 1980. It illustrates a useful narrow deployment whose knowledge must be maintained, rather than general intelligence.",
        "connections": ["Representing constraints and maintaining domain knowledge are engineering work alongside executing the rules."],
        "sources": [{"title": "Extending a Knowledge-Based System to Deal with Ad Hoc Constraints (1981)", "url": "https://ijcai.org/proceedings/1981/157"}],
        "paper_ids": [],
    },
    {
        "id": "cart", "year": 1984, "date": "1984", "era": "Statistical learning",
        "title": "Classification and Regression Trees (CART)", "people": "Leo Breiman; Jerome Friedman; Richard Olshen; Charles Stone",
        "summary": "The CART monograph presents tree-structured prediction rules for classification and regression.",
        "significance": "Recursive partitions make nonlinear prediction available through interpretable split rules. Controlling tree complexity remains essential to generalization.",
        "connections": ["Inspect the Decision Tree recipe's nodes and rules alongside held-out scores and training-only validation."],
        "sources": [{"title": "Classification and Regression Trees", "url": "https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Olshen-Stone/p/book/9780412048418"}],
        "paper_ids": [],
    },
    {
        "id": "backpropagation", "year": 1986, "date": "1986", "era": "Deep learning",
        "title": "Backpropagation and learned representations", "people": "David Rumelhart; Geoffrey Hinton; Ronald Williams",
        "summary": "An influential backpropagation paper demonstrates how hidden units can learn useful representations from an error signal.",
        "significance": "The chain rule assigns derivatives to parameters across multiple layers. This paper helped establish the approach; differentiation and related algorithms have earlier histories.",
        "connections": ["Forward inference and backward gradient computation have different graphs, memory lifetimes and costs."],
        "sources": [{"title": "Learning Representations by Back-Propagating Errors", "url": "https://doi.org/10.1038/323533a0"}],
        "paper_ids": ["matrix-calculus"],
    },
    {
        "id": "q-learning", "year": 1992, "date": "1992", "era": "Statistical learning",
        "title": "Q-learning and action values", "people": "Christopher Watkins; Peter Dayan",
        "summary": "Q-learning's journal paper analyzes learning optimal action values under stated sampling and learning-rate conditions.",
        "significance": "An agent can learn from rewards and transitions rather than labeled input/output pairs. The convergence result's assumptions matter; it is not a guarantee for every neural reinforcement-learning system.",
        "connections": ["Keep supervised prediction, sequential decisions and their evaluation protocols distinct."],
        "sources": [{"title": "Q-learning", "url": "https://doi.org/10.1007/BF00992698"}],
        "paper_ids": [],
    },
    {
        "id": "svm", "year": 1995, "date": "1995", "era": "Statistical learning",
        "title": "Support-vector networks", "people": "Corinna Cortes; Vladimir Vapnik",
        "summary": "Support-vector networks develop classifiers that balance margin and training errors, with kernels supporting nonlinear boundaries.",
        "significance": "The method makes generalization and constrained optimization central to model design. Its useful representation can differ substantially from a neural feature hierarchy.",
        "connections": ["The SVM recipe exposes kernel and regularization settings; fit scaling on training data only."],
        "sources": [{"title": "Support-Vector Networks", "url": "https://doi.org/10.1007/BF00994018"}],
        "paper_ids": [],
    },
    {
        "id": "adaboost", "year": 1997, "date": "1995 / 1997", "era": "Statistical learning",
        "title": "AdaBoost and adaptive ensembles", "people": "Yoav Freund; Robert Schapire",
        "summary": "The boosting work combines weak learners while adapting the emphasis on training examples.",
        "significance": "The linked journal article appeared in 1997 after a 1995 conference abstract. Sequential reweighting supplies a different ensemble principle from bagging or random forests.",
        "connections": ["Compare ensemble mechanisms and validation scores, rather than treating every collection of trees as the same model."],
        "sources": [{"title": "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting", "url": "https://www.sciencedirect.com/science/article/pii/S002200009791504X"}],
        "paper_ids": [],
    },
    {
        "id": "deep-blue", "year": 1997, "date": "1997", "era": "Early AI",
        "title": "Deep Blue: chess search on specialized hardware", "people": "IBM Deep Blue team; Garry Kasparov",
        "summary": "Deep Blue wins a six-game match against the reigning world chess champion under the match's conditions.",
        "significance": "Domain evaluation, large-scale search and specialized hardware combine to solve a demanding narrow task. Chess strength does not establish a universal reasoning capability.",
        "connections": ["Hardware speedups must be interpreted with the workload and algorithm that use them."],
        "sources": [{"title": "IBM: Deep Blue", "url": "https://www.ibm.com/history/deep-blue"}],
        "paper_ids": [],
    },
    {
        "id": "lenet", "year": 1998, "date": "1998", "era": "Deep learning",
        "title": "LeNet and convolutional document recognition", "people": "Yann LeCun; Léon Bottou; Yoshua Bengio; Patrick Haffner",
        "summary": "The document-recognition paper describes convolutional networks and gradient-based learning within larger recognition systems.",
        "significance": "Local connectivity and shared weights match useful image structure and permit learned features. The 1998 paper consolidates work that developed over earlier years.",
        "connections": ["Weight reuse, activation shapes and memory traffic help explain why an operation graph matters to hardware."],
        "sources": [{"title": "Gradient-Based Learning Applied to Document Recognition", "url": "https://leon.bottou.org/papers/lecun-98h"}],
        "paper_ids": [],
    },
    {
        "id": "random-forests", "year": 2001, "date": "2001", "era": "Statistical learning",
        "title": "Random forests", "people": "Leo Breiman",
        "summary": "Random Forests studies collections of randomized tree predictors and their aggregate behavior.",
        "significance": "Combining diverse trees can improve prediction without discarding the statistical problem of generalization. Feature-importance values do not by themselves establish causality.",
        "connections": ["Use the Random Forest and Model Comparison recipes to compare a forest with a regularized linear baseline on shared folds."],
        "sources": [{"title": "Random Forests", "url": "https://doi.org/10.1023/A:1010933404324"}],
        "paper_ids": [],
    },
    {
        "id": "alexnet", "year": 2012, "date": "2012", "era": "Deep learning",
        "title": "AlexNet: ImageNet training on GPUs", "people": "Alex Krizhevsky; Ilya Sutskever; Geoffrey Hinton",
        "summary": "The ImageNet convolutional-network study combines GPU training, a large labeled dataset and a deep model.",
        "significance": "The result demonstrates the practical interaction of architecture, labeled data and GPU compute. The benchmark helped draw attention to learning representations with larger neural networks.",
        "connections": ["A model's score and its runtime are separate observations; measure the device, precision, batch and workload."],
        "sources": [{"title": "ImageNet Classification with Deep Convolutional Neural Networks", "url": "https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html"}],
        "paper_ids": [],
    },
    {
        "id": "neural-attention", "year": 2014, "date": "2014 preprint", "era": "Deep learning",
        "title": "Neural translation learns attention", "people": "Dzmitry Bahdanau; Kyunghyun Cho; Yoshua Bengio",
        "summary": "Neural translation learns soft alignments so a decoder can use relevant source representations at each step.",
        "significance": "The proposal reduces reliance on one fixed-length sentence representation. It was posted in 2014 and presented at ICLR 2015; attention also has other research lineages.",
        "connections": ["Compare attention's data dependencies with recurrence when reading a Transformer graph."],
        "sources": [{"title": "Neural Machine Translation by Jointly Learning to Align and Translate", "url": "https://arxiv.org/abs/1409.0473"}],
        "paper_ids": ["illustrated-transformer"],
    },
    {
        "id": "dqn", "year": 2015, "date": "2015", "era": "Deep learning",
        "title": "DQN: deep reinforcement learning from pixels", "people": "Volodymyr Mnih and DeepMind collaborators",
        "summary": "The DQN study learns action values from Atari game observations using a neural network, experience replay and target values.",
        "significance": "Representation learning and reinforcement learning can operate within one system. Reported performance belongs to the evaluated games and protocol, not every possible environment.",
        "connections": ["A training algorithm may include buffers, target models and environment interaction that are absent from its inference graph."],
        "sources": [{"title": "Human-Level Control through Deep Reinforcement Learning", "url": "https://doi.org/10.1038/nature14236"}],
        "paper_ids": [],
    },
    {
        "id": "alphago", "year": 2016, "date": "2016", "era": "Deep learning",
        "title": "AlphaGo: neural networks guide tree search", "people": "David Silver and DeepMind collaborators",
        "summary": "AlphaGo combines neural policy/value models with tree search; the 2016 Nature paper reports the system and its match against Fan Hui.",
        "significance": "Learned components can guide search through a huge decision space. The paper's reported match is distinct from the later 2016 Lee Sedol match.",
        "connections": ["System performance can depend on search-time compute in addition to the cost of one model forward pass."],
        "sources": [{"title": "Mastering the Game of Go with Deep Neural Networks and Tree Search", "url": "https://doi.org/10.1038/nature16961"}],
        "paper_ids": [],
    },
    {
        "id": "tpu", "year": 2017, "date": "2017 paper", "era": "Deep learning",
        "title": "The TPU: a domain-specific inference accelerator", "people": "Norman Jouppi and Google collaborators",
        "summary": "A production TPU study evaluates a custom neural-inference ASIC centered on matrix multiplication and software-managed on-chip memory.",
        "significance": "The paper connects workload characteristics with architecture, memory organization, latency and energy efficiency. Its 2017 publication analyzes hardware already deployed since 2015.",
        "connections": ["Relate matrix operations in a model graph to data reuse, on-chip capacity and memory traffic.", "Compare measured application performance with the arithmetic unit's peak throughput."],
        "sources": [{"title": "In-Datacenter Performance Analysis of a Tensor Processing Unit", "url": "https://arxiv.org/abs/1704.04760"}],
        "paper_ids": [],
    },
    {
        "id": "transformer", "year": 2017, "date": "2017", "era": "Language models",
        "title": "Attention Is All You Need: the Transformer", "people": "Ashish Vaswani and collaborators",
        "summary": "Attention Is All You Need introduces an attention-based encoder/decoder architecture without recurrent or convolutional sequence layers.",
        "significance": "The design permits more parallel sequence processing during training and emphasizes matrix products, attention and feed-forward blocks. Its original experiments concern translation, not every later language-model use.",
        "connections": ["In the Research lab, inspect attention projections, matrix shapes and residual dependencies.", "Prefill processes a prompt; autoregressive decode repeatedly extends it, changing the workload."],
        "sources": [{"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762"}],
        "paper_ids": ["transformer", "illustrated-transformer"],
    },
    {
        "id": "matrix-calculus", "year": 2018, "date": "2018", "era": "Deep learning",
        "title": "The Matrix Calculus You Need for Deep Learning", "people": "Terence Parr; Jeremy Howard",
        "summary": "The Matrix Calculus You Need For Deep Learning explains derivatives, Jacobians and chain-rule calculations for neural networks.",
        "significance": "This is an explanatory reading in the local collection, not a claim that matrix calculus originated in 2018. It connects scalar calculus to the tensor operations used by learning systems.",
        "connections": ["Use shape-aware derivatives to distinguish forward operations from the gradients required for training."],
        "sources": [{"title": "The Matrix Calculus You Need For Deep Learning", "url": "https://arxiv.org/abs/1802.01528"}],
        "paper_ids": ["matrix-calculus"],
    },
    {
        "id": "illustrated-transformer", "year": 2018, "date": "2018", "era": "Language models",
        "title": "The Illustrated Transformer", "people": "Jay Alammar",
        "summary": "The Illustrated Transformer walks through embeddings, attention heads, positional information and encoder/decoder blocks.",
        "significance": "Jay Alammar's explanatory diagrams give a visual reading companion to the architecture paper. The timeline uses the article's 2018 publication year; the local PDF is a later print export.",
        "connections": ["Relate a diagram's conceptual blocks to the smaller actual operations and aliases in a captured graph."],
        "sources": [{"title": "The Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/"}],
        "paper_ids": ["illustrated-transformer", "transformer"],
    },
    {
        "id": "bert", "year": 2018, "date": "2018 preprint", "era": "Language models",
        "title": "BERT: bidirectional language representations", "people": "Jacob Devlin; Ming-Wei Chang; Kenton Lee; Kristina Toutanova",
        "summary": "BERT uses a bidirectional Transformer encoder and pretraining objectives, followed by adaptation to downstream tasks.",
        "significance": "The work demonstrates a reusable pretrained representation rather than training every task from scratch. Its encoder and masked-language objective differ from autoregressive generation.",
        "connections": ["Architecture and training objective determine what a model's outputs mean; not every Transformer is a chat generator."],
        "sources": [{"title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "url": "https://arxiv.org/abs/1810.04805"}],
        "paper_ids": [],
    },
    {
        "id": "scaling-laws", "year": 2020, "date": "2020", "era": "Language models",
        "title": "Scaling laws for neural language models", "people": "Jared Kaplan and collaborators",
        "summary": "An empirical study relates language-model cross-entropy to parameter count, dataset size and training computation.",
        "significance": "Observed scaling relationships support resource-planning hypotheses within studied regimes. They are empirical findings, not laws guaranteeing every downstream capability.",
        "connections": ["Track parameter storage, data, training compute and inference cost as different quantities."],
        "sources": [{"title": "Scaling Laws for Neural Language Models", "url": "https://arxiv.org/abs/2001.08361"}],
        "paper_ids": ["gpt3"],
    },
    {
        "id": "gpt3", "year": 2020, "date": "2020", "era": "Language models",
        "title": "GPT-3 and few-shot prompting", "people": "Tom B. Brown and collaborators",
        "summary": "The GPT-3 paper studies zero-, one- and few-shot task performance using examples supplied in the context of a large autoregressive model.",
        "significance": "In-context learning changes how a pretrained model can be used. The paper also reports limitations; benchmark performance and generated fluency do not establish reliable understanding of every task.",
        "connections": ["The Local LLMs prompt is inference input, not fine-tuning.", "Longer contexts increase prefill work and key/value-cache demand."],
        "sources": [{"title": "Language Models are Few-Shot Learners", "url": "https://arxiv.org/abs/2005.14165"}],
        "paper_ids": ["gpt3"],
    },
    {
        "id": "switch-transformers", "year": 2021, "date": "2021 preprint", "era": "Language models",
        "title": "Switch Transformers and sparse experts", "people": "William Fedus; Barret Zoph; Noam Shazeer",
        "summary": "Switch Transformers simplify sparse mixture-of-experts routing, selecting part of the model for each incoming example.",
        "significance": "Total parameter count can grow without activating every parameter for every token. Routing, communication and storage remain costs; this is a landmark in an older mixture-of-experts lineage.",
        "connections": ["For a MoE model, distinguish resident weights, activated weights and actual measured memory traffic."],
        "sources": [{"title": "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", "url": "https://arxiv.org/abs/2101.03961"}],
        "paper_ids": ["qwen3"],
    },
    {
        "id": "instructgpt", "year": 2022, "date": "2022", "era": "Language models",
        "title": "InstructGPT: learning from demonstrations and feedback", "people": "Long Ouyang and collaborators",
        "summary": "InstructGPT combines supervised demonstrations with rankings of generated answers and reinforcement learning from human feedback.",
        "significance": "The study treats following a user's instructions as a training objective and evaluates answers through human preferences. The authors also report continuing errors in the resulting models.",
        "connections": ["A base language model and an instruction-tuned checkpoint can behave differently with the same prompt.", "Keep the training objective, chat formatting and inference settings explicit when comparing local models."],
        "sources": [{"title": "Training language models to follow instructions with human feedback", "url": "https://arxiv.org/abs/2203.02155"}],
        "paper_ids": [],
    },
    {
        "id": "chinchilla", "year": 2022, "date": "2022", "era": "Language models",
        "title": "Chinchilla and compute-optimal training", "people": "Jordan Hoffmann and collaborators",
        "summary": "The Chinchilla study revisits how to allocate fixed training compute between parameter count and training data.",
        "significance": "The study argues that many models in its comparison were undertrained for their size. Compute-optimal training and economical deployment are related but different optimization problems.",
        "connections": ["A smaller, longer-trained model can have different deployment costs from a larger model trained under another budget."],
        "sources": [{"title": "Training Compute-Optimal Large Language Models", "url": "https://arxiv.org/abs/2203.15556"}],
        "paper_ids": [],
    },
    {
        "id": "flashattention", "year": 2022, "date": "2022", "era": "Language models",
        "title": "FlashAttention: attention and memory traffic", "people": "Tri Dao; Daniel Fu; Stefano Ermon; Atri Rudra; Christopher Ré",
        "summary": "FlashAttention uses tiling and awareness of GPU memory levels to compute exact attention with less high-bandwidth-memory traffic.",
        "significance": "Equivalent mathematical operations can have very different execution costs because of memory movement and kernel organization. Logical graph capture alone does not prove which device kernel ran.",
        "connections": ["Relate tensor sizes and reuse to memory bandwidth.", "Keep the app's captured CPU logical graphs separate from future device/kernel profiling."],
        "sources": [{"title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "url": "https://arxiv.org/abs/2205.14135"}],
        "paper_ids": ["gqa"],
    },
    {
        "id": "llama", "year": 2023, "date": "2023", "era": "Language models",
        "title": "LLaMA: efficient foundation language models", "people": "Hugo Touvron and Meta AI collaborators",
        "summary": "The LLaMA paper presents a family of language models trained with publicly available datasets and evaluates the resulting models.",
        "significance": "Accessible model research and subsequent inference tooling support local experimentation. The paper title's use of open does not mean every model or derivative has the same license.",
        "connections": ["Local LLMs can compare selected model files and runtimes while keeping model identity and execution settings explicit."],
        "sources": [{"title": "LLaMA: Open and Efficient Foundation Language Models", "url": "https://arxiv.org/abs/2302.13971"}],
        "paper_ids": [],
    },
    {
        "id": "gqa", "year": 2023, "date": "2023", "era": "Language models",
        "title": "GQA: sharing key/value attention heads", "people": "Joshua Ainslie and collaborators",
        "summary": "Grouped-query attention uses fewer key/value heads than query heads, interpolating between multi-head and multi-query attention.",
        "significance": "The paper studies converting existing checkpoints and balancing inference speed with quality. Sharing key/value heads can reduce cache storage; actual speed depends on implementation and workload.",
        "connections": ["Inspect query-head versus key/value-head dimensions.", "Measure decode latency and cache-related memory costs separately from prompt processing."],
        "sources": [{"title": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "url": "https://arxiv.org/abs/2305.13245"}],
        "paper_ids": ["gqa"],
    },
    {
        "id": "qwen3", "year": 2025, "date": "2025", "era": "Language models",
        "title": "Qwen3: dense models, experts and reasoning modes", "people": "Qwen team; An Yang and collaborators",
        "summary": "The Qwen3 Technical Report describes dense and mixture-of-experts model families, training procedures and thinking/non-thinking modes.",
        "significance": "This report is the endpoint selected from the local collection. It brings architecture, training and inference behavior together in models suitable for further local study; a converted GGUF still needs its own identity and runtime provenance.",
        "connections": ["Compare total and active parameter counts for MoE models.", "Record the model file, runtime, prompt and settings when interpreting a graph capture or performance trial."],
        "sources": [{"title": "Qwen3 Technical Report", "url": "https://arxiv.org/abs/2505.09388"}],
        "paper_ids": ["qwen3"],
    },
]


def timeline_entries() -> list[dict]:
    """Return independent chronological records for the selected history."""
    return deepcopy(_ENTRIES)


def paper_specs() -> list[dict]:
    """Return the six known readings, without opening or searching for files."""
    return deepcopy(_PAPERS)


def _library_directory(directory: str | Path | None) -> Path:
    return (Path.home() / "learning" / "books" / "papers" / "ai-ML" if directory is None else Path(directory)).expanduser().resolve()


def resolve_library(directory: str | Path | None = None) -> list[dict]:
    """Describe only known PDFs in a folder; never read their content or recurse.

    Missing readings remain visible. A file symlink escaping the chosen folder
    is not treated as a linked reading. Resolving the chosen directory itself
    permits a user to explicitly select a directory reached through a symlink.
    """
    root = _library_directory(directory)
    resolved = []
    for paper in paper_specs():
        candidate = root / paper["filename"]
        try:
            actual = candidate.resolve()
            available = actual.is_relative_to(root) and actual.is_file()
        except (OSError, RuntimeError):
            available = False
        resolved.append({**paper, "path": str(candidate), "available": available})
    return resolved


def resolve_paper(paper_id: str, directory: str | Path | None = None) -> Path:
    """Resolve one catalog identity for an explicit user open action."""
    if not isinstance(paper_id, str) or not any(paper["id"] == paper_id for paper in _PAPERS):
        raise ValueError("Choose a reading from the research catalog.")
    paper = next(paper for paper in resolve_library(directory) if paper["id"] == paper_id)
    if not paper["available"]:
        raise FileNotFoundError("The linked PDF is unavailable in the selected papers folder: " + paper["filename"])
    return Path(paper["path"]).resolve()
