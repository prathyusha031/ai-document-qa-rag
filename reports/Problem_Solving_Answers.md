# Part 3 — Problem Solving Answers

---

## Q1. What is the difference between Machine Learning, Deep Learning, and Generative AI?

**Machine Learning (ML)** is a branch of artificial intelligence in which a
computer learns patterns from data instead of following explicitly programmed
rules. It builds a mathematical model from examples and uses it to make
predictions, for example classifying emails as spam or predicting house prices.

**Deep Learning (DL)** is a subfield of ML that uses artificial neural
networks with many layers ("deep" networks) to learn representations
automatically. It needs more data and computing power, but it excels at
complex tasks such as image recognition, speech recognition, and language
understanding. DL is what powers modern large language models (LLMs).

**Generative AI (GenAI)** is a category of AI that *creates new content* —
text, images, audio, or code — rather than only classifying or predicting.
It is usually built on deep learning (e.g. transformers). Examples include
ChatGPT, Gemini, and DALL·E.

**Relationship:** Generative AI ⊂ Deep Learning ⊂ Machine Learning ⊂ AI.
Not all ML is deep learning (e.g. linear regression is ML but not DL), and
not all DL is generative (e.g. a face-recognition network only classifies).

**Comparison:** ML *predicts*, DL *learns from raw data*, GenAI *generates
new outputs*. This project's RAG system uses embeddings (ML/DL) and a
generative LLM together.

---

## Q2. How would you reduce hallucinations in an AI chatbot?

**1. Ground the model with retrieval (RAG).** Instead of answering from the
model's memorised knowledge, retrieve relevant passages from a trusted
document store and force the model to answer using only that context. This
is exactly what this project does.

**2. Strong prompting.** Tell the model to answer only from the given
context and to say "I couldn't find this information" when the context is
insufficient — never to invent facts.

**3. Improve retrieval quality.** Use good chunking, better embeddings, and
a suitable number of retrieved chunks (K) so the model actually receives the
information it needs.

**4. Show sources.** Attach page numbers/chunk references to every answer so
users (and evaluators) can verify the answer against the document.

**5. Model configuration.** Use a low temperature (e.g. 0.2) so the output
is more deterministic and factual, and limit output length.

**6. Validation and review.** Add checks for answer-context consistency and,
for high-stakes domains (medicine, law, finance), keep a human in the loop.

RAG does not remove hallucinations completely, but it reduces them
significantly and makes them easy to detect.

---

## Q3. If an AI model gives biased predictions, how would you detect and reduce the bias?

**Detection:**
- **Dataset analysis:** inspect the training data for under-represented
  groups (e.g. gender, ethnicity, age) and for skewed labels.
- **Group comparison:** measure the model's accuracy, precision, recall, and
  error rates separately for each group and look for large gaps.
- **Fairness metrics:** use metrics such as equalised odds, demographic
  parity, and calibration across groups.
- **Bias testing:** probe the model with counterfactual inputs (same query,
  different protected attribute) and compare outputs.
- **Model evaluation:** run a dedicated test set designed to expose bias,
  not just the original benchmark.

**Reduction:**
- **Data balancing:** re-sample, re-weight, or augment data so groups are
  fairly represented; remove or relabel biased training examples.
- **Algorithmic fixes:** use fairness constraints during training or
  post-processing adjustments of predictions.
- **Monitoring:** after deployment, continuously track predictions across
  groups and re-train when drift or bias appears.
- **Human oversight:** have diverse teams review outputs and decisions,
  especially in sensitive areas like hiring or lending.

The goal is not to remove all differences (groups may legitimately differ)
but to remove *unfair* differences that are not justified by the task.
