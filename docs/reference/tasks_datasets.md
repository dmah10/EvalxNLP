The framework currently supports the following text classification tasks:

## 🧠 Supported Tasks

#### 💬 Sentiment Analysis
- Determines the sentiment expressed in a text, such as **positive**, **negative**, or **neutral**.
- Commonly used in applications like product reviews, social media analysis, and customer feedback.

#### 🚫 Hate Speech Detection
- Identifies and classifies text containing **hate speech**, **offensive language**, or **harmful content**.
- Essential for moderating online platforms and ensuring safe digital environments.

#### 🔗 Natural Language Inference (NLI)
- Determines the **logical relationship** between two sentences (e.g., premise and hypothesis).
- Tasks include identifying **entailment**, **contradiction**, or **neutrality** between sentences.
- Useful for applications like question answering, summarization, and reasoning tasks.

## 📚 Dataset Support

EvalxNLP includes a representative dataset for each of the supported tasks and allows users to extend the framework with additional classification datasets. All datasets are **rationale-annotated**, meaning they include human-annotated rationales that highlight the most important words or sentences for a given class label. These rationales enable the evaluation of alignment between model explanations and human understanding.


#### 🎬 MovieReviews

  *Task:* Sentiment Analysis  
  *Description:* Contains 1,000 positive and 1,000 negative movie reviews. Each review includes **phrase-level human-annotated rationales** that justify the sentiment label.
#### 📢 HateXplain 
  *Task:* Hate Speech Detection  
  *Description:* Comprises 20,000 posts from Gab and Twitter, annotated with one of three labels: **hate speech**, **offensive**, or **normal**.

#### 📄 e-SNLI
  *Task:* Natural Language Inference  
  *Description:* Contains 549,367 examples split into training, validation, and test sets. Each example includes a **premise** and a **hypothesis**, annotated with one of three labels: **entailment**, **contradiction**, or **neutral**.

---
