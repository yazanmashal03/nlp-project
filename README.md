# LLM Toxic Content Analysis

## Introduction
In recent years, the transformer-based architecture [Vaswani et al., 2023] has significantly advanced the field of natural language processing (NLP), revolutionizing text processing and text generation. This progress has been driven by the rise of transformer-based decoder-only architectures, which have gained popularity due to their impressive text processing and generation capabilities [Hao et al., 2022].

Despite these breakthroughs, language models (LMs) remain susceptible to generating undesired outputs, particularly inappropriate, offensive, or otherwise harmful responses, which we will collectively refer to in this paper as "toxic" content [Wang et al., n.d.]. Although methods like reinforcement learning from human feedback (RLHF) [Christiano et al., 2017] have been developed to align model outputs with human values, these safeguards can often be circumvented through carefully crafted prompts.

To explore this gap, this paper (https://drive.google.com/file/d/1aD4txp7KNLjEfj0frm9kXpeo4sIBfiJt/view) examines the extent to which LLMs generate toxic content when prompted, as well as the linguistic factors—both lexical and syntactic—that influence the production of such outputs in generative models.

## Research Questions
The research questions that will guide this paper are:

- **RQ1:** How prone are generative large language models to generate toxic outputs when prompted to?
- **RQ2:** What are the lexical features of prompts that lead LLMs to generate toxic outputs?
- **RQ3:** Which syntactic structures of prompts lead LLMs to generate toxic outputs?

---

### References
- Vaswani, A., et al. (2023). Attention Is All You Need. 
- Hao, H., et al. (2022). Language Modeling Advances in Transformers. 
- Wang, W., et al. (n.d.). Decoding Trust in Language Models. 
- Christiano, P., et al. (2017). Deep Reinforcement Learning from Human Feedback.
