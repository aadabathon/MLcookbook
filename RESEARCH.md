# Research timeline

Open **Research** in the sidebar to trace selected foundations and milestones in artificial intelligence. The chronology connects formal reasoning, computation, statistical learning, neural networks and language models to the recipes and Local LLM research tools in this workbench.

The timeline is curated history, with earlier precursors distinguished from AI as an organized research field. Approximate ancient dates remain approximate. The Dartmouth proposal was written in 1955 for the summer 1956 research project. A paper's original publication year identifies its milestone; a later PDF revision or print-export timestamp does not move that milestone forward.

## Browsing

- Search across ideas, people, readings and connections. Search terms combine with the era selector.
- Select a dated milestone, or use the arrow keys, to read its context, significance and sources.
- **Linked readings** narrows the chronology to entries connected to the six readings in the initial catalog. It includes a reading even when its local PDF is unavailable.
- **Open source** opens a cited reference in the default browser. **Open PDF** opens the selected catalog file in the default PDF application.
- **Papers folder…** changes the local reading directory. File availability is checked again when Research opens. A missing PDF leaves the timeline and online references available.

The default directory is `~/learning/books/papers/ai-ML`. The initial catalog links Adam's copies of these works:

| Reading | Original publication | Reference |
| --- | --- | --- |
| Attention Is All You Need | 2017 | [Vaswani and colleagues](https://arxiv.org/abs/1706.03762) |
| The Illustrated Transformer | 2018 | [Jay Alammar's explanation](https://jalammar.github.io/illustrated-transformer/) |
| The Matrix Calculus You Need for Deep Learning | 2018 | [Terence Parr and Jeremy Howard](https://arxiv.org/abs/1802.01528) |
| Language Models are Few-Shot Learners | 2020 | [Brown and colleagues](https://arxiv.org/abs/2005.14165) |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | 2023 | [Ainslie and colleagues](https://arxiv.org/abs/2305.13245) |
| Qwen3 Technical Report | 2025 | [Qwen team](https://arxiv.org/abs/2505.09388) |

The Illustrated Transformer is an explanatory article, presented separately from the original Transformer paper. Qwen3's report was submitted to arXiv on May 14, 2025; the local report's front page is dated May 15. The curated chronology ends with that report rather than claiming to cover every subsequent development.

## Local behavior and implementation

Browsing works offline. Opening Research reads the bundled catalog and checks only its known PDF filenames in the selected directory. It does not parse or copy the PDFs, fetch references, load inference weights, or run an analysis. Reading actions happen explicitly. The six source PDFs remain in the user's library and are excluded from packaging.

Search, era, selected milestone and timeline scroll survive navigation in memory. Research has independent state; active data, recipe reports/examples, Kitchen documents and Movement Study holdouts retain their existing behavior and save formats.

The submodule-only catalog and local-file resolver live in `src/mlcookbook/research.py`. `gui/screens/research.py` renders plain entries/library descriptors and emits actions. `app.py` owns navigation, the selected directory, catalog-checked source links and explicit file opening. The page adds no dependencies and can open before the scientific or inference runtime is initialized.

Tests in `test_research.py`, `test_research_screen.py` and `test_research_app.py` cover chronology/catalog integrity, fixed-file resolution, filtering, source actions, unavailable readings, navigation restoration, analytical-state boundaries, viewport containment and offline import behavior.
