# FYP_ARSynopsis
A specialized system for summarizing annual reports, incorporating sentence clustering, sentence selection, and abstractive summarization techniques to generate concise and informative summaries.


# Current Status

- Positional encoded embeddings improved the segmentation
- K-Means clustering- need to control number of clusters
- Removed Hyperparameter tuning process - To minimize time for segmentation & control hyperparameters according to the task
- Extractive summarization -> Text Rank Algo with BERT embeddings
- Compared summarization results of several transformer models
    - Decission: BART - large
    - Reasons: LongT5, PEGASUS X gives large context window, but poor summary quality: repetition of same sentences

# On progress
- Application development
- Segmentation process evaluation
- Model Finetuning

# Next Steps
- Advanced cleaning techniques
    - Removing duplicates - need to check for duplicate sentence removal in preprocessing phase or in extractive summarization process
    - Filtering after clustering - need to develop and algorithm
- PacSum Extractive Summarization
- Evaluating extractive summarization quality
    - TextRank vs PacSum vs MMRG
- UI
