# PI - Assignment 12

## Toxic comment classification

Check out the space at [https://huggingface.co/spaces/jasurbek-fm/toxic-comment](https://huggingface.co/spaces/jasurbek-fm/toxic-comment)

## Group Members
- Ergashev Jasurbek
- Mukhtor Eshimov
- Munira Rakhmatova

## **Problem and Solution:**
The increase in toxic behavior on social media is a major concern, negatively impacting online communities. To address this, we fine-tuned a machine learning model, DistilBERT, to classify text as either toxic or non-toxic.

## **Dataset Summary:**
We used the ToxicDataset from Kaggle, containing 837 toxic comments and 6,654 non-toxic comments. To ensure better balance and diversity, we added over 15,000 toxic comments from an external dataset, resulting in a comprehensive dataset for training and evaluation.

## **Methodology:**
- Data Preprocessing: Text data was cleaned and tokenized, with appropriate padding and truncation for DistilBERT input.
- Model Selection: DistilBERT was chosen due to its efficiency and ability to handle nuanced natural language processing tasks.
- Training and Validation: The dataset was split into training (80%) and validation (20%) sets. We fine-tuned DistilBERT using pre-trained weights, leveraging transfer learning for optimal performance.
- Evaluation Metrics: The model was evaluated using accuracy, F1-score, precision, and recall.

## **Results:**
The fine-tuned DistilBERT model achieved an accuracy of 93% and an F1-score of 0.91 on the test set, demonstrating its strong performance in detecting toxic content.

## **Challenges and Solutions:**
- Data Imbalance: The dataset initially had significantly fewer toxic comments compared to non-toxic ones. Adding 15,000 toxic samples from an external source addressed this imbalance.
- Resource Constraints: Fine-tuning DistilBERT required substantial computational resources, which we managed using cloud-based GPUs.
- Ambiguity in Language: Some comments were contextually ambiguous. The use of DistilBERT's contextual embeddings improved the model's ability to discern these subtleties.


This model demonstrates an effective solution for moderating toxic behavior on social media, offering a scalable tool for improving online interactions.
