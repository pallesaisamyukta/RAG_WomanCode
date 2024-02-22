from nltk.translate.bleu_score import sentence_bleu
import nltk


reference = 'For two consecutive days, eat wheat-based products with three meals, such as cereal for breakfast, a sandwich for lunch, and pasta for dinner. • For two consecutive days, remove all \
gluten-containing products from your diet. Read labels to make sure everything you eat is gluten-free. • The next day, eat a gluten-free diet until dinner. Then, at dinner, have a substantial \
amount of a glutencontaining food, such as pizza or pasta. Twenty minutes after eating, tune in to any symptoms. Do you have any pain or belching? Is there any swelling in your lower abdomen? \
Do you have any nasal congestion or a runny nose? Do you have a headache? Is there a rash or hives on your skin? Evaluate your symptoms two hours later, again looking for any bloating, gassiness, \
or flatulence. • Finally, the next morning, notice whether your bowels were affected. Are you constipated?'
# Example reference, RAG output, and No RAG output
reference = [['this', 'is', 'a', 'reference', 'sentence']]
rag_output = ['this', 'is', 'an', 'output', 'from', 'rag']
no_rag_output = ['this', 'is', 'an', 'output', 'from', 'no', 'rag']

# Compute BLEU score for RAG output
bleu_score_rag = sentence_bleu(reference, rag_output)

# Compute BLEU score for No RAG output
bleu_score_no_rag = sentence_bleu(reference, no_rag_output)

# Print the BLEU scores
print("BLEU Score (RAG):", bleu_score_rag)
print("BLEU Score (No RAG):", bleu_score_no_rag)