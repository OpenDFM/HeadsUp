XNLI consists of 10k English sentences translated into 14 languages: 

ar: Arabic
bg: Bulgarian
de: German
el: Greek
es: Spanish
fr: French
hi: Hindi
ru: Russian
sw: Swahili
th: Thai
tr: Turkish
ur: Urdu
vi: Vietnamese
zh: Chinese (Simplified)

The XNLI 15-way parallel corpus can be used for Machine Translation as evaluation sets, in particular for low-resource languages such as Swahili or Urdu.

We provide two files: xnli.15way.orig.tsv and xnli.15way.tok.tsv containing respectively the original and the tokenized version of the corpus.
The files consist of 15 tab-separated columns, each corresponding to one language as indicated by the header.

Please consider citing the following paper if using this dataset:

@InProceedings{conneau2018xnli,
  author =  "Conneau, Alexis
         and Rinott, Ruty
         and Lample, Guillaume
         and Williams, Adina
         and Bowman, Samuel R.
         and Schwenk, Holger
         and Stoyanov, Veselin",
  title =   "XNLI: Evaluating Cross-lingual Sentence Representations",
  booktitle =   "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year =    "2018",
  publisher =   "Association for Computational Linguistics",
  location =    "Brussels, Belgium",
}
