python3 scripts/bert_coref.py -m predict -w models/crossval/0.model -v data/litbank_tenfold_splits/0/test.conll -o preds/crossval/0.goldmentions.test.preds -s reference-coreference-scorers/scorer.pl
python3 scripts/bert_coref.py -m predict -w models/crossval/1.model -v data/litbank_tenfold_splits/1/test.conll -o preds/crossval/1.goldmentions.test.preds -s reference-coreference-scorers/scorer.pl
python3 scripts/bert_coref.py -m predict -w models/crossval/2.model -v data/litbank_tenfold_splits/2/test.conll -o preds/crossval/2.goldmentions.test.preds -s reference-coreference-scorers/scorer.pl
python3 scripts/bert_coref.py -m predict -w models/crossval/3.model -v data/litbank_tenfold_splits/3/test.conll -o preds/crossval/3.goldmentions.test.preds -s reference-coreference-scorers/scorer.pl
python3 scripts/bert_coref.py -m predict -w models/crossval/4.model -v data/litbank_tenfold_splits/4/test.conll -o preds/crossval/4.goldmentions.test.preds -s reference-coreference-scorers/scorer.pl
python3 scripts/bert_coref.py -m predict -w models/crossval/5.model -v data/litbank_tenfold_splits/5/test.conll -o preds/crossval/5.goldmentions.test.preds -s reference-coreference-scorers/scorer.pl
python3 scripts/bert_coref.py -m predict -w models/crossval/6.model -v data/litbank_tenfold_splits/6/test.conll -o preds/crossval/6.goldmentions.test.preds -s reference-coreference-scorers/scorer.pl
python3 scripts/bert_coref.py -m predict -w models/crossval/7.model -v data/litbank_tenfold_splits/7/test.conll -o preds/crossval/7.goldmentions.test.preds -s reference-coreference-scorers/scorer.pl
python3 scripts/bert_coref.py -m predict -w models/crossval/8.model -v data/litbank_tenfold_splits/8/test.conll -o preds/crossval/8.goldmentions.test.preds -s reference-coreference-scorers/scorer.pl
python3 scripts/bert_coref.py -m predict -w models/crossval/9.model -v data/litbank_tenfold_splits/9/test.conll -o preds/crossval/9.goldmentions.test.preds -s reference-coreference-scorers/scorer.pl
