TGen
====

*A statistical natural language generator for spoken dialogue systems*

TGen is a statistical natural language generator, with two different algorithms supported:

1. A statistical sentence planner based on A*-style search, with a candidate plan generator and a perceptron ranker
2. A sequence-to-sequence (seq2seq) recurrent neural network architecture based on the [TensorFlow](https://www.tensorflow.org/) toolkit

Both algoritms can be trained from pairs of source meaning representations (dialogue acts) and target sentences.
The newer seq2seq approach is preferrable: it yields higher performance in terms of both speed and quality.

Both algorithms support generating sentence plans (deep syntax trees), which are subsequently converted to text using the existing the surface realizer from [Treex NLP toolkit](http://ufal.cz/treex).
The seq2seq algorithm also supports direct string generation.

For more details on the algorithms, please refer to our papers:
* For A*-search based generation, see our [ACL 2015 paper](http://www.aclweb.org/anthology/P/P15/P15-1044.pdf).
* For seq2seq generation, see our [ACL 2016 paper](http://aclweb.org/anthology/P16-2008).
* For an improved version of the seq2seq generation that takes previous user utterance into account to generate a more contextually-appropriate response, see our [SIGDIAL 2016 paper](http://www.sigdial.org/workshops/conference17/proceedings/pdf/SIGDIAL22.pdf).

Installation and Usage
----------------------

Please refer to [USAGE.md](USAGE.md) for instructions on how to use TGen.

Notice
------

* TGen is highly experimental and only tested on a few datasets, so bugs are inevitable. If you find a bug, feel free to [contact me](https://github.com/tuetschek) or [open an issue](https://github.com/UFAL-DSG/tgen/issues).
* If you do not require a specific version of TGen, we recommended to install the current master version, which has the latest bugfixes and all the functionality of the ACL2016/SIGDIAL2016 version.
  * To get the version used in our ACL 2015 paper (A*-search only), see [this release](https://github.com/UFAL-DSG/tgen/releases/tag/ACL2015).
  * To get the version used in our ACL 2016 and SIGDIAL 2016 papers (seq2seq approach for generating sentence plans or strings, optionally using previous context), see [this release](https://github.com/UFAL-DSG/tgen/releases/tag/ACL2016).

Citing TGen
-----------

If you use or refer to the seq2seq generation in TGen, please cite [this paper](http://aclweb.org/anthology/P16-2008):

* Ondřej Dušek and Filip Jurčíček (2016): Sequence-to-Sequence Generation for Spoken Dialogue via Deep Syntax Trees and Strings. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics_, Berlin, Germany.

If you use or refer to the context-aware improved seq2seq generation, please cite [this paper](http://www.sigdial.org/workshops/conference17/proceedings/pdf/SIGDIAL22.pdf):

* Ondřej Dušek and Filip Jurčíček (2016): A Context-aware Natural Language Generator for Dialogue Systems. In _Proceedings of the 17th Annual Meeting of the Special Interest Group on Discourse and Dialogue_, Los Angeles, CA, USA.

If you use or refer to the A*-search generation in TGen, please cite [this paper](http://www.aclweb.org/anthology/P/P15/P15-1044.pdf):

* Ondřej Dušek and Filip Jurčíček (2015): Training a Natural Language Generator From Unaligned Data. In _Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing_, pages 451–461, Beijing, China.

License
-------
Author: [Ondřej Dušek](http://ufal.cz/ondrej-dusek)

Copyright © 2014-2017 Institute of Formal and Applied Linguistics, Charles University, Prague.

Licensed under the Apache License, Version 2.0 (see [LICENSE.txt](LICENSE.txt)).

Acknowledgements
----------------

Work on this project was funded by the Ministry of Education, Youth and Sports of the Czech Republic under the grant agreement LK11221 and core research funding, SVV projects 260 104 and 260 333, and GAUK grant 2058214 of Charles University in Prague. It used language resources stored and distributed by the LINDAT/CLARIN project of the Ministry of Education, Youth and Sports of the Czech Republic (projects LM201001 and LM2015071).
