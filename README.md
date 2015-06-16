TGen
====

*Natural language generator for spoken dialogue systems with a statistical sentence planner*

TGen is a natural language generator, composed of:
- a statistical sentence planner based on A*-style search and a perceptron ranker
- a rule-based surface realizer using the existing [Treex NLP toolkit](http://ufal.cz/treex)

Notice
------

TGen is currently highly experimental and not very well tested. Use at your own risk.

To get the version used in our ACL 2015 paper _Training a Natural Language Generator From Unaligned Data_, see [this release](https://github.com/UFAL-DSG/tgen/releases/tag/ACL2015).

Dependencies
------------

For TGen to work properly, you need to have these modules installed:

- [Alex](https://github.com/UFAL-DSG/alex)
- [Flect](https://github.com/UFAL-DSG/flect)
- [Treex](http://ufal.cz/treex)

The first two ones can be avoided by just copying a few libraries; these will be integrated here in the future.

License
-------
Author: [Ondřej Dušek](http://ufal.cz/ondrej-dusek)

Copyright © 2014-2015 Institute of Formal and Applied Linguistics, Charles University in Prague.

Licensed under the Apache License, Version 2.0.

Acknowledgements
----------------

Work on this project was funded by the Ministry of Education, Youth and Sports of the Czech Republic under the grant agreement LK11221 and core research funding, SVV project 260 104, and GAUK grant 2058214 of Charles University in Prague. It used language resources stored and distributed by the LINDAT/CLARIN project of the Ministry of Education, Youth and Sports of the Czech Republic (project LM2010013).
