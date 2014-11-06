#!/bin/bash

base=`echo $1 | sed -E 's/\.yaml(\.gz)?$//'`
if [ -n "$2" ]; then
    sent_no=##$2
fi
treex Read::YAML from=$1 Write::Treex to=$base.treex.gz
ttred $base.treex.gz$sent_no
