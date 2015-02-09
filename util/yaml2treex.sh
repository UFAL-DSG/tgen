#!/bin/bash

base=`echo $1 | sed -E 's/\.yaml(\.gz)?$//'`
treex Read::YAML from=$1 Write::Treex to=$base.treex.gz
