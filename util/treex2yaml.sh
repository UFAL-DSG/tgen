#!/bin/bash

base=`echo $1 | sed -E 's/\.treex(\.gz)?$//'`
treex Read::Treex from=$1 Write::YAML to=$base.yaml.gz
