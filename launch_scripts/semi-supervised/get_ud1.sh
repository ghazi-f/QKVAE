#!/usr/bin/env bash
wget "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1548/ud-treebanks-v1.2.tgz"
mkdir .data
mkdir .data/udpos
mkdir .data/udpos/en-ud-v1
tar -vxf ud-treebanks-v1.2.tgz universal-dependencies-1.2/UD_English/en-ud-dev.conllu
tar -vxf ud-treebanks-v1.2.tgz universal-dependencies-1.2/UD_English/en-ud-train.conllu
tar -vxf ud-treebanks-v1.2.tgz universal-dependencies-1.2/UD_English/en-ud-test.conllu
mv universal-dependencies-1.2/UD_English/* .data/udpos/en-ud-v1/
rm -r universal-dependencies-1.2
rm ud-treebanks-v1.2.tgz
