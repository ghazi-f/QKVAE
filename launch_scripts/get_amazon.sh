#!/usr/bin/env bash
wget "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Industrial_and_Scientific_5.json.gz"
wget "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Software_5.json.gz"
wget "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Luxury_Beauty_5.json.gz"
mkdir .data
mkdir .data/amazon
gzip -d Industrial_and_Scientific_5.json.gz
gzip -d Software_5.json.gz
gzip -d Luxury_Beauty_5.json.gz
mv Industrial_and_Scientific_5.json .data/amazon/
mv Software_5.json .data/amazon/
mv Luxury_Beauty_5.json .data/amazon/
