#!/bin/bash

# declare data path
ACE_path="/home/bensonliu/work/LDC2006T06/data/English"
Miwa_path="/home/bensonliu/work/miwa2016/corpus"
dest_path="/home/bensonliu/work/LDC2006T06/dataset"
ACE_set="timex2norm"

# create directory if don't have
if [ ! -d "$dest_path" ]; then
    mkdir -p "$dest_path"
fi
if [ ! -d "$dest_path/test" ]; then
    mkdir -p "$dest_path/test"
else 
    rm "$dest_path/test/"*
fi
if [ ! -d "$dest_path/train" ]; then
    mkdir -p "$dest_path/train"
else 
    rm "$dest_path/train/"*
fi
if [ ! -d "$dest_path/dev" ]; then
    mkdir -p "$dest_path/dev"
else 
    rm "$dest_path/dev/"*
fi
if [ ! -d "$dest_path/train+dev" ]; then
    mkdir -p "$dest_path/train+dev"
else 
    rm "$dest_path/train+dev/"*
fi

# get clssification of miwa dataset
test_set=`ls "$Miwa_path/test" | awk 'match($0, /.split.ann/){print($0)}' | sed 's/.split.ann//g'`
train_set=`ls "$Miwa_path/train" | awk 'match($0, /.split.ann/){print($0)}' | sed 's/.split.ann//g'`
dev_set=`ls "$Miwa_path/dev" | awk 'match($0, /.split.ann/){print($0)}' | sed 's/.split.ann//g'`

# clssify ACE data into sets
for src in `ls "$ACE_path"`; do
    targ_set=`ls "$ACE_path/$src/$ACE_set" | awk 'match($0, /.tab/){print($0)}' | sed 's/.tab//g'`
    for name in $targ_set; do
        sig=0
        file_name="$ACE_path/$src/$ACE_set/$name"
        if [[ $test_set =~ $name ]]; then
            cp "$file_name"* "$dest_path/test"
            sig=1
        fi
        if [[ $train_set =~ $name ]]; then
            cp "$file_name"* "$dest_path/train"
            cp "$file_name"* "$dest_path/train+dev"
            sig=1
        fi
        if [[ $dev_set =~ $name ]]; then
            cp "$file_name"* "$dest_path/dev"
            cp "$file_name"* "$dest_path/train+dev"
            sig=1
        fi
        if [ $sig -eq 0 ]; then
            cp "$file_name"* "$dest_path"
        fi
    done
done
