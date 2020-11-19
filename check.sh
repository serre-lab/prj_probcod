#!/bin/bash


files=$(ls logs/IVAE_eval*.err)

for f in ${files[@]}; do
v=$f
file_content=$(cat $f)
# echo $file_content
echo "$v ${#file_content}"
done