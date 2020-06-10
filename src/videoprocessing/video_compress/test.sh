#!/bin/bash
if [[ -z "$1" ]] || [[ -z "$2" ]]
then
    echo "Usage: sh test.sh input_folder_path output_folder_path"
else
    for file in $1/*; do
      var1=${file##*/}
      if [[ ${var1: -3} == ".py"  ]]
      then
          echo ${var1}
      fi
    done
fi