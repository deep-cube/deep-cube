#!/bin/bash
if [[ -z "$1" ]] || [[ -z "$2" ]]
then
    echo " Usage: sh loop_compress_small.sh input_folder_path output_folder_path \n Note: Recommend using absolute path!"
else
    for filepath in $1/*; do
      filename=${filepath##*/}
      if [[ ${filename: -4} == ".MP4" ]] || [[ ${filename: -4} == ".MOV" ]] || [[ ${filename: -4} == ".mp4" ]] || [[ ${filename: -4} == ".mov" ]]
      then
          inputpath="${1%/}/${filename}"
          outputpath="${2%/}/${filename%.*}.mp4"
          echo "\n----------------------------- Processing: ${filename} ------------------------------"
          ffmpeg -i ${inputpath} -filter:v "scale=90:68,fps=fps=30" -vcodec libx264 -crf 20 -an ${outputpath}
          # echo "$inputpath and $outputpath"
      fi
    done
fi

echo "Done."
