ffmpeg -i /Volumes/Untitled/DCIM/100MEDIA/DJI_0085.MP4 -filter:v "crop=iw-480:ih,scale=360:-1" -vcodec libx264 -crf 20 -an ~/Projects/deepCube/mydata/raw/v0427_01.mp4