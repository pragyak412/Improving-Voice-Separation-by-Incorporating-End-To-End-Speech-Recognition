#!/bin/bash
# download AVSpeech dataset or any dataset whose csv is in AVSpeech format, then convert the mp3 files into wav file format
# and clean the directory to remove corrupted files
#
# usage:      download.sh <path-to-csv-file> <output-directory> <output-directory-wav>
# env vars:   njobs=<n> download={0|1} wavfileconvert={0|1} cleanfile={0|1} numdownload=<n>
# dependency: GNU Parallel, youtube-dl, FFmpeg
# 
#
# examples:
#   ./download.sh avspeech_train.csv data/trainmp3 data/trainwav
#   njobs=12 wavfileconvert=0 ./download.sh avspeech_train.csv data/trainmp3 data/trainwav


#to mention how many to jobs will be done in parallel
njobs="${njobs:-0}" 
#make it 0 if dont want to download from csv
download="${download:-1}"
#make it 0 when no need to convert to wav file
wavfileconvert="${wavfileconvert:-1}"
#make it 0 when cleaning is not required
cleanfile="${cleanfile:-1}"
#number of files to be downloaded
numdownload="${numdownload:-100000}"

#cleaning is defined for wav files, if you want to clean mp3 change the input directory
if [[ "$wavfileconvert" -le 0 ]]; then
  cleanfile="${cleanfile:-0}"
fi
#check if all the arguements are given
if [[ $# -ne 3 ]]; then
	echo "usage: $(basename "$0") <path-to-csv-file> <output-directory-mp3> <output-directory-wav>"
	exit 1
fi

# csv file path
csvfile="$1"

#outputdirectory to store mp3 format
outdirmp3="$2"

#outputdirectory to store wav format
outdirwav="$3"

#corrupt file to be found while cleaning
corruptfile="corruptfilelist.txt"

# command definitions
youtubedl="youtube-dl --quiet --no-warnings"
ffmpeg="ffmpeg -y -loglevel error"
mv="mv -f"
rm="rm -f"
mkdir="mkdir -p"
parallel="parallel --no-notice"


format_seconds() {
	printf "%010.6f\n" "$1"
}

#download youtube video in form of mp3(audio file) code
download_audio() {
  # parse csv
	IFS=',' read -r ytid start end x y <<< "$1"

  id="${ytid}_$(format_seconds "$start")-$(format_seconds "$end")"
	filename="$id.mp3"

  #if file is already downloaded
	if test -f "$outdirmp3/$filename"; then
		return 1
	fi

  duration="$(bc <<< "$end - $start")"
	t="$(date +%s.%N)"

  #downloading audio using youtubedl and ffmpeg
  $ffmpeg -hide_banner -loglevel panic $(youtube-dl -g -f bestaudio --extract-audio --audio-format mp3 --external-downloader aria2c --external-downloader-args '"-j 8 -s 8 -x 8 -k 5M"' https://www.youtube.com/watch?v="$ytid" | sed  's/.*/-ss '"$start"' -i &/') -t "$duration" -c:a libmp3lame "$outdirmp3/$filename"
}

#coversion to wav file code
convert_audio(){
    IFS=',' read -r file <<<"$1"
    #converting video using ffmpeg
    $ffmpeg -i "$outdirmp3/$file" -acodec pcm_s16le -ac 1 -ar "$freq" "$outdirwav/${file%.mp3}.wav";
}

#cleaning refers to deleting files which cant be loaed by the model
clean_file(){
    python cleanWav.py "$outdirwav" "$corruptfile"
}

export youtubedl ffmpeg mv rm mkdir outdirmp3 outdirwav
export -f  format_seconds download_audio convert_audio clean_file

#calling when download is set to 1
if [[ "$download" -le 1 ]]; then
  mkdir "$outdirmp3"
  trap 'printf *** download interrupted ***"; exit 2' INT QUIT TERM
  printf "%s \n" "*** download starts ***"
  head -n "$numdownload" "$csvfile" | $parallel -j "$njobs" --timeout 600 download_audio
  printf "%s \n" "*** download ends ***"
fi

#calling when wavfileconvert is set to 1
if [[ "$wavfileconvert" -le 1 ]]; then
  mkdir "$outdirwav"
  trap 'printf *** conversion to wav interrupted ***"; exit 2' INT QUIT TERM
  printf  "%s \n" "*** conversion to wav  starts ***"
  ls "$outdirmp3" | $parallel -j  "$njobs" --timeout 600 convert_audio
  printf "%s \n" "*** conversion to wav ends ***"
fi

#calling when cleanfile set to 1
if [[ "$cleanfile" -le 1 ]]; then
  trap 'printf *** cleaning wav files interrupted ***"; exit 2' INT QUIT TERM
  printf "%s \n" "*** cleaning wav files starts ***"
  clean_file
  xargs rm < "$corruptfile";
  rm "$corruptfile"
  printf "%s \n" "*** cleaning wav files ends ***"
fi
