#!/usr/bin/env bash
# Mux the lossless render segments + score into delivery files.
# BT.709 matrix + tags so players don't shift colours. 2-pass to keep files under GitHub's 100 MB limit.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p output build
COL="-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"
enc () {  # $1 = scale filter prefix, $2 = bitrate, $3 = level, $4 = audio bitrate, $5 = out
  VF="${1}scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p"
  ffmpeg -y -loglevel error -f concat -safe 0 -i build/segments.txt -vf "$VF" \
    -c:v libx264 -preset slow -tune film -b:v "$2" -maxrate "$(( ${2%M} * 3 / 2 ))M" -bufsize "$(( ${2%M} * 2 ))M" \
    -profile:v high -level "$3" -g 48 $COL -pass 1 -passlogfile build/x264_$3 -an -f mp4 /dev/null
  ffmpeg -y -loglevel error -f concat -safe 0 -i build/segments.txt -i build/score.wav -vf "$VF" \
    -c:v libx264 -preset slow -tune film -b:v "$2" -maxrate "$(( ${2%M} * 3 / 2 ))M" -bufsize "$(( ${2%M} * 2 ))M" \
    -profile:v high -level "$3" -g 48 $COL -pass 2 -passlogfile build/x264_$3 \
    -c:a aac -b:a "$4" -shortest -movflags +faststart "$5"
}
enc "" 38M 5.1 320k output/PharmNight2026_CinemaSoiree_GrandPremiere_4K.mp4 &
enc "scale=1920:1080:flags=lanczos," 14M 4.2 256k output/PharmNight2026_CinemaSoiree_GrandPremiere_1080p.mp4 &
wait
cp build/score.wav output/PharmNight2026_CinemaSoiree_score.wav
ls -la output
