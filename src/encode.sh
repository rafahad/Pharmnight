#!/usr/bin/env bash
# Mux the lossless render segments + score into delivery files (BT.709 tagged, so players don't shift colours).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p output
COL="-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"
ffmpeg -y -loglevel error -f concat -safe 0 -i build/segments.txt -i build/score.wav \
  -vf "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p" \
  -c:v libx264 -preset slow -crf 15 -tune film -profile:v high -level 5.1 -g 48 $COL \
  -c:a aac -b:a 320k -shortest -movflags +faststart \
  output/PharmNight2026_CinemaSoiree_GrandPremiere_4K.mp4
ffmpeg -y -loglevel error -f concat -safe 0 -i build/segments.txt -i build/score.wav \
  -vf "scale=1920:1080:flags=lanczos+accurate_rnd+full_chroma_int:out_color_matrix=bt709:out_range=tv,format=yuv420p" \
  -c:v libx264 -preset slow -crf 16 -tune film -profile:v high -level 4.2 -g 48 $COL \
  -c:a aac -b:a 256k -shortest -movflags +faststart \
  output/PharmNight2026_CinemaSoiree_GrandPremiere_1080p.mp4
# silent 4K version for editors who will add their own music
ffmpeg -y -loglevel error -f concat -safe 0 -i build/segments.txt \
  -vf "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p" \
  -c:v libx264 -preset slow -crf 15 -tune film -profile:v high -level 5.1 -g 48 $COL -an -movflags +faststart \
  output/PharmNight2026_CinemaSoiree_GrandPremiere_4K_silent.mp4
cp build/score.wav output/PharmNight2026_CinemaSoiree_score.wav
ls -la output
