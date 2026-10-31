#!/usr/bin/env bash
# 9:16 Reels deliverables from build_vertical/ segments (BT.709, 2-pass, faststart).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p output/reels_9x16
O=output/reels_9x16
COL="-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"
python3 src/audio.py --cinematic build_vertical/score_cinematic.wav
enc () {  # $1 scale prefix, $2 bitrate (M), $3 level, $4 out
  VF="${1}scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p"
  X="-c:v libx264 -preset slow -tune film -b:v ${2}M -maxrate $(( $2 * 3 / 2 ))M -bufsize $(( $2 * 2 ))M -profile:v high -level $3 -g 48 $COL"
  ffmpeg -y -loglevel error -f concat -safe 0 -i build_vertical/segments.txt -vf "$VF" $X -pass 1 -passlogfile build_vertical/x264_$3 -an -f mp4 /dev/null
  ffmpeg -y -loglevel error -f concat -safe 0 -i build_vertical/segments.txt -i build_vertical/score_cinematic.wav -vf "$VF" $X \
    -pass 2 -passlogfile build_vertical/x264_$3 -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart "$4"
}
enc "scale=1080:1920:flags=lanczos," 16 4.2 $O/PharmNight2026_CinemaSoiree_Reels_1080x1920.mp4 &
enc "" 38 5.1 $O/PharmNight2026_CinemaSoiree_Reels_2160x3840_master.mp4 &
wait
# muted copy (identical picture) for adding your own sound in Instagram / CapCut
ffmpeg -y -loglevel error -i $O/PharmNight2026_CinemaSoiree_Reels_1080x1920.mp4 -map 0:v -c:v copy -an -movflags +faststart \
  $O/PharmNight2026_CinemaSoiree_Reels_1080x1920_MUTED.mp4
cp build_vertical/score_cinematic.wav $O/PharmNight2026_CinemaSoiree_Reels_cinematic_score.wav
ffmpeg -y -loglevel error -i build_vertical/seg_03.mkv -vf "select=eq(n\,107)" -frames:v 1 -pix_fmt rgb24 $O/final_frame_vertical.png
ls -la $O
