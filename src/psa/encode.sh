#!/usr/bin/env bash
# PSA intro deliverables: 4K60 + 1080p60 vertical (with original sound design), muted copy, WAV, final frame.
set -euo pipefail
cd "$(dirname "$0")/../.."
O=output/psa_intro_9x16
mkdir -p "$O"
B=build_psa
python3 src/psa/audio.py $B/psa_score.wav
# two-pass loudness normalisation to -14 LUFS / -1 dBTP (TikTok / Reels)
J=$(ffmpeg -hide_banner -i $B/psa_score.wav -af loudnorm=I=-14:TP=-1.0:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
g(){ echo "$J" | python3 -c "import sys,json; print(json.load(sys.stdin)['$1'])"; }
ffmpeg -y -loglevel error -i $B/psa_score.wav -af "loudnorm=I=-14:TP=-1.0:LRA=11:measured_I=$(g input_i):measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh):offset=$(g target_offset):linear=true,aresample=48000" -c:a pcm_s16le $B/psa_score_norm.wav
COL="-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"
enc () {  # $1 scale prefix, $2 Mbps, $3 level, $4 out
  VF="${1}scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p"
  X="-c:v libx264 -preset slow -b:v ${2}M -maxrate $(( $2 * 3 / 2 ))M -bufsize $(( $2 * 2 ))M -profile:v high -level $3 -g 120 -r 60 $COL"
  ffmpeg -y -loglevel error -f concat -safe 0 -i $B/segments.txt -vf "$VF" $X -pass 1 -passlogfile $B/x264_$3 -an -f mp4 /dev/null
  ffmpeg -y -loglevel error -f concat -safe 0 -i $B/segments.txt -i $B/psa_score_norm.wav -vf "$VF" $X \
    -pass 2 -passlogfile $B/x264_$3 -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart "$4"
}
enc "" 44 5.2 "$O/PSA_Crest_Intro_2160x3840_60fps.mp4" &
enc "scale=1080:1920:flags=lanczos," 16 4.2 "$O/PSA_Crest_Intro_1080x1920_60fps.mp4" &
wait
ffmpeg -y -loglevel error -i "$O/PSA_Crest_Intro_1080x1920_60fps.mp4" -map 0:v -c:v copy -an -movflags +faststart "$O/PSA_Crest_Intro_1080x1920_60fps_MUTED.mp4"
cp $B/psa_score_norm.wav "$O/PSA_Crest_Intro_sound_design.wav"
ffmpeg -y -loglevel error -sseof -0.02 -i $B/seg_03.mkv -frames:v 1 -pix_fmt rgb24 "$O/final_frame.png"
ls -la "$O"
