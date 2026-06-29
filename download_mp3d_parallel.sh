#!/bin/bash
# 并行分段下载 MP3D habitat 数据 (mp3d_habitat.zip)，可重复运行(幂等/续传)
set -u
URL="https://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip"
OUT="/mnt/cpfs/prediction/lyyy/myself/VLN/StreamVLN/data/scene_datasets/mp3d_habitat.zip"
TOTAL=16085306031
N=8
PARTDIR="/mnt/cpfs/prediction/lyyy/myself/VLN/mp3d_parts"
mkdir -p "$PARTDIR"
CHUNK=$((TOTAL / N))

download_part() {
  local i=$1
  local start=$((i * CHUNK))
  local end
  if [ "$i" -eq $((N - 1)) ]; then end=$((TOTAL - 1)); else end=$(((i + 1) * CHUNK - 1)); fi
  local expected=$((end - start + 1))
  local pf="$PARTDIR/part_$i"
  # 续传: 已有部分则从断点继续
  local have=0
  [ -f "$pf" ] && have=$(stat -c %s "$pf")
  if [ "$have" -eq "$expected" ]; then echo "part $i done($have)"; return 0; fi
  local rstart=$((start + have))
  echo "part $i: range $rstart-$end (have $have/$expected)"
  curl -sL --retry 10 --retry-delay 5 --connect-timeout 30 -r ${rstart}-${end} "$URL" >> "$pf"
}

for i in $(seq 0 $((N - 1))); do download_part "$i" & done
wait

# 校验各段
ok=1
for i in $(seq 0 $((N - 1))); do
  start=$((i * CHUNK)); if [ "$i" -eq $((N - 1)) ]; then end=$((TOTAL - 1)); else end=$(((i + 1) * CHUNK - 1)); fi
  expected=$((end - start + 1)); sz=$(stat -c %s "$PARTDIR/part_$i" 2>/dev/null || echo 0)
  if [ "$sz" -ne "$expected" ]; then echo "part $i incomplete: $sz/$expected"; ok=0; fi
done
if [ "$ok" -ne 1 ]; then echo "DOWNLOAD_INCOMPLETE - 重新运行本脚本续传"; exit 1; fi

cat "$PARTDIR"/part_* > "$OUT"
final=$(stat -c %s "$OUT")
echo "merged size=$final expected=$TOTAL"
if [ "$final" -eq "$TOTAL" ]; then echo "MP3D_DOWNLOAD_OK"; rm -rf "$PARTDIR"; else echo "MERGE_SIZE_MISMATCH"; exit 1; fi
