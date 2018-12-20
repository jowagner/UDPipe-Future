#!/bin/sh

while read code; do
  [ "$code" = mix ] && continue

  args=$(grep -o "$code .*" models-chosen/$code-*-last1.cmd | sed 's/^[^ ]* //; s#=../embeddings#=embeddings#')
  [ -z "$args" ] && { echo Skipping $code; continue; }

  args=$(echo "$args" | sed 's/--rnn_layers\(\|_parser\|_tagger\)=[^ ]*//g; s/^ *//; s/ *$//')
  case "$args" in
    --embeddings=*) args="" ;;
    ?*) echo BAD; exit 1;;
    *) echo Ignoring $code; continue;;
  esac
  for reg in --rnn_layers=2\ --rnn_layers_parser=1\ --rnn_layers_tagger=1; do
    qsub -q gpu*@dll* -cwd -p -101 -b y -l gpu=1,gpu_cc_min3.5=1 -j y withcuda venv/bin/python3 ud_parser5h.py ud-2.2-conll18/$code/$code $args $reg
  done
done <langs
