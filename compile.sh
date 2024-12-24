#!/bin/bash

myloc="$(dirname "$0")"

python "$myloc/text_encoder_1/compile.py"
python "$myloc/text_encoder_2/compile.py" -m 32
python "$myloc/transformer/compile.py" -hh 256 -w 256 -m 32
python "$myloc/decoder/compile.py" -hh 256 -w 256
