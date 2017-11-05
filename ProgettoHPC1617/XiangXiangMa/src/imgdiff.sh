#!/bin/sh

PERCENT="25"
FUZZ="10"

[ "$#" != "2" ] && echo "Syntax: $0 IMAGE1 IMAGE2" >&2 && exit 2
IMG1="$1"
IMG2="$2"

SIZE1=$(stat -c '%s' "$IMG1")
SIZE2=$(stat -c '%s' "$IMG2")

if [ "$SIZE1" -lt "$SIZE2" ]
then
	BIG="$IMG2"
	SMALL="$IMG1"
else
	BIG="$IMG1"
	SMALL="$IMG2"
fi

#echo "1) Scaling both images to $PERCENT% of smaller image"
#echo "2) Counting different pixels (color distance > $FUZZ%)"

SIZE=$(identify -format '%wx%h' "$SMALL")

W=$(echo "$SIZE" | cut -dx -f1)
H=$(echo "$SIZE" | cut -dx -f2)

W=$(( ($W * $PERCENT) / 100 ))
H=$(( ($H * $PERCENT) / 100 ))

DIFF=$(
convert "$SMALL" "$BIG" -resize "$W"x"$H"\! MIFF:- | compare -metric AE -fuzz "$FUZZ%" - null: 2>&1
)
[ "$?" != "0" ] && echo "$DIFF" >&2

DIFF_RATIO=$(awk "BEGIN {printf \"%.3f\n\", ($DIFF / ($W*$H))*100 }")
echo "pixel difference: $DIFF_RATIO%"

if [ "$DIFF" = 0 ]
then
	echo "OK"
	exit 0
else
	echo "NOK"
	exit 1
fi
