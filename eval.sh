#!/bin/bash

echo "$(date): Starting eval1.sh..."
./eval1.sh
echo "$(date): eval.sh finished (exit code: $?)"

echo "$(date): Starting eval2.sh..."
./eval2.sh
echo "$(date): eval2.sh finished (exit code: $?)"

echo "$(date): Starting eval3.sh..."
./eval2.sh
echo "$(date): eval3.sh finished (exit code: $?)"

echo "$(date): All evaluation scripts completed!"