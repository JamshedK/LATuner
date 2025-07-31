#!/usr/bin/env bash

cd /home/karimnazarovj/LATuner/benchbase/target/benchbase-mysql
BENCHNAME=$1
TIMESTAMP=$2
OUTPUTDIR="$(realpath "$3")"  # Convert to absolute path
OUTPUTLOG="$(realpath "$4")"  # Convert to absolute path

java -jar benchbase.jar -b $BENCHNAME -c config/mysql/sample_${BENCHNAME}_config.xml --execute=true --directory=$OUTPUTDIR > ${OUTPUTLOG}/${BENCHNAME}_${TIMESTAMP}.log