#!/usr/bin/env bash
# Download the JHU CSSE COVID-19 time-series (public, CC BY 4.0) used by Q-EpiGNN
set -e
mkdir -p data/raw && cd data/raw
B=https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series
for f in time_series_covid19_confirmed_US.csv time_series_covid19_deaths_US.csv time_series_covid19_confirmed_global.csv; do
  curl -L -o $f $B/$f
done
